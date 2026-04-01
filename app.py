import collections
import collections.abc
collections.MutableSequence = collections.abc.MutableSequence
collections.MutableMapping  = collections.abc.MutableMapping
collections.Mapping         = collections.abc.Mapping
collections.Callable        = collections.abc.Callable
collections.Iterator        = collections.abc.Iterator
collections.Iterable        = collections.abc.Iterable

from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os
import numpy as np
import librosa
import soundfile as sf
import madmom
from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Madmom 프로세서 (서버 시작 시 1회만 로드)
_featproc = CNNChordFeatureProcessor()
_recproc  = CRFChordRecognitionProcessor()

CHUNK_SEC = 30  # 청크 길이 (초)
MAX_WORKERS = 4  # 병렬 처리 워커 수


def compress_chords(chords):
    """연속된 동일 코드 병합"""
    if not chords:
        return []
    result = [chords[0].copy()]
    for item in chords[1:]:
        if item['label'] == result[-1]['label']:
            result[-1]['end'] = item['end']
        else:
            result.append(item.copy())
    return result


def apply_grid_cut(audio, sr, bpm, grid_offset_sec, cut_ratio=0.12):
    """박자 끝 12% 묵음 처리 → 퀀타이즈 효과"""
    beat_sec      = 60.0 / bpm
    cut_samples   = int(beat_sec * cut_ratio * sr)
    audio         = audio.copy()
    total_samples = len(audio)
    beat_idx      = int(grid_offset_sec * sr)

    while beat_idx < total_samples:
        beat_end   = beat_idx + int(beat_sec * sr)
        mute_start = max(0, beat_end - cut_samples)
        mute_end   = min(beat_end, total_samples)
        audio[mute_start:mute_end] = 0.0
        beat_idx = beat_end

    return audio


def snap_to_grid(chords, bpm, grid_offset_sec):
    """Madmom 결과(초)를 0.5박 단위 그리드에 스냅"""
    beat_sec      = 60.0 / bpm
    half_beat_sec = beat_sec / 2.0
    bar_sec       = beat_sec * 4
    result        = []

    for (cs, ce, cl) in chords:
        rel_s = cs - grid_offset_sec
        rel_e = ce - grid_offset_sec

        snapped_s = round(rel_s / half_beat_sec) * half_beat_sec
        snapped_e = round(rel_e / half_beat_sec) * half_beat_sec

        bar  = int(snapped_s / bar_sec) + 1
        beat = round(((snapped_s % bar_sec) / beat_sec + 1) * 2) / 2

        result.append({
            'start': round(grid_offset_sec + snapped_s, 3),
            'end':   round(grid_offset_sec + snapped_e, 3),
            'label': cl,
            'bar':   bar,
            'beat':  beat
        })

    return result


def analyze_chunk(chunk_audio, sr, chunk_start_sec):
    """단일 청크 Madmom 분석 (병렬 처리용)"""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            sf.write(tmp.name, chunk_audio, sr)
            tmp_path = tmp.name

        features = _featproc(tmp_path)
        chords   = _recproc(features)

        # 청크 시작 시간 오프셋 적용
        return [(cs + chunk_start_sec, ce + chunk_start_sec, cl) for (cs, ce, cl) in chords]

    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except: pass


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok'})


@app.route('/analyze_grid', methods=['POST'])
def analyze_grid():
    """
    파이프라인:
    1. 오디오 + BPM + gridOffset 수신
    2. 박자 끝 12% 묵음 처리 (퀀타이즈)
    3. 청크 분할 → 병렬 Madmom 분석
    4. 결과 병합 + 그리드 스냅 → 반환
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'audio file required'}), 400

    f               = request.files['audio']
    bpm             = float(request.form.get('bpm', 128))
    grid_offset_sec = float(request.form.get('gridOffset', 0.0))

    print(f'[분석] BPM={bpm}, gridOffset={grid_offset_sec}s, 파일={f.filename}')

    suffix   = os.path.splitext(f.filename)[1] or '.mp3'
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name

        # 오디오 로드 (44100Hz 고정 - Madmom 요구사항)
        audio, sr = librosa.load(tmp_path, sr=44100, mono=True)
        total_sec = len(audio) / sr
        print(f'[오디오] sr={sr}, 길이={total_sec:.1f}초')

        # 박자 끝 12% 묵음 처리
        audio = apply_grid_cut(audio, sr, bpm, grid_offset_sec)

        # 청크 분할
        chunk_samples = int(CHUNK_SEC * sr)
        chunks = []
        start = 0
        while start < len(audio):
            end = min(start + chunk_samples, len(audio))
            chunks.append((audio[start:end], start / sr))
            start = end

        print(f'[청크] {len(chunks)}개 청크 병렬 분석 시작')

        # 병렬 Madmom 분석
        all_chords = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(analyze_chunk, chunk_audio, sr, chunk_start): i
                for i, (chunk_audio, chunk_start) in enumerate(chunks)
            }
            results = {}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as e:
                    print(f'[청크{i}] 실패: {e}')
                    results[i] = []

        # 순서대로 병합
        for i in sorted(results.keys()):
            all_chords.extend(results[i])

        # 시간순 정렬
        all_chords.sort(key=lambda x: x[0])
        print(f'[병합] 총 {len(all_chords)}개 코드')

        # 그리드 스냅
        snapped    = snap_to_grid(all_chords, bpm, grid_offset_sec)
        raw        = [{'start': s['start'], 'end': s['end'], 'label': s['label']} for s in snapped]
        compressed = compress_chords(raw)

        # 마디/박자 재계산
        beat_sec = 60.0 / bpm
        bar_sec  = beat_sec * 4
        result   = []
        for item in compressed:
            rel  = item['start'] - grid_offset_sec
            bar  = int(rel / bar_sec) + 1
            beat = round(((rel % bar_sec) / beat_sec + 1) * 2) / 2
            result.append({**item, 'bar': bar, 'beat': beat})

        print(f'[완료] {len(result)}개 코드 반환')
        return jsonify({'chords': result, 'bpm': bpm, 'gridOffset': grid_offset_sec})

    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except: pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
