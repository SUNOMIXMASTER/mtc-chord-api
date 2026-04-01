import collections
import collections.abc
collections.MutableSequence = collections.abc.MutableSequence
collections.MutableMapping = collections.abc.MutableMapping
collections.Mapping = collections.abc.Mapping
collections.Callable = collections.abc.Callable
collections.Iterator = collections.abc.Iterator
collections.Iterable = collections.abc.Iterable

from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os
import numpy as np
import madmom
from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

_featproc = CNNChordFeatureProcessor()
_recproc  = CRFChordRecognitionProcessor()

GEMINI_API_KEY_ENV = os.environ.get('GEMINI_API_KEY', '')

def compress_chords(raw):
    """연속된 동일 코드 병합"""
    if not raw:
        return []
    compressed = [raw[0].copy()]
    for item in raw[1:]:
        if item['label'] == compressed[-1]['label']:
            compressed[-1]['end'] = item['end']
        else:
            compressed.append(item.copy())
    return compressed

def apply_grid_resample(audio, sr, bpm, grid_offset_sec, cut_ratio=0.12):
    """
    그리드 기준으로 각 박자 끝 cut_ratio(12%) 묵음 처리
    - bpm: 곡 BPM
    - grid_offset_sec: 마디1 beat1 시작 시간(초)
    - cut_ratio: 박자 끝 묵음 비율 (0.12 = 12%)
    """
    beat_sec     = 60.0 / bpm
    cut_samples  = int(beat_sec * cut_ratio * sr)
    audio        = audio.copy()
    total_samples = len(audio)
    offset_samples = int(grid_offset_sec * sr)

    beat_idx = offset_samples
    while beat_idx < total_samples:
        beat_end   = beat_idx + int(beat_sec * sr)
        mute_start = beat_end - cut_samples
        mute_end   = min(beat_end, total_samples)

        if mute_start < total_samples:
            audio[mute_start:mute_end] = 0.0

        beat_idx = beat_end

    return audio

def snap_to_grid(chords, bpm, grid_offset_sec):
    """
    Madmom 결과(초)를 0.5박 단위 그리드에 스냅
    """
    beat_sec      = 60.0 / bpm
    half_beat_sec = beat_sec / 2.0
    bar_sec       = beat_sec * 4

    snapped = []
    for (cs, ce, cl) in chords:
        rel_start = cs - grid_offset_sec
        rel_end   = ce - grid_offset_sec

        snapped_start = round(rel_start / half_beat_sec) * half_beat_sec
        snapped_end   = round(rel_end   / half_beat_sec) * half_beat_sec

        bar  = int(snapped_start / bar_sec) + 1
        beat = (snapped_start % bar_sec) / beat_sec + 1

        snapped.append({
            'start': round(grid_offset_sec + snapped_start, 3),
            'end':   round(grid_offset_sec + snapped_end,   3),
            'label': cl,
            'bar':   bar,
            'beat':  round(beat * 2) / 2
        })

    return snapped

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok', 'geminiKey': GEMINI_API_KEY_ENV})

@app.route('/analyze', methods=['POST'])
def analyze():
    """기존 엔드포인트 유지"""
    if 'audio' not in request.files:
        return jsonify({'error': 'audio file required'}), 400

    f            = request.files['audio']
    chunk_offset = float(request.form.get('chunkOffset', 0.0))
    print(f'[청크] chunkOffset={chunk_offset:.2f}s, 파일={f.filename}')

    suffix = os.path.splitext(f.filename)[1] or '.wav'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        features = _featproc(tmp_path)
        chords   = _recproc(features)

        silence_count = sum(1 for c in chords if c[2] in ('N', 'X', 'N/A'))
        print(f'[Madmom] {len(chords)}개 감지 (코드={len(chords)-silence_count}, N/침묵={silence_count})')

        raw = []
        for (cs, ce, cl) in chords:
            raw.append({
                'start': round(chunk_offset + cs, 3),
                'end':   round(chunk_offset + ce, 3),
                'label': cl
            })

        compressed = compress_chords(raw)
        print(f'[청크 완료] raw {len(raw)}개 → 압축 {len(compressed)}개 반환')
        return jsonify({'raw': compressed, 'engine': 'madmom'})
    finally:
        os.unlink(tmp_path)

@app.route('/analyze_grid', methods=['POST'])
def analyze_grid():
    """
    새 파이프라인:
    1. 오디오 + BPM + gridOffset 수신
    2. 그리드 기준 리샘플링 (12% 묵음)
    3. Madmom 코드 분석 1회
    4. 그리드 스냅 → 마디/박자별 코드 반환
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'audio file required'}), 400

    f               = request.files['audio']
    bpm             = float(request.form.get('bpm', 128))
    grid_offset_sec = float(request.form.get('gridOffset', 0.0))

    print(f'[그리드 분석] BPM={bpm}, gridOffset={grid_offset_sec}s, 파일={f.filename}')

    suffix = os.path.splitext(f.filename)[1] or '.wav'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    resampled_path = None
    try:
        import librosa
        import soundfile as sf

        # 오디오 로드
        audio, sr = librosa.load(tmp_path, sr=None, mono=True)
        print(f'[오디오] sr={sr}, 길이={len(audio)/sr:.2f}초')

        # 그리드 기준 리샘플링 (12% 묵음)
        resampled = apply_grid_resample(audio, sr, bpm, grid_offset_sec, cut_ratio=0.12)
        print(f'[리샘플링] 완료')

        # 리샘플링된 오디오를 임시 wav로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp2:
            sf.write(tmp2.name, resampled, sr)
            resampled_path = tmp2.name

        # Madmom 코드 분석
        features = _featproc(resampled_path)
        chords   = _recproc(features)

        silence_count = sum(1 for c in chords if c[2] in ('N', 'X', 'N/A'))
        print(f'[Madmom] {len(chords)}개 감지 (코드={len(chords)-silence_count}, N/침묵={silence_count})')

        # 그리드 스냅
        snapped = snap_to_grid(chords, bpm, grid_offset_sec)

        # 압축
        raw = [{'start': s['start'], 'end': s['end'], 'label': s['label']} for s in snapped]
        compressed = compress_chords(raw)

        # 마디/박자 정보 재계산
        beat_sec = 60.0 / bpm
        bar_sec  = beat_sec * 4
        result = []
        for item in compressed:
            rel  = item['start'] - grid_offset_sec
            bar  = int(rel / bar_sec) + 1
            beat = (rel % bar_sec) / beat_sec + 1
            result.append({
                'start': item['start'],
                'end':   item['end'],
                'label': item['label'],
                'bar':   bar,
                'beat':  round(beat * 2) / 2
            })

        print(f'[그리드 완료] {len(result)}개 코드 반환')
        return jsonify({
            'chords':     result,
            'engine':     'madmom_grid',
            'bpm':        bpm,
            'gridOffset': grid_offset_sec
        })

    finally:
        os.unlink(tmp_path)
        if resampled_path:
            try:
                os.unlink(resampled_path)
            except:
                pass

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
