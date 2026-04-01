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
import librosa
import soundfile as sf
import madmom
from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Madmom 프로세서 (서버 시작 시 1회만 로드)
_featproc = CNNChordFeatureProcessor()
_recproc  = CRFChordRecognitionProcessor()


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
    """
    각 박자 끝 cut_ratio(12%) 묵음 처리 → 퀀타이즈 효과
    """
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
    """
    Madmom 결과(초)를 0.5박 단위 그리드에 스냅
    → 마디/박자 정보 포함하여 반환
    """
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


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok'})


@app.route('/analyze_grid', methods=['POST'])
def analyze_grid():
    """
    파이프라인:
    1. 오디오 + BPM + gridOffset 수신
    2. 박자 끝 12% 묵음 처리 (퀀타이즈)
    3. Madmom 코드 분석
    4. 그리드 스냅 → 마디/박자별 코드 반환
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'audio file required'}), 400

    f               = request.files['audio']
    bpm             = float(request.form.get('bpm', 128))
    grid_offset_sec = float(request.form.get('gridOffset', 0.0))

    print(f'[분석] BPM={bpm}, gridOffset={grid_offset_sec}s, 파일={f.filename}')

    suffix   = os.path.splitext(f.filename)[1] or '.wav'
    tmp_path = resampled_path = None

    try:
        # 오디오 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name

        # 오디오 로드
        audio, sr = librosa.load(tmp_path, sr=None, mono=True)
        print(f'[오디오] sr={sr}, 길이={len(audio)/sr:.1f}초')

        # 박자 끝 12% 묵음 처리
        audio = apply_grid_cut(audio, sr, bpm, grid_offset_sec)

        # 임시 wav 저장 후 Madmom 분석
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp2:
            sf.write(tmp2.name, audio, sr)
            resampled_path = tmp2.name

        features = _featproc(resampled_path)
        chords   = _recproc(features)
        print(f'[Madmom] {len(chords)}개 감지')

        # 그리드 스냅 + 압축
        snapped    = snap_to_grid(chords, bpm, grid_offset_sec)
        raw        = [{'start': s['start'], 'end': s['end'], 'label': s['label']} for s in snapped]
        compressed = compress_chords(raw)

        # 마디/박자 재계산
        bar_sec = (60.0 / bpm) * 4
        result  = []
        for item in compressed:
            rel  = item['start'] - grid_offset_sec
            bar  = int(rel / bar_sec) + 1
            beat = round(((rel % bar_sec) / (60.0 / bpm) + 1) * 2) / 2
            result.append({**item, 'bar': bar, 'beat': beat})

        print(f'[완료] {len(result)}개 코드 반환')
        return jsonify({'chords': result, 'bpm': bpm, 'gridOffset': grid_offset_sec})

    finally:
        for p in [tmp_path, resampled_path]:
            if p:
                try: os.unlink(p)
                except: pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
