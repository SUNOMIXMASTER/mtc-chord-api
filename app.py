import collections, collections.abc
collections.MutableSequence = collections.abc.MutableSequence
collections.MutableMapping  = collections.abc.MutableMapping
collections.Mapping         = collections.abc.Mapping
collections.Callable        = collections.abc.Callable
collections.Iterator        = collections.abc.Iterator
collections.Iterable        = collections.abc.Iterable

from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile, os, numpy as np, librosa

app = Flask(__name__)
CORS(app)

# ── 코드 템플릿 (경량 엔진) ──
CHORD_TEMPLATES = {}
roots = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
maj_intervals = [0,4,7]
min_intervals = [0,3,7]

for i, root in enumerate(roots):
    maj = np.zeros(12); [maj.__setitem__((i+x)%12, 1) for x in maj_intervals]
    min_ = np.zeros(12); [min_.__setitem__((i+x)%12, 1) for x in min_intervals]
    CHORD_TEMPLATES[root] = maj / np.linalg.norm(maj)
    CHORD_TEMPLATES[f'{root}m'] = min_ / np.linalg.norm(min_)

CHORD_TEMPLATES['N'] = np.zeros(12)


def compress_chords(chords):
    if not chords: return []
    result = [chords[0].copy()]
    for item in chords[1:]:
        if item['label'] == result[-1]['label']:
            result[-1]['end'] = item['end']
        else:
            result.append(item.copy())
    return result


def smooth_chords(chords, min_duration=0.5):
    filtered = [c for c in chords if (c['end'] - c['start']) >= min_duration]
    if not filtered: return chords
    merged = [filtered[0].copy()]
    for c in filtered[1:]:
        if c['label'] == merged[-1]['label']:
            merged[-1]['end'] = c['end']
        else:
            merged.append(c.copy())
    return merged


def run_chord_engine(audio_path, hop_length=512):
    """크로마그램 기반 경량 코드 인식"""
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # 크로마그램 추출
    chroma = librosa.feature.chroma_cqt(
        y=y, sr=sr, hop_length=hop_length,
        bins_per_octave=24
    )  # shape: (12, frames)

    fps = sr / hop_length
    n_frames = chroma.shape[1]

    # 각 프레임별 코드 판별
    chord_seq = []
    for f in range(n_frames):
        frame = chroma[:, f]
        norm = np.linalg.norm(frame)
        if norm < 0.1:
            chord_seq.append('N')
            continue
        frame = frame / norm

        best_chord = 'N'
        best_score = 0.3  # 최소 임계값
        for name, tmpl in CHORD_TEMPLATES.items():
            if name == 'N': continue
            score = float(np.dot(frame, tmpl))
            if score > best_score:
                best_score = score
                best_chord = name

        chord_seq.append(best_chord)

    # 연속 동일 코드 병합 → (start, end, label)
    results = []
    prev = chord_seq[0]
    start_f = 0
    for f, c in enumerate(chord_seq[1:], 1):
        if c != prev:
            results.append((start_f / fps, f / fps, prev))
            prev = c
            start_f = f
    results.append((start_f / fps, n_frames / fps, prev))
    return results


def snap_to_grid(chords, bpm, grid_offset_sec):
    beat_sec      = 60.0 / bpm
    half_beat_sec = beat_sec / 2.0
    bar_sec       = beat_sec * 4
    result        = []
    for (cs, ce, cl) in chords:
        rel_s     = cs - grid_offset_sec
        rel_e     = ce - grid_offset_sec
        snapped_s = round(rel_s / half_beat_sec) * half_beat_sec
        snapped_e = round(rel_e / half_beat_sec) * half_beat_sec
        if snapped_e <= snapped_s:
            snapped_e = snapped_s + half_beat_sec
        bar  = int(snapped_s / bar_sec) + 1
        beat = round(((snapped_s % bar_sec) / beat_sec + 1) * 2) / 2
        result.append({
            'start': round(grid_offset_sec + snapped_s, 3),
            'end':   round(grid_offset_sec + snapped_e, 3),
            'label': cl, 'bar': bar, 'beat': beat
        })
    return result


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok', 'engine': 'chroma'})


@app.route('/analyze_grid', methods=['POST'])
def analyze_grid():
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

        raw_chords = run_chord_engine(tmp_path)
        print(f'[엔진] {len(raw_chords)}개 감지')

        raw_list = [{'start': s, 'end': e, 'label': l} for s, e, l in raw_chords]
        smoothed = smooth_chords(raw_list, min_duration=0.5)

        tuples     = [(c['start'], c['end'], c['label']) for c in smoothed]
        snapped    = snap_to_grid(tuples, bpm, grid_offset_sec)
        flat       = [{'start': s['start'], 'end': s['end'], 'label': s['label']} for s in snapped]
        compressed = compress_chords(flat)

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
