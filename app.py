import collections, collections.abc
collections.MutableSequence = collections.abc.MutableSequence
collections.MutableMapping  = collections.abc.MutableMapping
collections.Mapping         = collections.abc.Mapping
collections.Callable        = collections.abc.Callable
collections.Iterator        = collections.abc.Iterator
collections.Iterable        = collections.abc.Iterable

from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile, os, sys

app = Flask(__name__)
CORS(app)

# chord_recognition.py가 있는 폴더를 경로에 추가
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHORD_DIR = BASE_DIR
sys.path.insert(0, CHORD_DIR)

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

def snap_to_grid(chords, bpm, grid_offset_sec):
    beat_sec      = 60.0 / bpm
    half_beat_sec = beat_sec / 2.0
    bar_sec       = beat_sec * 4
    result        = []
    for c in chords:
        cs, ce, cl = c['start'], c['end'], c['label']
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

def run_chord_recognition(audio_path):
    """music-x-lab chord_recognition 함수 호출"""
    from chord_recognition import chord_recognition
    import uuid

    tmp_lab = f'/tmp/{uuid.uuid4()}.lab'
    try:
        chord_recognition(audio_path, tmp_lab, chord_dict_name='submission')
        results = []
        with open(tmp_lab) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    start = float(parts[0])
                    end   = float(parts[1])
                    label = parts[2]
                    results.append({'start': start, 'end': end, 'label': label})
        return results
    finally:
        if os.path.exists(tmp_lab):
            os.unlink(tmp_lab)

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok', 'engine': 'music-x-lab'})

@app.route('/analyze_grid', methods=['POST'])
def analyze_grid():
    if 'audio' not in request.files:
        return jsonify({'error': 'audio file required'}), 400

    f               = request.files['audio']
    bpm             = float(request.form.get('bpm', 128))
    grid_offset_sec = float(request.form.get('gridOffset', 0.0))

    print(f'[분석] BPM={bpm}, gridOffset={grid_offset_sec}s')

    suffix   = os.path.splitext(f.filename)[1] or '.mp3'
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name

        raw_list   = run_chord_recognition(tmp_path)
        print(f'[엔진] {len(raw_list)}개 감지')

        smoothed   = smooth_chords(raw_list, min_duration=0.5)
        snapped    = snap_to_grid(smoothed, bpm, grid_offset_sec)
        compressed = compress_chords(snapped)

        beat_sec = 60.0 / bpm
        bar_sec  = beat_sec * 4
        result   = []
        for item in compressed:
            rel  = item['start'] - grid_offset_sec
            bar  = int(rel / bar_sec) + 1
            beat = round(((rel % bar_sec) / beat_sec + 1) * 2) / 2
            result.append({**item, 'bar': bar, 'beat': beat})

        print(f'[완료] {len(result)}개 코드')
        return jsonify({'chords': result, 'bpm': bpm, 'gridOffset': grid_offset_sec})

    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except: pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
