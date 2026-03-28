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
import numpy as np
import tempfile
import os
import librosa
import madmom
from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# 전역 프로세서 (한 번만 로드)
_featproc = CNNChordFeatureProcessor()
_recproc  = CRFChordRecognitionProcessor()

# ── 코드 인식 (Madmom CNN) ──
def analyze_chords_madmom(audio_path, bpm, key_num, is_minor, grid_offset=0.0):
    try:
        features = _featproc(audio_path)
        chords   = _recproc(features)

        print(f'[Madmom] 원시 코드 {len(chords)}개 감지')
        for c in chords[:10]:
            print(f'  {c[0]:.2f}~{c[1]:.2f}: {c[2]}')

        beat_sec   = 60.0 / bpm
        bar_sec    = beat_sec * 4
        duration   = librosa.get_duration(path=audio_path)
        total_bars = int(np.ceil((duration - grid_offset) / bar_sec))

        NOTE_MAP = {'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,
                    'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11}

        MAJOR_ROMAN = {0:'I',1:'bII',2:'II',3:'bIII',4:'III',5:'IV',
                       6:'#IV',7:'V',8:'bVI',9:'VI',10:'bVII',11:'VII'}
        MINOR_ROMAN = {0:'i',1:'bII',2:'ii',3:'III',4:'#III',5:'iv',
                       6:'#iv',7:'v',8:'VI',9:'vi',10:'VII',11:'#VII'}

        def chord_label_to_roman(label, key_num, is_minor):
            if not label or label in ('N', 'X', 'N/A'):
                return None
            root_str = label.split(':')[0]
            rn = NOTE_MAP.get(root_str)
            if rn is None:
                return None
            interval = (rn - key_num + 12) % 12
            return (MINOR_ROMAN if is_minor else MAJOR_ROMAN).get(interval)

        def get_dominant_chord_in_range(start, end):
            best_label = None
            best_duration = 0
            for (cs, ce, cl) in chords:
                if cl in ('N', 'X', 'N/A'): continue
                overlap_start = max(cs, start)
                overlap_end = min(ce, end)
                if overlap_end > overlap_start:
                    dur = overlap_end - overlap_start
                    if dur > best_duration:
                        best_duration = dur
                        best_label = cl
            return best_label

        results = []
        for bar_idx in range(total_bars):
            bar_start = grid_offset + bar_idx * bar_sec
            beat_romans = []

            for beat in range(4):
                beat_start = bar_start + beat * beat_sec
                beat_end = beat_start + beat_sec
                dominant_label = get_dominant_chord_in_range(beat_start, beat_end)
                roman = chord_label_to_roman(dominant_label, key_num, is_minor)
                beat_romans.append(roman)

            valid = [r for r in beat_romans if r]
            if len(valid) < 1:
                continue

            for b in range(4):
                if not beat_romans[b]:
                    if b > 0 and beat_romans[b-1]:
                        beat_romans[b] = beat_romans[b-1]
                    elif valid:
                        beat_romans[b] = valid[0]

            pattern = ' - '.join(beat_romans[:4])
            results.append({
                'time': bar_start,
                'chord': pattern,
                'source': 'madmom'
            })

            if bar_idx < 10:
                print(f'[Madmom] Bar {bar_idx+1}: {pattern}')

        return results

    except Exception as e:
        import traceback
        print(f'[Madmom ERROR] {e}')
        traceback.print_exc()
        return []


# ── 폴백: librosa ──
def analyze_chords_librosa(audio_path, bpm, key_num, is_minor, grid_offset=0.0):
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)

    beat_sec  = 60.0 / bpm
    bar_sec   = beat_sec * 4
    hop_sec   = 512 / 22050
    total_bars = int(np.ceil((len(y)/sr - grid_offset) / bar_sec))

    MAJOR_ROMAN = {0:'I',2:'II',4:'III',5:'IV',7:'V',9:'VI',11:'VII'}
    MINOR_ROMAN = {0:'i',3:'III',5:'iv',7:'v',8:'VI',10:'VII'}

    templates = {
        'maj': np.array([1,0,0,0,1,0,0,1,0,0,0,0], dtype=float),
        'min': np.array([1,0,0,1,0,0,0,1,0,0,0,0], dtype=float),
    }

    candidates = []
    if not is_minor:
        for iv, t, r in [(0,'maj','I'),(2,'min','ii'),(4,'min','iii'),(5,'maj','IV'),(7,'maj','V'),(9,'min','vi'),(11,'maj','VII')]:
            candidates.append({'root':(key_num+iv)%12,'tmpl':templates[t],'roman':r})
    else:
        for iv, t, r in [(0,'min','i'),(3,'maj','III'),(5,'min','iv'),(7,'min','v'),(8,'maj','VI'),(10,'maj','VII')]:
            candidates.append({'root':(key_num+iv)%12,'tmpl':templates[t],'roman':r})

    results = []
    for bar_idx in range(total_bars):
        bar_start = grid_offset + bar_idx * bar_sec
        beat_romans = []

        for beat in range(4):
            t = bar_start + beat * beat_sec + beat_sec * 0.65
            frame = min(int(t / hop_sec), chroma.shape[1]-1)
            cv = np.roll(chroma[:, frame], -key_num)

            best_score, best_roman = -np.inf, None
            for c in candidates:
                score = float(np.dot(cv, np.roll(c['tmpl'], c['root'] - key_num)))
                if score > best_score:
                    best_score = score
                    best_roman = c['roman']
            beat_romans.append(best_roman)

        valid = [r for r in beat_romans if r]
        if len(valid) < 2:
            continue
        for b in range(4):
            if not beat_romans[b]:
                beat_romans[b] = beat_romans[b-1] if b > 0 else valid[0]

        results.append({'time': bar_start, 'chord': ' - '.join(beat_romans[:4]), 'source': 'librosa'})

    return results


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok'})


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'audio file required'}), 400

    f        = request.files['audio']
    bpm      = float(request.form.get('bpm', 128))
    key_num  = int(request.form.get('keyNum', 0))
    is_minor = request.form.get('isMinor', 'false').lower() == 'true'
    grid_offset = float(request.form.get('gridOffset', 0.0))

    suffix = os.path.splitext(f.filename)[1] or '.wav'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        results = analyze_chords_madmom(tmp_path, bpm, key_num, is_minor, grid_offset)
        engine  = 'madmom'

        if not results:
            results = analyze_chords_librosa(tmp_path, bpm, key_num, is_minor, grid_offset)
            engine  = 'librosa'

        return jsonify({'chords': results, 'engine': engine})
    finally:
        os.unlink(tmp_path)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
