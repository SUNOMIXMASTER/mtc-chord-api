import collections
import collections.abc
collections.MutableSequence = collections.abc.MutableSequence
collections.MutableMapping = collections.abc.MutableMapping
collections.Mapping = collections.abc.Mapping
collections.Callable = collections.abc.Callable
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tempfile
import os
import librosa

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── 코드 인식 (Madmom CNN) ──
def analyze_chords_madmom(audio_path, bpm, key_num, is_minor, grid_offset=0.0):
    try:
        import madmom
        from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor
        
        featproc = CNNChordFeatureProcessor()
        recproc  = CRFChordRecognitionProcessor()
        
        features = featproc(audio_path)
        chords   = recproc(features)
        # chords: [(start, end, label), ...]
        
        # 마디 그리드로 변환
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
        
        results = []
        for bar_idx in range(total_bars):
            bar_start = grid_offset + bar_idx * bar_sec
            beat_romans = []
            
            for beat in range(4):
                beat_time = bar_start + beat * beat_sec + beat_sec * 0.5
                # beat_time에서 활성화된 코드 찾기
                active_label = None
                for (cs, ce, cl) in chords:
                    if cs <= beat_time < ce:
                        active_label = cl
                        break
                roman = chord_label_to_roman(active_label, key_num, is_minor)
                beat_romans.append(roman)
            
            valid = [r for r in beat_romans if r]
            if len(valid) < 2:
                continue
            # null 채우기
            for b in range(4):
                if not beat_romans[b]:
                    beat_romans[b] = beat_romans[b-1] if b > 0 else valid[0]
            
            results.append({
                'time': bar_start,
                'chord': ' - '.join(beat_romans[:4]),
                'source': 'madmom'
            })
        
        return results

    except Exception as e:
        print(f'[Madmom ERROR] {e}')
        return []


# ── 폴백: librosa NNLS-Chroma ──
def analyze_chords_librosa(audio_path, bpm, key_num, is_minor, grid_offset=0.0):
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    
    beat_sec  = 60.0 / bpm
    bar_sec   = beat_sec * 4
    hop_sec   = 512 / 22050
    total_bars = int(np.ceil((len(y)/sr - grid_offset) / bar_sec))
    
    MAJOR_ROMAN = {0:'I',1:'bII',2:'II',3:'bIII',4:'III',5:'IV',
                   6:'#IV',7:'V',8:'bVI',9:'VI',10:'bVII',11:'VII'}
    MINOR_ROMAN = {0:'i',1:'bII',2:'ii',3:'III',4:'#III',5:'iv',
                   6:'#iv',7:'v',8:'VI',9:'vi',10:'VII',11:'#VII'}
    
    templates = {
        'maj': np.array([1,0,0,0,1,0,0,1,0,0,0,0], dtype=float),
        'min': np.array([1,0,0,1,0,0,0,1,0,0,0,0], dtype=float),
    }
    
    candidates = []
    if not is_minor:
        for iv, t, r in [(0,'maj','I'),(2,'min','ii'),(4,'min','iii'),(5,'maj','IV'),
                         (7,'maj','V'),(9,'min','vi'),(11,'maj','VII')]:
            candidates.append({'root':(key_num+iv)%12,'tmpl':templates[t],'roman':r})
    else:
        for iv, t, r in [(0,'min','i'),(3,'maj','III'),(5,'min','iv'),(7,'min','v'),
                         (8,'maj','VI'),(10,'maj','VII'),(7,'maj','V')]:
            candidates.append({'root':(key_num+iv)%12,'tmpl':templates[t],'roman':r})
    
    results = []
    for bar_idx in range(total_bars):
        bar_start = grid_offset + bar_idx * bar_sec
        beat_romans = []
        
        for beat in range(4):
            t = bar_start + beat * beat_sec + beat_sec * 0.65
            frame = int(t / hop_sec)
            frame = min(frame, chroma.shape[1]-1)
            cv = chroma[:, frame]
            cv = np.roll(cv, -key_num)  # key-relative
            
            best_score, best_roman = -np.inf, None
            for c in candidates:
                shifted = np.roll(c['tmpl'], c['root'] - key_num)
                score = float(np.dot(cv, shifted))
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
        
        results.append({
            'time': bar_start,
            'chord': ' - '.join(beat_romans[:4]),
            'source': 'librosa'
        })
    
    return results


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok'})


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'audio file required'}), 400
    
    f       = request.files['audio']
    bpm     = float(request.form.get('bpm', 128))
    key_num = int(request.form.get('keyNum', 0))
    is_minor = request.form.get('isMinor', 'false').lower() == 'true'
    grid_offset = float(request.form.get('gridOffset', 0.0))
    
    # 임시 파일로 저장
    suffix = os.path.splitext(f.filename)[1] or '.mp3'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        results = analyze_chords_madmom(tmp_path, bpm, key_num, is_minor, grid_offset)
        engine  = 'madmom'
        
        # Madmom 실패시 librosa 폴백
        if not results:
            results = analyze_chords_librosa(tmp_path, bpm, key_num, is_minor, grid_offset)
            engine  = 'librosa'
        
        return jsonify({'chords': results, 'engine': engine})
    
    finally:
        os.unlink(tmp_path)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
