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
import requests
import json
from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

_featproc = CNNChordFeatureProcessor()
_recproc  = CRFChordRecognitionProcessor()

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

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

def gemini_grid_mapping(raw_chords, bpm, key_num, is_minor, grid_offset, chunk_offset):
    if not GEMINI_API_KEY:
        return None

    key_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    key_name = key_names[key_num] + ('m' if is_minor else '')
    beat_sec = 60.0 / bpm
    bar_sec = beat_sec * 4

    # 원시 코드 데이터 텍스트 구성
    raw_lines = []
    for (cs, ce, cl) in raw_chords:
        if cl in ('N', 'X', 'N/A'):
            continue
        roman = chord_label_to_roman(cl, key_num, is_minor)
        abs_start = chunk_offset + cs
        abs_end = chunk_offset + ce
        duration = ce - cs
        raw_lines.append(f'  {abs_start:.3f}s ~ {abs_end:.3f}s | {cl} → {roman} | duration: {duration:.3f}s')

    raw_text = '\n'.join(raw_lines)

    prompt = f"""You are an expert music analyst specializing in DJ mixing and electronic dance music.
Your task is to analyze chord data extracted from an audio analysis engine (Madmom CNN) and map it precisely onto a musical bar grid — exactly like a DAW chord track (e.g. Logic Pro chord track).

═══════════════════════════════
RAW CHORD DATA FROM MADMOM:
═══════════════════════════════
{raw_text}

═══════════════════════════════
SONG PARAMETERS:
═══════════════════════════════
- BPM: {bpm}
- Key: {key_name}
- Grid offset (position of beat 1): {grid_offset:.4f}s
- Bar duration: {bar_sec:.4f}s
- Beat duration: {beat_sec:.4f}s
- This audio chunk starts at: {chunk_offset:.2f}s

═══════════════════════════════
YOUR ANALYSIS TASK:
═══════════════════════════════
1. Using the BPM and grid offset, calculate the exact bar boundaries.
2. For each bar, examine which chords from Madmom fall within it.
3. Make a MUSICAL judgment:
   - Does the harmony stay the same for the whole bar? → repeat the same chord in all 4 beats
   - Does it change every 2 beats? → first chord in beats 1-2, second chord in beats 3-4
   - Does it change every beat? → assign each beat individually
   - Base this judgment on the actual chord transition timestamps, NOT assumptions.
4. Use roman numeral notation relative to the key {key_name}.
5. Ignore very short chord fragments under 0.2 seconds (likely noise).
6. If a bar has no clear chord data, use the most recent chord.

═══════════════════════════════
OUTPUT FORMAT (STRICT JSON ONLY):
═══════════════════════════════
Return ONLY a JSON object. No explanation, no markdown, no extra text.
{{
  "bars": [
    {{"bar": 1, "time": 0.000, "beat1": "i", "beat2": "i", "beat3": "III", "beat4": "III"}},
    {{"bar": 2, "time": 1.905, "beat1": "VII", "beat2": "VII", "beat3": "VI", "beat4": "VI"}}
  ]
}}

Roman numeral reference for key {key_name}:
- Minor keys use lowercase: i, ii, III, iv, v, VI, VII
- Major keys use uppercase: I, II, III, IV, V, VI, VII
- Accidentals allowed: bII, #IV, bVI, bVII etc."""

    import time
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}'
    payload = {
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {'temperature': 0.1, 'maxOutputTokens': 8192}
    }
    for attempt in range(3):
        try:
            if attempt > 0:
                wait = attempt * 3
                print(f'[Gemini] {wait}초 대기 후 재시도 ({attempt+1}/3)')
                time.sleep(wait)
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code == 429:
                print(f'[Gemini] 429 Too Many Requests - 재시도 대기')
                continue
            resp.raise_for_status()
            data = resp.json()
            text = data['candidates'][0]['content']['parts'][0]['text'].strip()
            if '```' in text:
                text = text.split('```')[1]
                if text.startswith('json'):
                    text = text[4:]
            text = text.strip()
            parsed = json.loads(text)
            bars = parsed['bars']
            print(f'[Gemini] {len(bars)}개 마디 반환')
            for b in bars[:10]:
                print(f'  Bar {b["bar"]}: {b["beat1"]} - {b["beat2"]} - {b["beat3"]} - {b["beat4"]}')
            return bars
        except Exception as e:
            print(f'[Gemini ERROR] attempt {attempt+1}: {e}')
    print('[Gemini] 3회 시도 모두 실패')
    return None


def analyze_chords_madmom(audio_path, bpm, key_num, is_minor, grid_offset=0.0, chunk_offset=0.0):
    try:
        features = _featproc(audio_path)
        chords   = _recproc(features)

        print(f'[Madmom] 원시 코드 {len(chords)}개 감지')
        for c in chords[:10]:
            print(f'  {c[0]:.2f}~{c[1]:.2f}: {c[2]}')

        # Gemini 그리드 매핑
        gemini_bars = gemini_grid_mapping(chords, bpm, key_num, is_minor, grid_offset, chunk_offset)

        if gemini_bars:
            results = []
            for bar in gemini_bars:
                pattern = f"{bar['beat1']} - {bar['beat2']} - {bar['beat3']} - {bar['beat4']}"
                results.append({
                    'time': bar['time'],
                    'chord': pattern,
                    'source': 'gemini'
                })
            return results

        # Gemini 실패시 기존 방식 폴백
        print('[Gemini 실패] → 기존 그리드 방식으로 폴백')
        beat_sec   = 60.0 / bpm
        bar_sec    = beat_sec * 4
        duration   = librosa.get_duration(path=audio_path)
        total_bars = int(np.ceil((duration - grid_offset) / bar_sec))

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
                    beat_romans[b] = beat_romans[b-1] if b > 0 else valid[0]
            pattern = ' - '.join(beat_romans[:4])
            results.append({'time': bar_start, 'chord': pattern, 'source': 'madmom'})
        return results

    except Exception as e:
        import traceback
        print(f'[Madmom ERROR] {e}')
        traceback.print_exc()
        return []


def analyze_chords_librosa(audio_path, bpm, key_num, is_minor, grid_offset=0.0):
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    beat_sec  = 60.0 / bpm
    bar_sec   = beat_sec * 4
    hop_sec   = 512 / 22050
    total_bars = int(np.ceil((len(y)/sr - grid_offset) / bar_sec))
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
    grid_offset  = float(request.form.get('gridOffset', 0.0))
    chunk_offset = float(request.form.get('chunkOffset', 0.0))

    print(f'[파라미터] bpm={bpm}, keyNum={key_num}, isMinor={is_minor}, gridOffset={grid_offset}, chunkOffset={chunk_offset}')

    suffix = os.path.splitext(f.filename)[1] or '.wav'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        results = analyze_chords_madmom(tmp_path, bpm, key_num, is_minor, grid_offset, chunk_offset)
        engine  = 'gemini'

        if not results:
            results = analyze_chords_librosa(tmp_path, bpm, key_num, is_minor, grid_offset)
            engine  = 'librosa'

        return jsonify({'chords': results, 'engine': engine})
    finally:
        os.unlink(tmp_path)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
