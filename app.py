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
import time
from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

_featproc = CNNChordFeatureProcessor()
_recproc  = CRFChordRecognitionProcessor()

GEMINI_API_KEY_ENV = os.environ.get('GEMINI_API_KEY', '')

NOTE_MAP = {'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,
            'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11}
MINOR_ROMAN = {0:'i',2:'ii',3:'III',5:'iv',7:'v',8:'VI',10:'VII',11:'vii'}
MAJOR_ROMAN = {0:'I',2:'ii',4:'iii',5:'IV',7:'V',9:'vi',11:'vii'}

def chord_to_roman(label, key_num, is_minor):
    if not label or label in ('N', 'X', 'N/A'):
        return None
    root_str = label.split(':')[0]
    rn = NOTE_MAP.get(root_str)
    if rn is None:
        return None
    interval = (rn - key_num + 12) % 12
    return (MINOR_ROMAN if is_minor else MAJOR_ROMAN).get(interval)

def gemini_grid_mapping(raw_chords, bpm, key_num, is_minor, grid_offset, chunk_offset, api_key):
    if not api_key:
        print('[Gemini] API 키 없음')
        return None

    key_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    key_name = key_names[key_num] + ('m' if is_minor else '')
    beat_sec = 60.0 / bpm
    bar_sec = beat_sec * 4

    # N/X 포함 전체 데이터 전달
    raw_lines = []
    for (cs, ce, cl) in raw_chords:
        roman = chord_to_roman(cl, key_num, is_minor) if cl not in ('N', 'X', 'N/A') else 'Silence'
        abs_start = chunk_offset + cs
        abs_end = chunk_offset + ce
        duration = ce - cs
        raw_lines.append(f'  {abs_start:.3f}s ~ {abs_end:.3f}s | {cl} ({roman}) | {duration:.3f}s')

    raw_text = '\n'.join(raw_lines)

    prompt = (
        'You are an expert music analyst AND experienced DJ specializing in electronic dance music.\n'
        'Your job: analyze Madmom chord data and map it onto a precise bar grid like a Logic Pro chord track.\n\n'
        'RAW CHORD DATA (including silence/N regions for timing reference):\n'
        + raw_text + '\n\n'
        'SONG PARAMETERS:\n'
        f'- BPM: {bpm}\n'
        f'- Key: {key_name}\n'
        f'- Grid offset (beat 1 position): {grid_offset:.4f}s\n'
        f'- Bar duration: {bar_sec:.4f}s\n'
        f'- Beat duration: {beat_sec:.4f}s\n'
        f'- Chunk starts at: {chunk_offset:.2f}s\n\n'
        'STRICT RULES:\n'
        '1. Calculate EXACT bar boundaries using BPM and grid offset.\n'
        '2. For each bar, check which chords fall within it by timestamp.\n'
        '3. Musical judgment per bar:\n'
        '   - Same chord whole bar: repeat in all 4 beats\n'
        '   - 2 chords per bar: first chord beats 1-2, second beats 3-4\n'
        '   - Different each beat: assign individually\n'
        f'4. MANDATORY: Every single bar MUST have all 4 beats filled. NO empty beats allowed.\n'
        '5. For bars with N/Silence/missing data: USE YOUR DJ INTUITION to infer the most\n'
        '   musically logical chord based on surrounding bars and common chord progressions.\n'
        '   Do NOT leave any beat empty or null.\n'
        f'6. Use roman numerals relative to key {key_name} only.\n'
        '7. Ignore chord fragments under 0.2 seconds (noise).\n\n'
        'OUTPUT: Return ONLY valid JSON. No explanation, no markdown.\n'
        '{"bars": [{"bar": 1, "time": 0.000, "beat1": "i", "beat2": "i", "beat3": "III", "beat4": "III"}]}'
    )

    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}'
    payload = {
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {'temperature': 0.1, 'maxOutputTokens': 8192}
    }

    for attempt in range(3):
        try:
            if attempt > 0:
                wait = attempt * 15
                print(f'[Gemini] {wait}초 대기 후 재시도 ({attempt+1}/3)')
                time.sleep(wait)
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code == 429:
                print('[Gemini] 429 Too Many Requests')
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
            for b in bars[:5]:
                print(f'  Bar {b["bar"]}: {b["beat1"]} - {b["beat2"]} - {b["beat3"]} - {b["beat4"]}')
            return bars
        except Exception as e:
            print(f'[Gemini ERROR] attempt {attempt+1}: {e}')

    print('[Gemini] 3회 모두 실패')
    return None


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok'})


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'audio file required'}), 400

    f            = request.files['audio']
    bpm          = float(request.form.get('bpm', 128))
    key_num      = int(request.form.get('keyNum', 0))
    is_minor     = request.form.get('isMinor', 'false').lower() == 'true'
    grid_offset  = float(request.form.get('gridOffset', 0.0))
    chunk_offset = float(request.form.get('chunkOffset', 0.0))

    # 요청에서 온 키 우선, 없으면 환경변수
    api_key = request.form.get('geminiApiKey', '').strip() or GEMINI_API_KEY_ENV

    print(f'[파라미터] bpm={bpm}, keyNum={key_num}, isMinor={is_minor}, gridOffset={grid_offset}, chunkOffset={chunk_offset}')
    print(f'[Gemini 키] {"요청에서" if request.form.get("geminiApiKey") else "환경변수에서"} 로드')

    suffix = os.path.splitext(f.filename)[1] or '.wav'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        features = _featproc(tmp_path)
        chords   = _recproc(features)

        print(f'[Madmom] 원시 코드 {len(chords)}개 감지')
        for c in chords[:10]:
            print(f'  {c[0]:.2f}~{c[1]:.2f}: {c[2]}')

        # Gemini 호출
        gemini_bars = gemini_grid_mapping(chords, bpm, key_num, is_minor, grid_offset, chunk_offset, api_key)

        if gemini_bars:
            results = []
            for bar in gemini_bars:
                pattern = f"{bar['beat1']} - {bar['beat2']} - {bar['beat3']} - {bar['beat4']}"
                results.append({
                    'time': bar['time'],
                    'chord': pattern,
                    'source': 'gemini'
                })
            return jsonify({'chords': results, 'engine': 'gemini'})

        # Gemini 실패시 폴백
        print('[Gemini 실패] 폴백')
        beat_sec = 60.0 / bpm
        bar_sec = beat_sec * 4
        duration = librosa.get_duration(path=tmp_path)
        total_bars = int(np.ceil((duration - grid_offset) / bar_sec))

        results = []
        for bar_idx in range(total_bars):
            bar_start = grid_offset + bar_idx * bar_sec
            beat_romans = []
            for beat in range(4):
                beat_start = bar_start + beat * beat_sec
                beat_end = beat_start + beat_sec
                best_label = None
                best_dur = 0
                for (cs, ce, cl) in chords:
                    if cl in ('N', 'X', 'N/A'): continue
                    overlap = min(chunk_offset + ce, beat_end) - max(chunk_offset + cs, beat_start)
                    if overlap > best_dur:
                        best_dur = overlap
                        best_label = cl
                roman = chord_to_roman(best_label, key_num, is_minor)
                beat_romans.append(roman)
            valid = [r for r in beat_romans if r]
            if len(valid) < 1: continue
            for b in range(4):
                if not beat_romans[b]:
                    beat_romans[b] = beat_romans[b-1] if b > 0 else valid[0]
            results.append({'time': bar_start, 'chord': ' - '.join(beat_romans[:4]), 'source': 'madmom'})

        return jsonify({'chords': results, 'engine': 'madmom'})

    finally:
        os.unlink(tmp_path)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
