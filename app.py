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

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

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

def gemini_grid_mapping(raw_chords, bpm, key_num, is_minor, grid_offset, chunk_offset):
    if not GEMINI_API_KEY:
        print('[Gemini] API 키 없음')
        return None

    key_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    key_name = key_names[key_num] + ('m' if is_minor else '')
    beat_sec = 60.0 / bpm
    bar_sec = beat_sec * 4

    raw_lines = []
    for (cs, ce, cl) in raw_chords:
        if cl in ('N', 'X', 'N/A'):
            continue
        roman = chord_to_roman(cl, key_num, is_minor)
        abs_start = chunk_offset + cs
        abs_end = chunk_offset + ce
        duration = ce - cs
        raw_lines.append(f'  {abs_start:.3f}s ~ {abs_end:.3f}s | {cl} -> {roman} | duration: {duration:.3f}s')

    raw_text = '\n'.join(raw_lines)

    prompt = f"""You are an expert music analyst for DJ mixing and electronic dance music.
Analyze chord data from Madmom CNN and map it onto a musical bar grid like a DAW chord track.

RAW CHORD DATA:
{raw_text}

SONG PARAMETERS:
- BPM: {bpm}
- Key: {key_name}
- Grid offset (beat 1 position): {grid_offset:.4f}s
- Bar duration: {bar_sec:.4f}s
- Beat duration: {beat_sec:.4f}s
- Chunk starts at: {chunk_offset:.2f}s

TASK:
1. Calculate bar boundaries using BPM and grid offset.
2. For each bar, assign chords to 4 beats based on actual timestamps.
3. If chord spans whole bar: repeat in all 4 beats.
4. If 2 chords per bar: beats 1-2 first chord, beats 3-4 second chord.
5. Use roman numerals relative to key {key_name}.
6. Ignore chord fragments under 0.2 seconds.

OUTPUT: JSON only, no explanation.
{{"bars": [{{"bar": 1, "time": 0.000, "beat1": "i", "beat2": "i", "beat3": "III", "beat4": "III"}}]}}

Roman numerals for {key_name}:
- Minor: i ii III iv v VI VII
- Major: I ii iii IV V vi vii"""

    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}'
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

    print(f'[파라미터] bpm={bpm}, keyNum={key_num}, isMinor={is_minor}, gridOffset={grid_offset}, chunkOffset={chunk_offset}')

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

        # raw 데이터 반환 (index.html이 Gemini 호출)
        raw = []
        for (cs, ce, cl) in chords:
            raw.append({
                'start': round(float(chunk_offset + cs), 3),
                'end':   round(float(chunk_offset + ce), 3),
                'label': cl
            })
        return jsonify({
            'raw': raw,
            'bpm': bpm,
            'keyNum': key_num,
            'isMinor': is_minor,
            'gridOffset': grid_offset
        })

    finally:
        os.unlink(tmp_path)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
