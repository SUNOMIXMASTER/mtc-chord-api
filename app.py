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

_featproc = CNNChordFeatureProcessor()
_recproc  = CRFChordRecognitionProcessor()

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

        # 원시 데이터를 절대 시간으로 변환해서 반환
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
