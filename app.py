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
import madmom
from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

_featproc = CNNChordFeatureProcessor()
_recproc  = CRFChordRecognitionProcessor()

GEMINI_API_KEY_ENV = os.environ.get('GEMINI_API_KEY', '')


def compress_chords(raw):
    """연속된 동일 코드 병합 + 소수점 3자리 정리"""
    if not raw:
        return []
    compressed = [raw[0].copy()]
    for item in raw[1:]:
        if item['label'] == compressed[-1]['label']:
            # 같은 코드면 끝 시간만 연장
            compressed[-1]['end'] = item['end']
        else:
            compressed.append(item.copy())
    return compressed


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok', 'geminiKey': GEMINI_API_KEY_ENV})


@app.route('/analyze', methods=['POST'])
def analyze():
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

        # 원시 raw 생성
        raw = []
        for (cs, ce, cl) in chords:
            raw.append({
                'start': round(chunk_offset + cs, 3),
                'end':   round(chunk_offset + ce, 3),
                'label': cl
            })

        # 연속 동일 코드 압축
        compressed = compress_chords(raw)
        print(f'[청크 완료] raw {len(raw)}개 → 압축 {len(compressed)}개 반환')
        return jsonify({'raw': compressed, 'engine': 'madmom'})

    finally:
        os.unlink(tmp_path)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
