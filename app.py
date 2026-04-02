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
import tempfile, os, numpy as np, librosa

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── CREMA 엔진 로드 (서버 시작 시 1회) ──
CREMA_MODEL = None

def load_crema():
    global CREMA_MODEL
    try:
        from crema.analyze import analyze as crema_analyze
        CREMA_MODEL = crema_analyze
        print('[CREMA 엔진] 로드 완료')
    except Exception as e:
        print(f'[CREMA 엔진] 로드 실패: {e}')
        CREMA_MODEL = None

load_crema()


# ── 유틸 ──

def compress_chords(chords):
    """연속 동일 코드 병합"""
    if not chords:
        return []
    result = [chords[0].copy()]
    for item in chords[1:]:
        if item['label'] == result[-1]['label']:
            result[-1]['end'] = item['end']
        else:
            result.append(item.copy())
    return result


def smooth_chords(chords, min_duration=0.3):
    """짧은 코드 제거 + 연속 동일 코드 병합 (Logic 스타일)"""
    filtered = [c for c in chords if (c['end'] - c['start']) >= min_duration]
    if not filtered:
        return chords
    merged = [filtered[0].copy()]
    for c in filtered[1:]:
        if c['label'] == merged[-1]['label']:
            merged[-1]['end'] = c['end']
        else:
            merged.append(c.copy())
    return merged


def normalize_chord_label(label):
    """CREMA 레이블 → 간결한 코드명 변환
    예: C:maj → C, A:min → Am, N → N
    """
    if label in ('N', 'X'):
        return label
    if ':' not in label:
        return label

    root, quality = label.split('/', 1)[0].split(':', 1)

    quality_map = {
        'maj':  '',    # C:maj → C
        'min':  'm',   # A:min → Am
        'maj7': 'maj7',
        'min7': 'm7',
        '7':    '7',
        'dim':  'dim',
        'aug':  'aug',
        'sus2': 'sus2',
        'sus4': 'sus4',
        'hdim7':'m7b5',
    }
    q = quality_map.get(quality, quality)
    return f'{root}{q}'


def run_crema_engine(audio_path):
    """CREMA 엔진으로 코드 분석 → [(start, end, label), ...]"""
    if CREMA_MODEL is None:
        raise RuntimeError('CREMA 엔진 사용 불가')

    jam = CREMA_MODEL(filename=audio_path)
    ann = jam.annotations['chord'][0]

    results = []
    for obs in ann.data:
        start = obs.time.total_seconds()
        end   = start + obs.duration.total_seconds()
        label = normalize_chord_label(obs.value)
        results.append((start, end, label))

    return results


def snap_to_grid(chords, bpm, grid_offset_sec):
    """코드(초)를 0.5박 단위 차체(그리드)에 안착"""
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
            'label': cl,
            'bar':   bar,
            'beat':  beat
        })

    return result


# ── 엔드포인트 ──

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({
        'status': 'ok',
        'engine': 'CREMA' if CREMA_MODEL else 'unavailable'
    })


@app.route('/analyze_grid', methods=['POST'])
def analyze_grid():
    """
    파이프라인:
    1. 오디오 + BPM + gridOffset 수신
    2. CREMA 엔진 분석
    3. Smooth (Logic 스타일)
    4. 차체(그리드) 안착 → 반환
    """
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

        # CREMA 엔진
        raw_chords = run_crema_engine(tmp_path)
        print(f'[CREMA] {len(raw_chords)}개 감지')

        # Smooth (Logic 스타일)
        raw_list = [{'start': s, 'end': e, 'label': l} for s, e, l in raw_chords]
        smoothed = smooth_chords(raw_list, min_duration=0.3)

        # 차체(그리드) 안착
        tuples   = [(c['start'], c['end'], c['label']) for c in smoothed]
        snapped  = snap_to_grid(tuples, bpm, grid_offset_sec)
        flat     = [{'start': s['start'], 'end': s['end'], 'label': s['label']} for s in snapped]
        compressed = compress_chords(flat)

        # 마디/박자 재계산
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
