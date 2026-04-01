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
import tempfile
import os
import numpy as np
import librosa
import torch
import yaml

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── BTC 엔진 로드 (서버 시작 시 1회) ──
BTC_MODEL    = None
BTC_DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# majmin 코드 레이블 (N + 12 maj + 12 min = 25)
ROOTS        = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
CHORD_LABELS = ['N'] + [r+':maj' for r in ROOTS] + [r+':min' for r in ROOTS]


def load_btc_engine():
    global BTC_MODEL
    try:
        base            = os.path.dirname(os.path.abspath(__file__))
        config_path     = os.path.join(base, 'btc', 'run_config.yaml')
        checkpoint_path = os.path.join(base, 'btc', 'models', 'BTC-SL', 'checkpoint.pth')

        with open(config_path) as f:
            config = yaml.safe_load(f)

        from btc.btc_model import BTC
        BTC_MODEL = BTC(config=config['model'])
        ckpt = torch.load(checkpoint_path, map_location=BTC_DEVICE)
        BTC_MODEL.load_state_dict(ckpt['model'])
        BTC_MODEL.to(BTC_DEVICE)
        BTC_MODEL.eval()
        print(f'[BTC 엔진] 로드 완료 → {BTC_DEVICE}')
    except Exception as e:
        print(f'[BTC 엔진] 로드 실패: {e}')
        BTC_MODEL = None

load_btc_engine()


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


def apply_grid_cut(audio, sr, bpm, grid_offset_sec, cut_ratio=0.12):
    """박자 끝 12% 묵음 → 퀀타이즈 효과"""
    beat_sec      = 60.0 / bpm
    cut_samples   = int(beat_sec * cut_ratio * sr)
    audio         = audio.copy()
    total_samples = len(audio)
    beat_idx      = int(grid_offset_sec * sr)

    while beat_idx < total_samples:
        beat_end   = beat_idx + int(beat_sec * sr)
        mute_start = max(0, beat_end - cut_samples)
        mute_end   = min(beat_end, total_samples)
        audio[mute_start:mute_end] = 0.0
        beat_idx = beat_end

    return audio


def run_btc_engine(audio, sr):
    """
    BTC 엔진 코드 분석
    - 입력: 22050Hz, CQT (24 bins/octave, hop 512)
    - 10초 윈도우, 5초 오버랩 슬라이딩
    - 반환: [(start_sec, end_sec, label), ...]
    """
    if BTC_MODEL is None:
        raise RuntimeError('BTC 엔진 사용 불가')

    TARGET_SR    = 22050
    HOP_LENGTH   = 512
    N_BINS       = 144   # 24 bins/octave × 6 octaves
    BINS_OCTAVE  = 24
    WINDOW_SEC   = 10.0
    OVERLAP_SEC  = 5.0

    # 22050Hz로 리샘플
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # CQT 추출
    cqt = np.abs(librosa.cqt(
        audio, sr=sr, hop_length=HOP_LENGTH,
        n_bins=N_BINS, bins_per_octave=BINS_OCTAVE
    ))
    cqt = librosa.amplitude_to_db(cqt, ref=np.max)
    cqt = (cqt - cqt.mean()) / (cqt.std() + 1e-8)

    fps           = sr / HOP_LENGTH          # ≈ 43.07 프레임/초
    win_frames    = int(WINDOW_SEC * fps)
    step_frames   = int((WINDOW_SEC - OVERLAP_SEC) * fps)
    total_frames  = cqt.shape[1]

    all_probs = np.zeros((total_frames, len(CHORD_LABELS)))
    count     = np.zeros(total_frames)

    start = 0
    while start < total_frames:
        end   = min(start + win_frames, total_frames)
        chunk = cqt[:, start:end]

        # 부족한 부분 패딩
        if chunk.shape[1] < win_frames:
            pad   = np.zeros((N_BINS, win_frames - chunk.shape[1]))
            chunk = np.concatenate([chunk, pad], axis=1)

        x = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(BTC_DEVICE)

        with torch.no_grad():
            logits = BTC_MODEL(x)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        valid = end - start
        all_probs[start:start+valid] += probs[:valid]
        count[start:start+valid]     += 1

        if end >= total_frames:
            break
        start += step_frames

    # 평균 → 코드 결정
    avg_probs = all_probs / np.maximum(count[:, None], 1)
    chord_seq = [CHORD_LABELS[i] for i in np.argmax(avg_probs, axis=1)]

    # 연속 동일 코드 병합 → (start_sec, end_sec, label)
    results     = []
    prev        = chord_seq[0]
    start_frame = 0
    for i, c in enumerate(chord_seq[1:], 1):
        if c != prev:
            results.append((start_frame / fps, i / fps, prev))
            prev        = c
            start_frame = i
    results.append((start_frame / fps, len(chord_seq) / fps, prev))

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
        bar       = int(snapped_s / bar_sec) + 1
        beat      = round(((snapped_s % bar_sec) / beat_sec + 1) * 2) / 2

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
        'engine': 'BTC' if BTC_MODEL else 'unavailable'
    })


@app.route('/analyze_grid', methods=['POST'])
def analyze_grid():
    """
    파이프라인:
    1. 오디오 + BPM + gridOffset 수신
    2. 박자 끝 12% 묵음 (퀀타이즈)
    3. BTC 엔진 분석
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

        audio, sr = librosa.load(tmp_path, sr=None, mono=True)
        print(f'[오디오] sr={sr}, 길이={len(audio)/sr:.1f}초')

        # 퀀타이즈
        audio = apply_grid_cut(audio, sr, bpm, grid_offset_sec)

        # BTC 엔진
        raw_chords = run_btc_engine(audio, sr)
        print(f'[BTC] {len(raw_chords)}개 감지')

        # 차체 안착
        snapped    = snap_to_grid(raw_chords, bpm, grid_offset_sec)
        flat       = [{'start': s['start'], 'end': s['end'], 'label': s['label']} for s in snapped]
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
