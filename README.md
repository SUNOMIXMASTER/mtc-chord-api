# MTC Chord API

Madmom 기반 코드 인식 API 서버 (MIX THE CHORD 전용)

## 엔드포인트

### GET /ping
서버 상태 확인 (웨이크업용)

### POST /analyze
오디오 파일 코드 분석

**Form Data:**
- `audio` : 오디오 파일 (mp3, wav, flac 등)
- `bpm` : BPM (숫자)
- `keyNum` : 키 번호 (0=C, 1=C#, ... 11=B)
- `isMinor` : 단조 여부 (true/false)
- `gridOffset` : 비트그리드 오프셋(초)

**Response:**
```json
{
  "chords": [
    { "time": 0.0, "chord": "I - I - V - V", "source": "madmom" }
  ],
  "engine": "madmom"
}
```

## 배포 (Render)

1. 이 레포를 GitHub에 올리기
2. Render → New Web Service → GitHub 레포 연결
3. 자동 배포 완료
