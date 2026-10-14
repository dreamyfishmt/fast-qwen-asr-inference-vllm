# Word timestamps and speaker diarization (ONNX images)

Both ONNX images (GPU and CPU) can return, for a whole recording, the transcript with word timings and the speaker
of each part, in one request:

```bash
curl -X POST "http://127.0.0.1:8907/transcribe?diarize=true" -F "files=@meeting.wav"
```

```json
[{
  "text": "这是一个测试说话人日志的音频，下面我们来播放一些测试音频。非常累的时候，就会站在阳台上吹吹风……",
  "language": "Chinese",
  "timestamps": {"items": [{"text": "这", "start_time": 0.56, "end_time": 0.72}, ...]},
  "speakers": [{"start": 0.318, "end": 6.865, "speaker": 0}, {"start": 7.017, "end": 10.747, "speaker": 1}, ...],
  "segments": [
    {"start": 0.56, "end": 6.72, "text": "这是一个测试说话人日志的音频，下面我们来播放一些测试音频。", "speaker": 0},
    {"start": 7.04, "end": 13.6, "text": "非常累的时候，就会站在阳台上吹吹风，看远处的灯火明明暗暗。", "speaker": 1},
    ...
  ]
}]
```

Both features are optional and off by default; each needs its own model download.

| Feature | Model | Memory | Speed (4 CPU cores) |
|---|---|---|---|
| Word timestamps (`ENABLE_ALIGNER_MODEL`) | [`valoomba/Qwen3-ForcedAligner-0.6B-ONNX`](https://huggingface.co/valoomba/Qwen3-ForcedAligner-0.6B-ONNX) `onnx/model_q4.onnx`, a community int4 export of [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) (~1 GB) | ~1.3 GB | ~5 s per 30 s of audio |
| Speaker diarization (`ENABLE_DIARIZATION`) | [pyannote segmentation-3.0](https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0) + [3D-Speaker CAM++](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models) (zh/en), run by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (~35 MB) | ~0.15 GB | ~5 s per minute of audio |

The aligner runs with the ASR model's execution provider (CUDA in the GPU image); diarization always runs on the
CPU (sherpa-onnx brings its own CPU build of ONNX Runtime).

## Setup

1. Download the models next to the ASR model:

   ```bash
   scripts/download-models.sh aligner /srv/models       # -> /srv/models/qwen3-forcedaligner-0.6b-onnx
   scripts/download-models.sh diarization /srv/models   # -> /srv/models/speaker-diarization
   ```

2. Enable them in `.env` (folder names inside `MODEL_DIR`):

   ```bash
   ENABLE_ALIGNER_MODEL=true
   ALIGNER_MODEL_DIR=qwen3-forcedaligner-0.6b-onnx
   ENABLE_DIARIZATION=true
   DIARIZATION_MODEL_DIR=speaker-diarization
   ```

3. Restart (`docker compose -f compose.gpu.yaml up -d`, or `compose.cpu.yaml`). `GET /health` reports
   `"forced_alignment": true` and `"diarization": true` once they are loaded.

On the CPU image, plan for ~3.5 GB of RAM with both enabled (the 0.6B ASR model, the aligner and diarization).

## Requests

`POST /transcribe`:

| Query | Result |
|---|---|
| `forced_alignment=true` | `timestamps` (word timings) and `segments` (transcript pieces with start/end) |
| `diarize=true` | `speakers` (speaker turns); with the aligner also `timestamps`, and `segments` with a `speaker` each |
| `num_speakers=N` | With `diarize`: the number of speakers, if known (1–32). Otherwise they are found by `DIARIZATION_THRESHOLD` |

`POST /v1/audio/transcriptions` (OpenAI-compatible):

- `response_format=verbose_json` with `timestamp_granularities[]=word` (and/or `segment`), `srt`, `vtt`: timings
  from the aligner.
- `response_format=diarized_json`: OpenAI's diarized format, segments labelled `A`, `B`, …. Needs both features.
  An extra form field `num_speakers` sets the number of speakers.

```bash
curl http://127.0.0.1:8907/v1/audio/transcriptions -F file=@meeting.wav -F response_format=diarized_json -F num_speakers=2
```

## How it works

1. The audio is transcribed in segments of at most `ONNX_SEGMENT_SEC` (30 s), cut at pauses.
2. Each segment is aligned right after it is transcribed: the aligner predicts a start and an end slot (80 ms) for
   every word (every character in Chinese and Japanese) in one forward pass; the timings are made monotonic and
   shifted by the segment's start.
3. Diarization runs on the whole recording, so speaker numbers are consistent across it: speakers are numbered
   `0, 1, …` in order of their first turn.
4. Each word gets the speaker of the turn it overlaps most (or the nearest turn). `segments` break where the speaker
   changes, at pauses over 0.8 s, and after 10 s; their text is cut from the transcript, so it keeps the punctuation
   (and is converted to Traditional Chinese for `zh-TW` / `zh-HK`).

## Limitations

- The words in `timestamps` are the transcript without punctuation, in the model's original script (Simplified
  Chinese even with `zh-TW`).
- qwen-asr splits Japanese words with nagisa and Korean with soynlp. To avoid those dependencies, Japanese is split
  into characters here and Korean at spaces; timings are per character / per space-separated word.
- Overlapping speech is attributed to one speaker; very short turns (below `DIARIZATION_MIN_DURATION_ON`) are
  dropped. If the number of speakers comes out wrong, pass `num_speakers` or tune `DIARIZATION_THRESHOLD`
  (higher = fewer speakers).
- The embedding model is trained on Chinese and English speech; it works for other languages, but less well.
- The aligner handles up to ~5 minutes per pass. Keep `ONNX_SEGMENT_SEC` (30 s by default) well below that; with
  `ONNX_SEGMENT_SEC=0` the whole recording is aligned at once.

All settings: [configuration.md](configuration.md#speaker-diarization).
