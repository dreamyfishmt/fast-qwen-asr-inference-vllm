"""Qwen3-ASR inference server.

  config       environment settings
  language     language codes -> Qwen language names, OpenCC conversion
  audio        upload decoding and resampling
  auth         API token check and log redaction
  service      shared state: engine, loading status, concurrency limits
  app          FastAPI app: /health, /ready, /transcribe, /transcribe-streaming
  openai_api   OpenAI-compatible /v1/audio/transcriptions
  engines      inference backends (vLLM, ONNX Runtime)
"""
