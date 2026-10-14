"""OpenAI-compatible transcription API: POST /v1/audio/transcriptions and GET /v1/models.

Lets clients and SDKs written for OpenAI's Whisper API use this server, e.g.
    OpenAI(base_url="http://127.0.0.1:8907/v1", api_key=API_TOKEN or "none").audio.transcriptions.create(...)
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from .config import env
from .diarization import MAX_SPEAKERS
from .language import map_language
from .service import Service, ServiceError
from .timeline import words_from_alignment

logger = logging.getLogger(__name__)

OPENAI_MODEL_NAME = env(
    "OPENAI_MODEL_NAME", "qwen3-asr", "Model id listed by `GET /v1/models` (the `model` form field is accepted but ignored)",
    group="OpenAI API",
)
RESPONSE_FORMATS = ("json", "text", "srt", "verbose_json", "vtt", "diarized_json")

router = APIRouter(prefix="/v1")


def _error(status: int, message: str, param: Optional[str] = None) -> JSONResponse:
    kind = "invalid_request_error" if status < 500 else "server_error"
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": kind, "param": param, "code": None}})


@router.get("/models")
async def list_models():
    return {"object": "list", "data": [{"id": OPENAI_MODEL_NAME, "object": "model", "created": 0, "owned_by": "qwen"}]}


@router.post("/audio/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: Optional[float] = Form(None),  # accepted for compatibility; decoding is greedy
    timestamp_granularities: Optional[List[str]] = Form(None, alias="timestamp_granularities[]"),
    stream: Optional[bool] = Form(None),
    num_speakers: Optional[int] = Form(None),  # extension for diarized_json: the number of speakers, if known
):
    service: Service = request.app.state.service
    if response_format not in RESPONSE_FORMATS:
        return _error(400, f"response_format must be one of {', '.join(RESPONSE_FORMATS)}", "response_format")
    if stream:
        return _error(400, "stream=true is not supported; use the WebSocket endpoint /transcribe-streaming", "stream")
    granularities = set(timestamp_granularities or [])
    if granularities - {"word", "segment"}:
        return _error(400, "timestamp_granularities[] may contain 'word' and 'segment'", "timestamp_granularities[]")
    try:
        full_lang, opencc_config = map_language(language)
    except ValueError as e:
        return _error(400, str(e), "language")
    diarize = response_format == "diarized_json"
    if diarize and (service.diarizer is None or not service.engine.supports_alignment):
        return _error(
            400, "diarized_json needs speaker diarization and the forced aligner (ENABLE_DIARIZATION=true, ENABLE_ALIGNER_MODEL=true)",
            "response_format",
        )
    if num_speakers is not None and not 1 <= num_speakers <= MAX_SPEAKERS:
        return _error(400, f"num_speakers must be between 1 and {MAX_SPEAKERS}", "num_speakers")

    # Word timings come from the forced aligner; segment timings use them when it is available.
    want_words = response_format == "verbose_json" and "word" in granularities
    if want_words and not service.engine.supports_alignment:
        return _error(
            400, "Word timestamps need the forced aligner (ENABLE_ALIGNER_MODEL=true)",
            "timestamp_granularities[]",
        )
    timed = response_format in ("srt", "vtt", "verbose_json")
    align = want_words or (timed and service.engine.supports_alignment)

    try:
        await service.wait_ready()
        audios = await service.decode([await file.read()], [file.filename])
        result = (await service.transcribe(
            audios, full_lang, opencc_config, prompt or "", align=align, diarize=diarize, num_speakers=num_speakers
        ))[0]
    except ServiceError as e:
        return _error(e.status, e.message, "file" if e.status == 400 else None)

    text = result["text"]
    if response_format == "json":
        return {"text": text}
    if response_format == "text":
        return PlainTextResponse(text)

    wav, sr = audios[0]
    duration = round(wav.shape[0] / sr, 3)
    words = words_from_alignment(result.get("timestamps"))
    # without word timings, the whole file is one segment
    segments = result.get("segments") or ([{"start": 0.0, "end": duration, "text": text}] if text else [])

    if diarize:
        return _diarized(text, duration, segments)

    if response_format == "srt":
        return PlainTextResponse(_srt(segments))
    if response_format == "vtt":
        return PlainTextResponse(_vtt(segments))

    body = {"task": "transcribe", "language": (result["language"] or "").lower(), "duration": duration, "text": text}
    # Like OpenAI: segments unless only word timestamps were requested
    if granularities != {"word"}:
        body["segments"] = [
            {"id": i, "seek": 0, "start": s["start"], "end": s["end"], "text": s["text"], "tokens": [],
             "temperature": 0.0, "avg_logprob": 0.0, "compression_ratio": 0.0, "no_speech_prob": 0.0}
            for i, s in enumerate(segments)
        ]
    if want_words:
        body["words"] = words
    return body


def _speaker_name(index: int) -> str:
    """OpenAI's labels for unknown speakers: A, B, ..."""
    return chr(ord("A") + index) if index < 26 else f"S{index}"


def _diarized(text: str, duration: float, segments: List[dict]) -> dict:
    """OpenAI's diarized_json: the transcript in pieces, each with its speaker."""
    return {
        "task": "transcribe",
        "duration": duration,
        "text": text,
        "segments": [
            {"type": "transcript.text.segment", "id": f"seg_{i:03d}", "start": s["start"], "end": s["end"],
             "text": s["text"], "speaker": _speaker_name(s.get("speaker", 0))}
            for i, s in enumerate(segments)
        ],
    }


def _timestamp(sec: float, sep: str) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _srt(segments: List[dict]) -> str:
    return "".join(
        f"{i}\n{_timestamp(s['start'], ',')} --> {_timestamp(s['end'], ',')}\n{s['text']}\n\n"
        for i, s in enumerate(segments, 1)
    )


def _vtt(segments: List[dict]) -> str:
    return "WEBVTT\n\n" + "".join(
        f"{_timestamp(s['start'], '.')} --> {_timestamp(s['end'], '.')}\n{s['text']}\n\n" for s in segments
    )

