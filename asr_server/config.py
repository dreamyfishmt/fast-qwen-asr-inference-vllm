"""Environment-variable settings.

Every setting is declared with `env()`, which reads the variable and records it (with its default
and description) so that `scripts/gen-config-docs.py` can generate the table in docs/configuration.md.
"""

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple


@dataclass(frozen=True)
class Setting:
    name: str
    default: str
    help: str
    group: str


# (group, name) -> Setting; a variable can appear in several groups (e.g. once per backend)
REGISTRY: Dict[Tuple[str, str], Setting] = {}

_TRUE = ("true", "1", "yes", "on")


def env(name: str, default, help: str, group: str = "Server", cast: Optional[Callable] = None):
    """Read environment variable `name` (falling back to `default`) and register it for the docs.

    The type of `default` picks the conversion (bool, int, float or str) unless `cast` is given.
    """
    REGISTRY[(group, name)] = Setting(name, _show(default), help, group)
    raw = os.getenv(name)
    if raw is None or (raw.strip() == "" and not isinstance(default, str)):
        return default
    if cast is not None:
        return cast(raw)
    if isinstance(default, bool):
        return raw.strip().lower() in _TRUE
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw.strip()


def _show(default) -> str:
    if isinstance(default, bool):
        return "true" if default else "false"
    return str(default)


def _cpu_count() -> int:
    return os.cpu_count() or 4


# -----------------------------
# Server settings (backend settings live in asr_server/engines/)
# -----------------------------
ASR_BACKEND = env("ASR_BACKEND", "vllm", "Inference backend: `vllm` or `onnx` (set by each image)")
API_TOKEN = env(
    "API_TOKEN", "", "Shared secret; clients send `Authorization: Bearer <token>` (or `?token=`). Empty = no auth",
)
CORS_ORIGINS = env("CORS_ORIGINS", "*", "Comma-separated origins allowed by CORS (`*` = any)")

MAX_CONCURRENT_DECODE = env("MAX_CONCURRENT_DECODE", 4, "Audio files decoded in parallel", group="Limits")
MAX_CONCURRENT_INFER = env("MAX_CONCURRENT_INFER", 1, "Inference calls run in parallel (GPU: usually 1)", group="Limits")
THREADPOOL_WORKERS = env(
    "THREADPOOL_WORKERS", 0, "Worker threads for blocking calls (0 = 5 × CPU cores)", group="Limits",
) or _cpu_count() * 5
MAX_UPLOAD_MB = env("MAX_UPLOAD_MB", 100.0, "Largest accepted request body for the upload endpoints (0 = unlimited)", group="Limits")
MAX_FILES = env("MAX_FILES", 16, "Most files accepted in one `POST /transcribe` request", group="Limits")
FFMPEG_TIMEOUT_SEC = env("FFMPEG_TIMEOUT_SEC", 60.0, "Time limit for decoding one upload with ffmpeg", group="Limits")
EXIT_ON_LOAD_FAILURE = env(
    "EXIT_ON_LOAD_FAILURE", True,
    "Stop the server when the model fails to load, so the container restart policy retries", group="Limits",
)

STREAM_MIN_SAMPLES = env(
    "STREAM_MIN_SAMPLES", 1600, "Samples (16 kHz) buffered before they are fed to the stream (1600 = 100 ms)", group="Streaming",
)
PARTIAL_INTERVAL_MS = env("PARTIAL_INTERVAL_MS", 120, "Minimum interval between `partial` messages", group="Streaming")
STREAM_EXPECT_SR = env(
    "STREAM_EXPECT_SR", 16000, "Sample rate assumed when `start` has no `sample_rate_hz`", group="Streaming",
)
STREAM_MAX_SEC = env(
    "STREAM_MAX_SEC", 300.0, "Audio beyond this many seconds per utterance is dropped (0 = unlimited)", group="Streaming",
)
STREAM_IDLE_TIMEOUT_SEC = env(
    "STREAM_IDLE_TIMEOUT_SEC", 30.0, "Close a stream that sends nothing for this long (0 = never)", group="Streaming",
)
MAX_STREAMS = env("MAX_STREAMS", 64, "Most concurrent WebSocket streams; more are refused (0 = unlimited)", group="Streaming")

OPENCC_TW_CONFIG = env("OPENCC_TW_CONFIG", "s2twp", "OpenCC config for `zh-TW` / `zh-Hant`", group="Languages")
OPENCC_HK_CONFIG = env("OPENCC_HK_CONFIG", "s2hk", "OpenCC config for `zh-HK` / `zh-MO`", group="Languages")
