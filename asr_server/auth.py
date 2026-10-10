"""API token check, and redaction of tokens from log lines."""

import hmac
import logging
import re

from . import config


def token_ok(headers, query_params) -> bool:
    if not config.API_TOKEN:
        return True
    supplied = query_params.get("token") or ""
    auth = headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        supplied = auth[7:].strip()
    return hmac.compare_digest(supplied.encode(), config.API_TOKEN.encode())


_TOKEN_RE = re.compile(r"([?&]token=)[^&\s\"']*", re.IGNORECASE)


def redact(text: str) -> str:
    return _TOKEN_RE.sub(r"\1***", text)


class RedactTokenFilter(logging.Filter):
    """Masks `?token=...` in log records. Uvicorn logs the full request path, query string included,
    for every HTTP request (uvicorn.access) and WebSocket handshake (uvicorn.error)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str) and "token=" in record.msg:
            record.msg = redact(record.msg)
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(redact(a) if isinstance(a, str) and "token=" in a else a for a in record.args)
            elif isinstance(record.args, dict):
                record.args = {k: redact(v) if isinstance(v, str) else v for k, v in record.args.items()}
        return True


def install_log_redaction() -> None:
    f = RedactTokenFilter()
    for name in ("uvicorn.access", "uvicorn.error", "uvicorn"):
        logger = logging.getLogger(name)
        if not any(isinstance(x, RedactTokenFilter) for x in logger.filters):
            logger.addFilter(f)
