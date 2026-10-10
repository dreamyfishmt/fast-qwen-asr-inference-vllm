import logging

import pytest

from asr_server.auth import RedactTokenFilter, redact
from asr_server.language import map_language


@pytest.mark.parametrize("code,expected", [
    ("en", ("English", None)), ("en-US", ("English", None)), ("zh-CN", ("Chinese", None)),
    ("zh_TW", ("Chinese", "s2twp")), ("zh-HK", ("Chinese", "s2hk")), ("Chinese", ("Chinese", None)),
    ("yue", ("Cantonese", None)), (None, (None, None)), ("  ", (None, None)),
])
def test_map_language(code, expected):
    assert map_language(code) == expected


def test_map_language_rejects_unknown():
    with pytest.raises(ValueError):
        map_language("xx")


def test_redact():
    assert redact("/transcribe?language=de&token=abc123") == "/transcribe?language=de&token=***"
    assert redact("/x?token=abc&b=1") == "/x?token=***&b=1"
    assert redact("/x?mytoken=abc") == "/x?mytoken=abc"


def test_filter_redacts_uvicorn_access_record():
    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:5000", "GET", "/health?token=s3cret", "1.1", 200), None,
    )
    RedactTokenFilter().filter(record)
    assert "s3cret" not in record.getMessage()
