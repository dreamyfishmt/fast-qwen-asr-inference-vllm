"""Forced alignment (word timestamps) with an ONNX export of Qwen3-ForcedAligner-0.6B.

The model is the community export valoomba/Qwen3-ForcedAligner-0.6B-ONNX (onnx/model_q4.onnx, int4 weights):
one graph that takes the prompt ids, the log-mel features and their masks and returns, for every prompt
position, logits over 5000 time slots of `timestamp_segment_time` ms. It is not autoregressive: one run
aligns a whole transcript.

The prompt and the post-processing follow qwen_asr's Qwen3ForcedAligner (Apache-2.0, Copyright 2026 The
Alibaba Qwen team): the transcript is split into words (CJK characters count as words), each followed by
two <timestamp> tokens whose predicted slots are the word's start and end; a longest non-decreasing
subsequence keeps the timestamps monotonic and the outliers are filled in from their neighbours.
qwen_asr splits Japanese with nagisa and Korean with soynlp; here Japanese is split into characters and
Korean at spaces, which needs no extra dependencies.
"""

import json
import logging
import unicodedata
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

TIMESTAMP = "<timestamp>"
AUDIO_START, AUDIO_PAD, AUDIO_END = "<|audio_start|>", "<|audio_pad|>", "<|audio_end|>"
GRAPH_FILES = ("onnx/model_q4.onnx", "model_q4.onnx", "onnx/model.onnx", "model.onnx")


def _is_kept_char(ch: str) -> bool:
    return ch == "'" or unicodedata.category(ch)[0] in "LN"


def _clean(token: str) -> str:
    return "".join(ch for ch in token if _is_kept_char(ch))


def _is_cjk_ideograph(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF or 0xF900 <= cp <= 0xFAFF
    )


def _is_kana(ch: str) -> bool:
    cp = ord(ch)
    return 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF or 0xFF66 <= cp <= 0xFF9F


def split_words(text: str, language: Optional[str]) -> List[str]:
    """The units that get a timestamp: words, with CJK characters (and Japanese kana) on their own."""
    single = (lambda ch: _is_cjk_ideograph(ch) or _is_kana(ch)) if (language or "").lower() == "japanese" else _is_cjk_ideograph
    words: List[str] = []
    for chunk in text.split():
        buf = ""
        for ch in _clean(chunk):
            if single(ch):
                if buf:
                    words.append(buf)
                    buf = ""
                words.append(ch)
            else:
                buf += ch
        if buf:
            words.append(buf)
    return words


def fix_timestamps(values: Sequence[float]) -> List[float]:
    """Make timestamps non-decreasing: keep a longest non-decreasing subsequence and replace the rest.

    Runs of up to two outliers take the nearer neighbour's value; longer runs are interpolated linearly.
    Same rules and tie-breaking as qwen_asr's Qwen3ForceAlignProcessor.fix_timestamp.
    """
    data = [float(v) for v in values]
    n = len(data)
    if n == 0:
        return []
    # longest non-decreasing subsequence; each element links to the first predecessor of a longest one
    arr = np.asarray(data)
    dp = np.ones(n, dtype=np.int64)
    parent = [-1] * n
    for i in range(1, n):
        cand = np.where(arr[:i] <= arr[i], dp[:i], 0)
        j = int(np.argmax(cand))
        if cand[j] > 0:
            dp[i] = cand[j] + 1
            parent[i] = j
    normal = [False] * n
    i = int(np.argmax(dp))
    while i != -1:
        normal[i] = True
        i = parent[i]

    result = data[:]
    i = 0
    while i < n:
        if normal[i]:
            i += 1
            continue
        j = i
        while j < n and not normal[j]:
            j += 1
        left = result[i - 1] if i > 0 else None  # i - 1 and j are on the subsequence
        right = data[j] if j < n else None
        for k in range(i, j):
            if left is None:
                result[k] = right
            elif right is None:
                result[k] = left
            elif j - i <= 2:
                result[k] = left if (k - (i - 1)) <= (j - k) else right
            else:
                result[k] = left + (right - left) / (j - i + 1) * (k - i + 1)
        i = j
    return result


def audio_token_count(frames: int) -> int:
    """Audio encoder output length for `frames` mel frames (Qwen3-ASR's _get_feat_extract_output_lengths)."""
    rest = frames % 100
    feat = (rest - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (frames // 100) * 13


class OnnxAligner:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)

    def load(self, make_session: Callable[[Path], object]) -> None:
        """`make_session(path)` returns an onnxruntime InferenceSession (the engine's providers and options)."""
        from tokenizers import Tokenizer

        d = self.model_dir
        if not d.is_dir():
            raise RuntimeError(f"Aligner model directory not found: {d}")
        graph = next((d / name for name in GRAPH_FILES if (d / name).is_file()), None)
        if graph is None:
            raise RuntimeError(f"No aligner graph in {d} (expected one of {', '.join(GRAPH_FILES)})")
        with open(d / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        self.timestamp_ms = float(cfg.get("timestamp_segment_time", 80))
        self._tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.timestamp_id = int(cfg.get("timestamp_token_id") or self._tokenizer.token_to_id(TIMESTAMP))
        self._session = make_session(graph)
        logger.info(f"Aligner loaded ({graph.relative_to(d)}, {self.timestamp_ms:g} ms slots)")

    def align(self, mel: np.ndarray, text: str, language: Optional[str]) -> List[dict]:
        """Word timings for the transcript `text` of the audio whose log-mel features are `mel` [1, 128, frames].

        Returns [{"text", "start_time", "end_time"}] with times in seconds.
        """
        words = split_words(text, language)
        if not words:
            return []
        frames = mel.shape[2]
        prompt = (
            AUDIO_START + AUDIO_PAD * audio_token_count(frames) + AUDIO_END
            + (TIMESTAMP * 2).join(words) + TIMESTAMP * 2
        )
        ids = np.array([self._tokenizer.encode(prompt, add_special_tokens=False).ids], dtype=np.int64)
        logits = self._session.run(["logits"], {
            "input_ids": ids,
            "attention_mask": np.ones_like(ids),
            "input_features": mel.astype(np.float32, copy=False),
            "feature_attention_mask": np.ones((1, frames), dtype=np.int32),
        })[0]
        slots = logits[0].argmax(axis=-1)[ids[0] == self.timestamp_id]
        times = fix_timestamps(slots * self.timestamp_ms)
        return [
            {"text": w, "start_time": round(times[2 * i] / 1000.0, 3), "end_time": round(times[2 * i + 1] / 1000.0, 3)}
            for i, w in enumerate(words)
        ]


def offset_items(items: Sequence[dict], offset_sec: float) -> List[dict]:
    return [
        {**it, "start_time": round(it["start_time"] + offset_sec, 3), "end_time": round(it["end_time"] + offset_sec, 3)}
        for it in items
    ]

