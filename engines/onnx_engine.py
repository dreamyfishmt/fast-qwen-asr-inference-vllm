"""ONNX Runtime backend for Qwen3-ASR (CPU).

Runs an int4 ONNX export of Qwen3-ASR (encoder + decoder with KV cache), e.g.
rhasspy/qwen3-asr-0.6b-onnx-int4-merged. No PyTorch, no GPU.

The pipeline (mel front-end, chat-template prompt, greedy decode with a cached
system-turn KV) follows rhasspy/wyoming-faster-whisper's qwen3_asr_handler.py
(MIT License, Copyright (c) 2025 Michael Hansen).

Two decoder layouts are supported, chosen by the files in the model directory:
  merged - decoder_merged.int4.onnx (+ .data): one graph, KV cache + dynamic length
  split  - decoder_init.int4.onnx + decoder_step.int4.onnx (+ decoder_weights.int4.data)

Streaming: Qwen3-ASR has no incremental decoder for this export, so a partial
result means re-transcribing all audio so far. On a small CPU this is only
affordable for short utterances, so partials run every STREAM_PARTIAL_INTERVAL_SEC
until the utterance is STREAM_PARTIAL_MAX_SEC long; the final result is always a
full transcription of the whole utterance.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import Audio, Engine, Stream

logger = logging.getLogger(__name__)

RATE = 16000

# Mel front-end (identical to Whisper's)
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
FMIN = 0.0
FMAX = 8000.0

# Chat template token ids
IM_START = 151644
IM_END = 151645
SYSTEM = 8948
USER = 872
ASSISTANT = 77091
NEWLINE = 198
AUDIO_START = 151669
AUDIO_END = 151670
AUDIO_PAD = 151676
ASR_TEXT = 151704  # <asr_text>: everything before it is the "language X" preamble
EOS_IDS = frozenset((151643, 151645))

MERGED_DECODER = "decoder_merged.int4.onnx"
MASK_BLOCKED = np.finfo(np.float32).min

# Encoder attention spans windows of n_window_infer=800 mel frames (8 s), so the
# encoder can run on 8 s segments without changing the result while bounding memory.
ENCODER_WINDOW_FRAMES = 800


def _env_bool(key: str, default: str) -> bool:
    return os.getenv(key, default).lower() in ("true", "1", "yes", "on")


ONNX_THREADS = int(os.getenv("ONNX_THREADS", "0")) or (os.cpu_count() or 1)
# The CPU memory arena keeps peak allocations around for reuse; off by default to keep RSS low.
ONNX_MEM_ARENA = _env_bool("ONNX_MEM_ARENA", "false")
# Encoder segment length in 8 s windows (0 = whole utterance in one pass)
ENCODER_SEGMENT_WINDOWS = int(os.getenv("ONNX_ENCODER_SEGMENT_WINDOWS", "1"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1024"))

STREAM_PARTIALS = _env_bool("STREAM_PARTIALS", "true")
STREAM_PARTIAL_INTERVAL_SEC = float(os.getenv("STREAM_PARTIAL_INTERVAL_SEC", "2.0"))
STREAM_PARTIAL_MAX_SEC = float(os.getenv("STREAM_PARTIAL_MAX_SEC", "10.0"))


def _mel_filterbank() -> np.ndarray:
    """Slaney-normalized mel filterbank, matching librosa.filters.mel(norm="slaney")."""
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    def hz_to_mel(freq: np.ndarray) -> np.ndarray:
        mels = freq / f_sp
        above = freq >= min_log_hz
        mels[above] = min_log_mel + np.log(freq[above] / min_log_hz) / logstep
        return mels

    def mel_to_hz(mels: np.ndarray) -> np.ndarray:
        freqs = mels * f_sp
        above = mels >= min_log_mel
        freqs[above] = min_log_hz * np.exp(logstep * (mels[above] - min_log_mel))
        return freqs

    fft_freqs = np.fft.rfftfreq(N_FFT, d=1.0 / RATE)
    mel_points = np.linspace(hz_to_mel(np.array([FMIN]))[0], hz_to_mel(np.array([FMAX]))[0], N_MELS + 2)
    band_hz = mel_to_hz(mel_points)

    diff = np.diff(band_hz)
    ramps = band_hz[:, np.newaxis] - fft_freqs[np.newaxis, :]
    weights = np.zeros((N_MELS, len(fft_freqs)), dtype=np.float32)
    for i in range(N_MELS):
        lower = -ramps[i] / diff[i]
        upper = ramps[i + 2] / diff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    weights *= (2.0 / (band_hz[2 : N_MELS + 2] - band_hz[:N_MELS]))[:, np.newaxis]
    return weights


def log_mel_spectrogram(audio: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """Whisper-compatible log-mel spectrogram, shape [1, n_mels, frames]."""
    if audio.shape[0] < N_FFT:
        audio = np.pad(audio, (0, N_FFT - audio.shape[0]))
    padded = np.pad(audio, N_FFT // 2, mode="reflect")
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)  # periodic Hann

    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH]
    spec = np.fft.rfft(frames * window, n=N_FFT, axis=-1)
    magnitudes = (np.abs(spec) ** 2).T.astype(np.float32)

    log_spec = np.log10(np.clip(mel_filters @ magnitudes, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    log_spec = log_spec[:, :-1]  # match WhisperFeatureExtractor's frame count
    return log_spec[np.newaxis, :, :].astype(np.float32)


def resample(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == RATE:
        return audio
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, RATE)
        return resample_poly(audio, RATE // g, sr // g).astype(np.float32)
    except ImportError:
        n = int(round(audio.shape[0] * RATE / sr))
        return np.interp(np.linspace(0, audio.shape[0] - 1, n), np.arange(audio.shape[0]), audio).astype(np.float32)


def _causal_mask(q_len: int, past_len: int) -> np.ndarray:
    """[1, 1, q_len, past_len + q_len] additive mask for the merged decoder."""
    new = np.triu(np.full((q_len, q_len), MASK_BLOCKED, dtype=np.float32), k=1)
    if past_len == 0:
        return new[np.newaxis, np.newaxis]
    past = np.zeros((q_len, past_len), dtype=np.float32)
    return np.concatenate([past, new], axis=-1)[np.newaxis, np.newaxis]


class OnnxStream(Stream):
    def __init__(self, engine: "OnnxEngine", language: Optional[str], context: str):
        self._engine = engine
        self._language = language
        self._context = context
        self._parts: List[np.ndarray] = []
        self._n = 0
        self._next_partial = int(STREAM_PARTIAL_INTERVAL_SEC * RATE)
        self.text = ""
        self.language = language or ""

    def _audio(self) -> np.ndarray:
        if len(self._parts) > 1:
            self._parts = [np.concatenate(self._parts)]
        return self._parts[0] if self._parts else np.zeros(0, dtype=np.float32)

    def feed(self, pcm: np.ndarray) -> bool:
        if pcm.size:
            self._parts.append(pcm.astype(np.float32, copy=False))
            self._n += pcm.size
        if not STREAM_PARTIALS or self._n < self._next_partial or self._n > STREAM_PARTIAL_MAX_SEC * RATE:
            return False
        self.text, self.language = self._engine.transcribe_one(self._audio(), self._language, self._context)
        # schedule from now, so a slow CPU never queues up back-to-back partials
        self._next_partial = self._n + int(STREAM_PARTIAL_INTERVAL_SEC * RATE)
        return True

    def finish(self) -> None:
        if self._n:
            self.text, self.language = self._engine.transcribe_one(self._audio(), self._language, self._context)
        self._parts = []


class OnnxEngine(Engine):
    name = "onnx"

    def __init__(self):
        self.model_dir = Path(os.getenv("ASR_MODEL_NAME", "/models/qwen3-asr-0.6b-onnx-int4-merged"))
        self._lock = threading.Lock()
        self._prefix_key: Optional[Tuple[int, ...]] = None
        self._prefix_kv: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._token_cache = {}

    def load(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = self.model_dir
        if not d.is_dir():
            raise RuntimeError(f"Model directory not found: {d}")
        logger.info(f"Loading ONNX model from {d} ({ONNX_THREADS} threads, mem arena {'on' if ONNX_MEM_ARENA else 'off'})")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = ONNX_THREADS
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_cpu_mem_arena = ONNX_MEM_ARENA
        providers = ["CPUExecutionProvider"]

        def session(name: str):
            return ort.InferenceSession(str(d / name), sess_options=opts, providers=providers)

        self._encoder = session("encoder.int4.onnx")
        self._encoder_inputs = [i.name for i in self._encoder.get_inputs()]

        self._merged = self._init = self._step = None
        if (d / MERGED_DECODER).is_file():
            self._merged = session(MERGED_DECODER)
            kv_shape = self._merged.get_outputs()[1].shape  # [layers, batch, kv_heads, seq, head_dim]
            self._empty_kv = np.zeros((kv_shape[0], 1, kv_shape[2], 0, kv_shape[4]), dtype=np.float32)
            logger.info("Using merged decoder")
        else:
            self._init = session("decoder_init.int4.onnx")
            self._step = session("decoder_step.int4.onnx")
            logger.info("Using split decoder")

        with open(d / "config.json", encoding="utf-8") as f:
            config = json.load(f)
        hidden = config["decoder"]["hidden_size"]
        # memory-mapped fp16 table: only the rows actually used are paged in
        self._embed = np.memmap(d / "embed_tokens.bin", dtype=np.float16, mode="r").reshape(-1, hidden)
        self._tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._mel_filters = _mel_filterbank()

    def warmup(self) -> None:
        self.transcribe_one(np.zeros(RATE, dtype=np.float32), "English", "")

    # ---- prompt ----

    def _tokens(self, text: str) -> List[int]:
        ids = self._token_cache.get(text)
        if ids is None:
            ids = self._tokenizer.encode(text, add_special_tokens=False).ids
            if len(self._token_cache) < 256:
                self._token_cache[text] = ids
        return ids

    # ---- encoder ----

    def _encode_mel(self, mel: np.ndarray) -> np.ndarray:
        inputs = {self._encoder_inputs[0]: mel}
        if len(self._encoder_inputs) > 1:
            inputs[self._encoder_inputs[1]] = np.array([mel.shape[2]], dtype=np.int64)
        return self._encoder.run(None, inputs)[0]

    def encode(self, audio: np.ndarray) -> np.ndarray:
        mel = log_mel_spectrogram(audio, self._mel_filters)
        seg = ENCODER_SEGMENT_WINDOWS * ENCODER_WINDOW_FRAMES
        if seg <= 0 or mel.shape[2] <= seg:
            return self._encode_mel(mel)
        parts = [self._encode_mel(np.ascontiguousarray(mel[:, :, i : i + seg])) for i in range(0, mel.shape[2], seg)]
        return np.concatenate(parts, axis=1)

    # ---- decoder ----

    def _run_merged(self, embeds, positions, past_k, past_v):
        return self._merged.run(
            ["logits", "present_keys", "present_values"],
            {
                "input_embeds": embeds,
                "position_ids": np.asarray(positions, dtype=np.int64)[np.newaxis, :],
                "attention_mask": _causal_mask(embeds.shape[1], past_k.shape[3]),
                "past_keys": past_k,
                "past_values": past_v,
            },
        )

    def _system_kv(self, system_ids: List[int]):
        key = tuple(system_ids)
        with self._lock:
            if self._prefix_key == key and self._prefix_kv is not None:
                return self._prefix_kv
        _, k, v = self._run_merged(
            self._embed[system_ids].astype(np.float32)[np.newaxis], np.arange(len(system_ids)), self._empty_kv, self._empty_kv
        )
        with self._lock:
            self._prefix_key, self._prefix_kv = key, (k, v)
        return k, v

    def _generate(self, features: np.ndarray, prompt: List[int], n_system: int, max_new: int) -> List[int]:
        audio_at = prompt.index(AUDIO_PAD)
        if self._merged is not None:
            past_k, past_v = self._system_kv(prompt[:n_system])
            rest = prompt[n_system:]
            embeds = self._embed[rest].astype(np.float32)[np.newaxis]
            at = audio_at - n_system
            embeds[0, at : at + features.shape[1]] = features[0]
            logits, past_k, past_v = self._run_merged(embeds, np.arange(n_system, len(prompt)), past_k, past_v)
        else:
            logits, past_k, past_v = self._init.run(
                ["logits", "present_keys", "present_values"],
                {
                    "input_ids": np.array(prompt, dtype=np.int64)[np.newaxis],
                    "position_ids": np.arange(len(prompt), dtype=np.int64)[np.newaxis],
                    "audio_features": features,
                    "audio_offset": np.array([audio_at], dtype=np.int64),
                },
            )

        token = int(np.argmax(logits[0, -1]))
        tokens = [token]
        pos = len(prompt)
        while token not in EOS_IDS and len(tokens) < max_new:
            emb = self._embed[token].astype(np.float32)[np.newaxis, np.newaxis]
            if self._merged is not None:
                logits, past_k, past_v = self._run_merged(emb, [pos], past_k, past_v)
            else:
                logits, past_k, past_v = self._step.run(
                    ["logits", "present_keys", "present_values"],
                    {
                        "input_embeds": emb,
                        "position_ids": np.array([[pos]], dtype=np.int64),
                        "past_keys": past_k,
                        "past_values": past_v,
                    },
                )
            token = int(np.argmax(logits[0, -1]))
            tokens.append(token)
            pos += 1
        return tokens

    def transcribe_one(self, audio: np.ndarray, language: Optional[str], context: str) -> Tuple[str, str]:
        """Transcribe 16 kHz float32 audio. Returns (text, language name)."""
        features = self.encode(audio)

        system = [IM_START, SYSTEM, NEWLINE, *(self._tokens(context) if context else []), IM_END, NEWLINE]
        lang_ids = self._tokens(f"language {language}") + [ASR_TEXT] if language else []
        prompt = (
            system
            + [IM_START, USER, NEWLINE, AUDIO_START]
            + [AUDIO_PAD] * features.shape[1]
            + [AUDIO_END, IM_END, NEWLINE, IM_START, ASSISTANT, NEWLINE]
            + lang_ids
        )
        # ~15 tokens/s covers fast Chinese speech; never cut a transcript short below 64 tokens
        max_new = min(MAX_NEW_TOKENS, 64 + int(audio.shape[0] / RATE * 15))
        tokens = self._generate(features, prompt, len(system), max_new)

        detected = language or ""
        if ASR_TEXT in tokens:
            i = tokens.index(ASR_TEXT)
            preamble = self._tokenizer.decode(tokens[:i], skip_special_tokens=True).strip()
            if preamble.lower().startswith("language "):
                detected = preamble[len("language ") :].strip()
            tokens = tokens[i + 1 :]
        text = self._tokenizer.decode([t for t in tokens if t not in EOS_IDS], skip_special_tokens=True).strip()
        return text, detected

    # ---- Engine API ----

    def transcribe(
        self, audios: Sequence[Audio], language: Optional[str], context: str = ""
    ) -> List[Tuple[str, str]]:
        out = []
        for wav, sr in audios:
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            out.append(self.transcribe_one(resample(wav.astype(np.float32), sr), language, context))
        return out

    def new_stream(self, language: Optional[str], context: str = "") -> Stream:
        return OnnxStream(self, language, context)

    def streaming_config(self) -> dict:
        return {
            "partials": STREAM_PARTIALS,
            "partial_interval_sec": STREAM_PARTIAL_INTERVAL_SEC,
            "partial_max_sec": STREAM_PARTIAL_MAX_SEC,
            "onnx_threads": ONNX_THREADS,
        }
