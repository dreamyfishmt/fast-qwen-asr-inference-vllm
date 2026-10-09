"""ONNX Runtime backend for Qwen3-ASR (CPU).

Runs an int4 ONNX export of Qwen3-ASR (encoder + decoder with KV cache): by default the
encoder of rhasspy/qwen3-asr-0.6b-onnx-int4-merged with a GQA decoder (see below). No PyTorch, no GPU.

The pipeline (mel front-end, chat-template prompt, greedy decode with a cached
system-turn KV) follows rhasspy/wyoming-faster-whisper's qwen3_asr_handler.py
(MIT License, Copyright (c) 2025 Michael Hansen).

Decoder layouts, chosen by the files in the model directory (or ONNX_DECODER):
  gqa    - decoder-*.onnx (+ .data) built with the onnxruntime-genai model builder
           (GroupQueryAttention, last-position logits), e.g. sorryhyun/qwen3-asr-onnx-gqa
           decoder-0.6b-fp32.onnx. Fastest on CPU: the per-token cost barely grows with length.
  merged - decoder_merged.int4.onnx (+ .data): one graph, KV cache + dynamic length
  split  - decoder_init.int4.onnx + decoder_step.int4.onnx (+ decoder_weights.int4.data)
The encoder (encoder.int4.onnx), embed_tokens.bin, tokenizer.json and config.json come from
rhasspy/qwen3-asr-0.6b-onnx-int4(-merged) in every case.

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
# Decoder graph file in the model dir (empty = auto-detect: gqa, then merged, then split)
ONNX_DECODER = os.getenv("ONNX_DECODER", "").strip()
# Encoder graph file (empty = encoder.int4.onnx, else encoder.onnx)
ONNX_ENCODER = os.getenv("ONNX_ENCODER", "").strip()
# Execution provider: cpu, or cuda (onnxruntime-gpu; falls back to CPU if CUDA can't be loaded)
ONNX_PROVIDER = os.getenv("ONNX_PROVIDER", "cpu").strip().lower()
ONNX_DEVICE_ID = int(os.getenv("ONNX_DEVICE_ID", "0"))

STREAM_PARTIALS = _env_bool("STREAM_PARTIALS", "true")
STREAM_PARTIAL_INTERVAL_SEC = float(os.getenv("STREAM_PARTIAL_INTERVAL_SEC", "2.0"))
STREAM_PARTIAL_MAX_SEC = float(os.getenv("STREAM_PARTIAL_MAX_SEC", "20.0"))
# Each partial and the final result continue from the previous partial (minus its last
# STREAM_UNFIXED_TOKEN_NUM tokens) instead of decoding the whole utterance again; the first
# STREAM_UNFIXED_CHUNK_NUM partials start from scratch. Same rollback strategy as qwen-asr's streaming.
STREAM_REUSE_PARTIAL = _env_bool("STREAM_REUSE_PARTIAL", "true")
# Whether the final result also continues from the last partial. Saves time on CPU, but an error in an
# early partial can survive into the final text; with a GPU a full re-decode is cheap, so turn it off there.
STREAM_FINAL_REUSE_PARTIAL = _env_bool("STREAM_FINAL_REUSE_PARTIAL", "true")
STREAM_UNFIXED_CHUNK_NUM = int(os.getenv("STREAM_UNFIXED_CHUNK_NUM", "2"))
STREAM_UNFIXED_TOKEN_NUM = int(os.getenv("STREAM_UNFIXED_TOKEN_NUM", "5"))


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


def last_logits_graph(src: Path) -> bytes:
    """Serialized copy of a decoder graph whose logits cover only the last position.

    The exported decoders apply lm_head (1024 -> 151936) to every input position. During
    prefill that is hundreds of audio positions, costing ~0.3 GFLOP and 0.6 MB of float32
    logits each, of which only the last row is used. A Slice in front of lm_head removes that
    work. Weights stay in the original external data file (see session.model_external_initializers_file_folder_path).
    """
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(src), load_external_data=False)
    graph = model.graph
    producer = next(n for n in graph.node if "logits" in n.output)
    hidden = producer.input[0]
    sliced = hidden + "/last_position"
    consts = {
        "last_pos/starts": np.array([-1], dtype=np.int64),
        "last_pos/ends": np.array([np.iinfo(np.int64).max], dtype=np.int64),
        "last_pos/axes": np.array([1], dtype=np.int64),
    }
    graph.initializer.extend(numpy_helper.from_array(v, k) for k, v in consts.items())
    idx = list(graph.node).index(producer)
    graph.node.insert(idx, helper.make_node("Slice", [hidden, *consts], [sliced], name="last_pos/Slice"))
    producer.input[0] = sliced
    return model.SerializeToString()


def _repeat_tail(tokens: List[int]) -> int:
    """Length of a degenerate repetition loop at the end of `tokens` (0 if none).

    Greedy decoding can lock into repeating the same phrase on noisy audio. A unit of n
    tokens repeated `reps` times in a row is treated as a loop; short units need more
    repeats so real speech ("对对对", "纠正纠正") is kept.
    """
    for n in range(1, 33):
        reps = 10 if n == 1 else 6 if n <= 3 else 4
        if len(tokens) < n * reps:
            break
        unit = tokens[-n:]
        if all(tokens[-(k + 1) * n : len(tokens) - k * n] == unit for k in range(1, reps)):
            return n * (reps - 1)  # keep one copy
    return 0


def _causal_mask(q_len: int, past_len: int) -> np.ndarray:
    """[1, 1, q_len, past_len + q_len] additive mask for the merged decoder."""
    new = np.triu(np.full((q_len, q_len), MASK_BLOCKED, dtype=np.float32), k=1)
    if past_len == 0:
        return new[np.newaxis, np.newaxis]
    past = np.zeros((q_len, past_len), dtype=np.float32)
    return np.concatenate([past, new], axis=-1)[np.newaxis, np.newaxis]


class Cancelled(Exception):
    pass


class OnnxStream(Stream):
    """Partials run on a background thread so audio keeps flowing; finish() aborts a running
    partial (within one decode step) and transcribes the whole utterance right away."""

    def __init__(self, engine: "OnnxEngine", language: Optional[str], context: str):
        self._engine = engine
        self._language = language
        self._context = context
        self._parts: List[np.ndarray] = []
        self._n = 0
        self._next_partial = int(STREAM_PARTIAL_INTERVAL_SEC * RATE)
        self._worker: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._result_lock = threading.Lock()
        self._version = 0
        self._reported = 0
        self._partial_tokens: List[int] = []
        self._partials_done = 0
        self.text = ""
        self.language = language or ""

    def _audio(self) -> np.ndarray:
        if len(self._parts) > 1:
            self._parts = [np.concatenate(self._parts)]
        return self._parts[0] if self._parts else np.zeros(0, dtype=np.float32)

    def _prefix(self) -> List[int]:
        if not STREAM_REUSE_PARTIAL:
            return []
        prefix = self._partial_tokens[:-STREAM_UNFIXED_TOKEN_NUM] if STREAM_UNFIXED_TOKEN_NUM else list(self._partial_tokens)
        if prefix and not self._language and ASR_TEXT not in prefix:
            return []  # rollback cut into the language preamble
        return prefix

    def _partial(self, audio: np.ndarray, prefix: List[int]) -> None:
        # skip rather than queue behind another utterance's transcription
        if not self._engine.infer_lock.acquire(blocking=False):
            return
        try:
            tokens = self._engine.transcribe_tokens(audio, self._language, self._context, prefix=prefix, cancel=self._cancel)
            text, language = self._engine.parse(tokens, self._language)
        except Cancelled:
            return
        except Exception as e:
            logger.warning(f"Partial transcription failed: {e}")
            return
        finally:
            self._engine.infer_lock.release()
        with self._result_lock:
            if not self._cancel.is_set():
                self.text, self.language = text, language
                self._partial_tokens = [t for t in tokens if t not in EOS_IDS]
                self._partials_done += 1
                self._version += 1

    def feed(self, pcm: np.ndarray, partial: bool = True) -> bool:
        if pcm.size:
            self._parts.append(pcm.astype(np.float32, copy=False))
            self._n += pcm.size
        idle = self._worker is None or not self._worker.is_alive()
        if partial and STREAM_PARTIALS and idle and self._next_partial <= self._n <= STREAM_PARTIAL_MAX_SEC * RATE:
            # next partial is due INTERVAL after this one started; a slow CPU just runs them back to back
            self._next_partial = self._n + int(STREAM_PARTIAL_INTERVAL_SEC * RATE)
            prefix = self._prefix() if self._partials_done >= STREAM_UNFIXED_CHUNK_NUM else []
            self._worker = threading.Thread(target=self._partial, args=(self._audio(), prefix), daemon=True)
            self._worker.start()
        with self._result_lock:
            changed = self._version != self._reported
            self._reported = self._version
        return changed

    def finish(self) -> None:
        self._cancel.set()
        if self._worker is not None:
            self._worker.join()
        if self._n:
            prefix = self._prefix() if STREAM_FINAL_REUSE_PARTIAL else []
            with self._engine.infer_lock:
                tokens = self._engine.transcribe_tokens(self._audio(), self._language, self._context, prefix=prefix)
            self.text, self.language = self._engine.parse(tokens, self._language)
        self._parts = []


class _GqaDecoder:
    """onnxruntime-genai export: per-layer past_key_values.N.{key,value}, attention_mask [1, total].

    The graph's I/O may be float32 or float16 (e.g. decoder-1.7b-fp16.onnx); embeddings are cast to
    match. With `device` set (CUDA), the KV cache stays on the GPU between steps via IOBinding and
    only the logits are copied back.
    """

    layout = "gqa"

    def __init__(self, sess, kv_heads: int, head_dim: int, device: Optional[str] = None, device_id: int = 0):
        import onnxruntime as ort

        self.sess = sess
        inputs = {i.name: i for i in sess.get_inputs()}
        self.dtype = np.float16 if inputs["inputs_embeds"].type == "tensor(float16)" else np.float32
        self.past_names = [n for n in inputs if n.startswith("past_key_values.")]
        self.present_names = ["present." + n[len("past_key_values."):] for n in self.past_names]
        self.outputs = ["logits"] + self.present_names
        self.device, self.device_id = device, device_id
        empty = np.zeros((1, kv_heads, 0, head_dim), dtype=self.dtype)
        if device:
            self._empty = ort.OrtValue.ortvalue_from_numpy(empty, device, device_id)
        else:
            self._empty = empty

    def empty_state(self):
        return ([self._empty] * len(self.past_names), 0)

    def extend(self, state, embeds: np.ndarray):
        past, length = state
        total = length + embeds.shape[1]
        embeds = embeds.astype(self.dtype, copy=False)
        mask = np.ones((1, total), dtype=np.int64)
        if not self.device:
            out = self.sess.run(
                self.outputs, {"inputs_embeds": embeds, "attention_mask": mask, **dict(zip(self.past_names, past))}
            )
            return out[0], (out[1:], total)

        binding = self.sess.io_binding()
        binding.bind_cpu_input("inputs_embeds", embeds)
        binding.bind_cpu_input("attention_mask", mask)
        for name, value in zip(self.past_names, past):
            binding.bind_ortvalue_input(name, value)
        binding.bind_output("logits", "cpu")
        for name in self.present_names:
            binding.bind_output(name, self.device, self.device_id)
        self.sess.run_with_iobinding(binding)
        out = binding.get_outputs()
        return out[0].numpy(), (out[1:], total)


class _MergedDecoder:
    """decoder_merged.int4.onnx: stacked past_keys/past_values, explicit positions and mask."""

    layout = "merged"

    def __init__(self, sess):
        self.sess = sess
        kv_shape = sess.get_outputs()[1].shape  # [layers, batch, kv_heads, seq, head_dim]
        self._empty = np.zeros((kv_shape[0], 1, kv_shape[2], 0, kv_shape[4]), dtype=np.float32)

    def empty_state(self):
        return (self._empty, self._empty)

    def extend(self, state, embeds: np.ndarray):
        past_k, past_v = state
        past = past_k.shape[3]
        logits, k, v = self.sess.run(
            ["logits", "present_keys", "present_values"],
            {
                "input_embeds": embeds,
                "position_ids": np.arange(past, past + embeds.shape[1], dtype=np.int64)[np.newaxis],
                "attention_mask": _causal_mask(embeds.shape[1], past),
                "past_keys": past_k,
                "past_values": past_v,
            },
        )
        return logits, (k, v)


class _SplitDecoder:
    """decoder_init (prompt ids + audio features, no KV input) + decoder_step (one token)."""

    layout = "split"

    def __init__(self, init, step):
        self.init, self.step = init, step

    def prefill(self, prompt: List[int], features: np.ndarray, audio_at: int):
        logits, k, v = self.init.run(
            ["logits", "present_keys", "present_values"],
            {
                "input_ids": np.array(prompt, dtype=np.int64)[np.newaxis],
                "position_ids": np.arange(len(prompt), dtype=np.int64)[np.newaxis],
                "audio_features": features,
                "audio_offset": np.array([audio_at], dtype=np.int64),
            },
        )
        return logits, (k, v)

    def extend(self, state, embeds: np.ndarray):
        past_k, past_v = state
        logits, k, v = self.step.run(
            ["logits", "present_keys", "present_values"],
            {
                "input_embeds": embeds,
                "position_ids": np.array([[past_k.shape[3]]], dtype=np.int64),
                "past_keys": past_k,
                "past_values": past_v,
            },
        )
        return logits, (k, v)


class OnnxEngine(Engine):
    name = "onnx"

    provider = ONNX_PROVIDER

    def __init__(self):
        self.model_dir = Path(os.getenv("ASR_MODEL_NAME", "/models/qwen3-asr-0.6b-onnx"))
        self._lock = threading.Lock()
        # one transcription at a time; partials skip instead of waiting (see OnnxStream)
        self.infer_lock = threading.Lock()
        self._prefix_key: Optional[Tuple[int, ...]] = None
        self._prefix_kv = None
        self._token_cache = {}

    def load(self) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = self.model_dir
        if not d.is_dir():
            raise RuntimeError(f"Model directory not found: {d}")
        logger.info(
            f"Loading ONNX model from {d} (provider {ONNX_PROVIDER}, {ONNX_THREADS} threads, "
            f"mem arena {'on' if ONNX_MEM_ARENA else 'off'})"
        )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = ONNX_THREADS
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_cpu_mem_arena = ONNX_MEM_ARENA
        providers = ["CPUExecutionProvider"]
        device = None
        if ONNX_PROVIDER == "cuda":
            # onnxruntime-gpu >= 1.21: load the CUDA/cuDNN libraries installed as nvidia-* pip packages
            if hasattr(ort, "preload_dlls"):
                try:
                    ort.preload_dlls()
                except Exception as e:
                    logger.warning(f"onnxruntime.preload_dlls() failed: {e}")
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers = [("CUDAExecutionProvider", {"device_id": ONNX_DEVICE_ID}), "CPUExecutionProvider"]
                device = "cuda"
            else:
                logger.warning("ONNX_PROVIDER=cuda but CUDAExecutionProvider is not available; using CPU")
        elif ONNX_PROVIDER != "cpu":
            raise RuntimeError(f"Unknown ONNX_PROVIDER: {ONNX_PROVIDER!r} (expected 'cpu' or 'cuda')")

        def session(name: str, last_logits: bool = False):
            path = d / name
            if last_logits:
                try:
                    model = last_logits_graph(path)
                    patched = ort.SessionOptions()
                    for attr in ("intra_op_num_threads", "inter_op_num_threads", "execution_mode",
                                 "graph_optimization_level", "enable_cpu_mem_arena"):
                        setattr(patched, attr, getattr(opts, attr))
                    patched.add_session_config_entry("session.model_external_initializers_file_folder_path", str(d))
                    return ort.InferenceSession(model, sess_options=patched, providers=providers)
                except Exception as e:
                    logger.warning(f"Could not patch {name} for last-position logits ({e}); using it unchanged")
            return ort.InferenceSession(str(path), sess_options=opts, providers=providers)

        encoder = ONNX_ENCODER or ("encoder.int4.onnx" if (d / "encoder.int4.onnx").is_file() else "encoder.onnx")
        self._encoder = session(encoder)
        if device and "CUDAExecutionProvider" not in self._encoder.get_providers():
            logger.warning("CUDA could not be initialized for ONNX Runtime; running on CPU")
            device = None
            providers[:] = ["CPUExecutionProvider"]  # don't retry CUDA for the decoder
        self._encoder_inputs = [i.name for i in self._encoder.get_inputs()]
        self._encoder_dtype = np.float16 if self._encoder.get_inputs()[0].type == "tensor(float16)" else np.float32
        self.provider = "cuda" if device else "cpu"

        with open(d / "config.json", encoding="utf-8") as f:
            dec_cfg = json.load(f)["decoder"]

        gqa = sorted(p.name for p in d.glob("decoder-*.onnx"))
        if ONNX_DECODER:
            name = ONNX_DECODER
        elif gqa:
            # fp16 I/O suits the GPU; fp32 I/O is faster on CPU
            prefer = "fp16" if device else "fp32"
            name = next((n for n in gqa if prefer in n), gqa[0])
        elif (d / MERGED_DECODER).is_file():
            name = MERGED_DECODER
        else:
            name = "decoder_init.int4.onnx"

        if name == "decoder_init.int4.onnx":
            self._decoder = _SplitDecoder(session(name, last_logits=True), session("decoder_step.int4.onnx"))
        elif name == MERGED_DECODER:
            self._decoder = _MergedDecoder(session(name, last_logits=True))
        else:
            self._decoder = _GqaDecoder(
                session(name), dec_cfg["num_key_value_heads"], dec_cfg["head_dim"], device, ONNX_DEVICE_ID
            )
        logger.info(f"Using {self._decoder.layout} decoder ({name}), encoder {encoder}, provider {self.provider}")

        hidden = dec_cfg["hidden_size"]
        # memory-mapped fp16 table: only the rows actually used are paged in
        self._embed = np.memmap(d / "embed_tokens.bin", dtype=np.float16, mode="r").reshape(-1, hidden)
        self._tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._mel_filters = _mel_filterbank()

    def warmup(self) -> None:
        with self.infer_lock:
            self.transcribe_one(np.zeros(RATE, dtype=np.float32), "English", "")

    # ---- prompt ----

    def _tokens(self, text: str) -> List[int]:
        ids = self._token_cache.get(text)
        if ids is None:
            ids = self._tokenizer.encode(text, add_special_tokens=False).ids
            if len(self._token_cache) < 256:
                self._token_cache[text] = ids
        return ids

    def _embeds(self, ids: Sequence[int]) -> np.ndarray:
        return self._embed[list(ids)].astype(np.float32)[np.newaxis]

    # ---- encoder ----

    def _encode_mel(self, mel: np.ndarray) -> np.ndarray:
        inputs = {self._encoder_inputs[0]: mel.astype(self._encoder_dtype, copy=False)}
        if len(self._encoder_inputs) > 1:
            inputs[self._encoder_inputs[1]] = np.array([mel.shape[2]], dtype=np.int64)
        return self._encoder.run(None, inputs)[0].astype(np.float32, copy=False)

    def encode(self, audio: np.ndarray, cancel: Optional[threading.Event] = None) -> np.ndarray:
        mel = log_mel_spectrogram(audio, self._mel_filters)
        seg = ENCODER_SEGMENT_WINDOWS * ENCODER_WINDOW_FRAMES
        if seg <= 0 or mel.shape[2] <= seg:
            return self._encode_mel(mel)
        parts = []
        for i in range(0, mel.shape[2], seg):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            parts.append(self._encode_mel(np.ascontiguousarray(mel[:, :, i : i + seg])))
        return np.concatenate(parts, axis=1)

    # ---- decoder ----

    def _system_state(self, system_ids: List[int]):
        """KV state after the system turn. It precedes the audio, so it is reused across utterances."""
        key = tuple(system_ids)
        with self._lock:
            if self._prefix_key == key and self._prefix_kv is not None:
                return self._prefix_kv
        _, state = self._decoder.extend(self._decoder.empty_state(), self._embeds(system_ids))
        with self._lock:
            self._prefix_key, self._prefix_kv = key, state
        return state

    def _generate(
        self, features: np.ndarray, prompt: List[int], n_system: int, max_new: int, cancel: Optional[threading.Event] = None
    ) -> List[int]:
        audio_at = prompt.index(AUDIO_PAD)
        if isinstance(self._decoder, _SplitDecoder):
            logits, state = self._decoder.prefill(prompt, features, audio_at)
        else:
            embeds = self._embeds(prompt[n_system:])
            at = audio_at - n_system
            embeds[0, at : at + features.shape[1]] = features[0]
            logits, state = self._decoder.extend(self._system_state(prompt[:n_system]), embeds)

        if cancel is not None and cancel.is_set():
            raise Cancelled()
        token = int(np.argmax(logits[0, -1]))
        tokens = [token]
        while token not in EOS_IDS and len(tokens) < max_new:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            logits, state = self._decoder.extend(state, self._embeds([token]))
            token = int(np.argmax(logits[0, -1]))
            tokens.append(token)
            loop = _repeat_tail(tokens)
            if loop:
                logger.warning(f"Stopped a repetition loop after {len(tokens)} tokens")
                return tokens[:-loop]
        return tokens

    def transcribe_tokens(
        self,
        audio: np.ndarray,
        language: Optional[str],
        context: str,
        prefix: Sequence[int] = (),
        cancel: Optional[threading.Event] = None,
    ) -> List[int]:
        """Generated tokens for 16 kHz float32 audio (language preamble included when not forced).

        `prefix`: tokens of an earlier transcription of the same, shorter audio. They are fed as
        the start of the answer, so only the rest has to be generated one token at a time.
        """
        features = self.encode(audio, cancel)

        system = [IM_START, SYSTEM, NEWLINE, *(self._tokens(context) if context else []), IM_END, NEWLINE]
        lang_ids = self._tokens(f"language {language}") + [ASR_TEXT] if language else []
        prompt = (
            system
            + [IM_START, USER, NEWLINE, AUDIO_START]
            + [AUDIO_PAD] * features.shape[1]
            + [AUDIO_END, IM_END, NEWLINE, IM_START, ASSISTANT, NEWLINE]
            + lang_ids
            + list(prefix)
        )
        # ~15 tokens/s covers fast Chinese speech; never cut a transcript short below 64 tokens
        max_new = min(MAX_NEW_TOKENS, 64 + int(audio.shape[0] / RATE * 15)) - len(prefix)
        if max_new <= 0:
            return list(prefix)
        return list(prefix) + self._generate(features, prompt, len(system), max_new, cancel)

    def parse(self, tokens: Sequence[int], language: Optional[str]) -> Tuple[str, str]:
        """(text, language name) from generated tokens."""
        tokens = list(tokens)
        detected = language or ""
        if ASR_TEXT in tokens:
            i = tokens.index(ASR_TEXT)
            preamble = self._tokenizer.decode(tokens[:i], skip_special_tokens=True).strip()
            if preamble.lower().startswith("language "):
                detected = preamble[len("language ") :].strip()
            tokens = tokens[i + 1 :]
        text = self._tokenizer.decode([t for t in tokens if t not in EOS_IDS], skip_special_tokens=True).strip()
        return text, detected

    def transcribe_one(
        self, audio: np.ndarray, language: Optional[str], context: str, cancel: Optional[threading.Event] = None
    ) -> Tuple[str, str]:
        """Transcribe 16 kHz float32 audio. Returns (text, language name). Raises Cancelled if `cancel` is set."""
        return self.parse(self.transcribe_tokens(audio, language, context, cancel=cancel), language)

    # ---- Engine API ----

    def transcribe(
        self, audios: Sequence[Audio], language: Optional[str], context: str = ""
    ) -> List[Tuple[str, str]]:
        out = []
        for wav, sr in audios:
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            with self.infer_lock:
                out.append(self.transcribe_one(resample(wav.astype(np.float32), sr), language, context))
        return out

    def new_stream(self, language: Optional[str], context: str = "") -> Stream:
        return OnnxStream(self, language, context)

    def streaming_config(self) -> dict:
        return {
            "partials": STREAM_PARTIALS,
            "partial_interval_sec": STREAM_PARTIAL_INTERVAL_SEC,
            "partial_max_sec": STREAM_PARTIAL_MAX_SEC,
            "reuse_partial": STREAM_REUSE_PARTIAL,
            "final_reuse_partial": STREAM_FINAL_REUSE_PARTIAL,
            "unfixed_chunk_num": STREAM_UNFIXED_CHUNK_NUM,
            "unfixed_token_num": STREAM_UNFIXED_TOKEN_NUM,
            "onnx_threads": ONNX_THREADS,
            "provider": getattr(self, "provider", ONNX_PROVIDER),
        }
