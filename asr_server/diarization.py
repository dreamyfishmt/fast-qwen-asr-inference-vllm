"""Speaker diarization ("who spoke when") with sherpa-onnx.

sherpa-onnx's offline pipeline: pyannote segmentation-3.0 (ONNX) finds speech and local speaker changes,
a speaker embedding model (3D-Speaker CAM++, ONNX) describes each piece, and the pieces are clustered into
speakers. It runs on the CPU with sherpa-onnx's own copy of ONNX Runtime, next to either backend.
"""

import logging
import os
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

from .audio import MODEL_RATE, resample
from .config import env

logger = logging.getLogger(__name__)

GROUP = "Speaker diarization"

ENABLE_DIARIZATION = env("ENABLE_DIARIZATION", False, "Load the speaker diarization models (`diarize=true`)", group=GROUP)
DIARIZATION_MODEL_NAME = env(
    "DIARIZATION_MODEL_NAME", "/models/speaker-diarization",
    "Directory with `segmentation.onnx` and `embedding.onnx` (see `scripts/download-models.sh diarization`)", group=GROUP,
)
DIARIZATION_THRESHOLD = env(
    "DIARIZATION_THRESHOLD", 0.5,
    "Clustering threshold when the number of speakers is not given: higher = fewer speakers", group=GROUP,
)
DIARIZATION_MIN_DURATION_ON = env("DIARIZATION_MIN_DURATION_ON", 0.3, "Shortest speech turn kept, in seconds", group=GROUP)
DIARIZATION_MIN_DURATION_OFF = env(
    "DIARIZATION_MIN_DURATION_OFF", 0.5, "Pauses of one speaker shorter than this are bridged, in seconds", group=GROUP,
)
DIARIZATION_THREADS = env("DIARIZATION_THREADS", 0, "Threads per diarization model (0 = all cores)", group=GROUP) or (os.cpu_count() or 1)
MAX_SPEAKERS = 32


class Diarizer:
    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir or DIARIZATION_MODEL_NAME)
        self._lock = threading.Lock()  # one run at a time; also guards the clustering config
        self._sd = None
        self._num_speakers: Optional[int] = None

    def _config(self, num_speakers: Optional[int]):
        import sherpa_onnx as so

        d = self.model_dir
        return so.OfflineSpeakerDiarizationConfig(
            segmentation=so.OfflineSpeakerSegmentationModelConfig(
                pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(d / "segmentation.onnx")),
                num_threads=DIARIZATION_THREADS,
            ),
            embedding=so.SpeakerEmbeddingExtractorConfig(model=str(d / "embedding.onnx"), num_threads=DIARIZATION_THREADS),
            clustering=so.FastClusteringConfig(num_clusters=num_speakers or -1, threshold=DIARIZATION_THRESHOLD),
            min_duration_on=DIARIZATION_MIN_DURATION_ON,
            min_duration_off=DIARIZATION_MIN_DURATION_OFF,
        )

    def load(self) -> None:
        import sherpa_onnx as so

        for name in ("segmentation.onnx", "embedding.onnx"):
            if not (self.model_dir / name).is_file():
                raise RuntimeError(f"Diarization model not found: {self.model_dir / name}")
        cfg = self._config(None)
        if not cfg.validate():
            raise RuntimeError(f"Invalid diarization models in {self.model_dir}")
        logger.info(f"Loading speaker diarization models from {self.model_dir} ({DIARIZATION_THREADS} threads)")
        self._sd = so.OfflineSpeakerDiarization(cfg)
        logger.info("Speaker diarization loaded.")

    def warmup(self) -> None:
        self.diarize(np.zeros(MODEL_RATE, dtype=np.float32), MODEL_RATE)

    def diarize(self, wav: np.ndarray, sr: int, num_speakers: Optional[int] = None) -> List[dict]:
        """Speaker turns [{"start", "end", "speaker"}] (seconds; speakers numbered 0, 1, ... by first turn).

        `num_speakers`: the number of speakers if known, else they are found by DIARIZATION_THRESHOLD.
        """
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        audio = np.ascontiguousarray(resample(wav.astype(np.float32, copy=False), sr, MODEL_RATE))
        if audio.shape[0] == 0:
            return []
        with self._lock:
            if num_speakers != self._num_speakers:
                self._sd.set_config(self._config(num_speakers))  # only the clustering settings change
                self._num_speakers = num_speakers
            result = self._sd.process(audio).sort_by_start_time()
        labels = {}
        turns = []
        for seg in result:
            speaker = labels.setdefault(seg.speaker, len(labels))
            turns.append({"start": round(float(seg.start), 3), "end": round(float(seg.end), 3), "speaker": speaker})
        return turns
