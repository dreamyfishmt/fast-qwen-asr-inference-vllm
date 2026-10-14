"""Word timings -> timed segments, and speaker turns -> speakers of words and segments."""

from typing import Callable, List, Optional, Sequence

from .language import join_text

# Start a new segment after a pause this long, or once a segment is this long
SEGMENT_GAP_SEC = 0.8
SEGMENT_MAX_SEC = 10.0


def words_from_alignment(alignment) -> List[dict]:
    """Aligner output ({"items": [{"text", "start_time", "end_time"}]}, or qwen_asr's ForcedAlignResult)
    -> [{"word", "start", "end"}]."""
    if alignment is None:
        return []
    if isinstance(alignment, dict):
        items = alignment.get("items", [])
    else:  # qwen_asr ForcedAlignResult (iterable of ForcedAlignItem)
        items = getattr(alignment, "items", alignment)
    words = []
    for it in items:
        get = it.get if isinstance(it, dict) else lambda k, it=it: getattr(it, k)
        words.append({"word": get("text"), "start": round(float(get("start_time")), 3), "end": round(float(get("end_time")), 3)})
    return words


def assign_speakers(words: List[dict], turns: Sequence[dict]) -> None:
    """Set each word's "speaker" to the turn it overlaps most, or else the nearest turn."""
    if not turns:
        return
    for w in words:
        best, best_key = None, None
        for t in turns:
            overlap = min(w["end"], t["end"]) - max(w["start"], t["start"])
            distance = max(t["start"] - w["end"], w["start"] - t["end"], 0.0)
            key = (overlap > 0, overlap if overlap > 0 else -distance)
            if best_key is None or key > best_key:
                best, best_key = t, key
        w["speaker"] = best["speaker"]


def word_offsets(text: str, words: Sequence[dict]) -> Optional[List[int]]:
    """Index in `text` where each word starts, or None if the words aren't `text`'s words in order.

    The aligner's words are the transcript's letters and digits with punctuation removed, so each one
    is found character by character after the previous one.
    """
    offsets = []
    pos = 0
    for w in words:
        start = None
        for ch in w["word"]:
            pos = text.find(ch, pos)
            if pos < 0:
                return None
            if start is None:
                start = pos
            pos += 1
        if start is None:
            return None
        offsets.append(start)
    return offsets


def build_segments(
    words: List[dict], text: str, duration: float, convert: Callable[[str], str] = lambda s: s,
) -> List[dict]:
    """Group timed words into segments: [{"start", "end", "text"[, "speaker"]}].

    A segment ends at a pause longer than SEGMENT_GAP_SEC, after SEGMENT_MAX_SEC, or where the speaker
    changes (words with a "speaker"). Segment texts are slices of `text`, so they keep its punctuation,
    converted with `convert` (OpenCC). Without word timings, one segment spans the whole file.
    """
    if not words:
        return [{"start": 0.0, "end": duration, "text": convert(text)}] if text else []
    groups, current = [], []
    for i, w in enumerate(words):
        prev = words[current[-1]] if current else None
        if prev and (
            w["start"] - prev["end"] > SEGMENT_GAP_SEC
            or w["end"] - words[current[0]]["start"] > SEGMENT_MAX_SEC
            or w.get("speaker") != prev.get("speaker")
        ):
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)

    offsets = word_offsets(text, words)
    segments = []
    for k, group in enumerate(groups):
        if offsets is not None:
            begin = 0 if k == 0 else offsets[group[0]]
            end = offsets[groups[k + 1][0]] if k + 1 < len(groups) else len(text)
            seg_text = text[begin:end].strip()
        else:
            seg_text = join_text(words[i]["word"] for i in group)
        first, last = words[group[0]], words[group[-1]]
        seg = {"start": first["start"], "end": last["end"], "text": convert(seg_text)}
        if "speaker" in first:
            seg["speaker"] = first["speaker"]
        segments.append(seg)
    return segments
