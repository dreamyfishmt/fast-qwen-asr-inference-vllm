"""Forced-aligner text handling and segment building (no model needed)."""

import numpy as np

from asr_server.engines.onnx_aligner import audio_token_count, fix_timestamps, split_words
from asr_server.timeline import assign_speakers, build_segments, word_offsets, words_from_alignment


def test_split_words():
    assert split_words("Hello, world! It's 2026.", "English") == ["Hello", "world", "It's", "2026"]
    assert split_words("我们用Kubernetes部署。", "Chinese") == ["我", "们", "用", "Kubernetes", "部", "署"]
    assert split_words("e-mail", None) == ["email"]
    # Japanese: kana are separate units too
    assert split_words("これはテスト", "Japanese") == ["こ", "れ", "は", "テ", "ス", "ト"]
    assert split_words("안녕하세요 여러분", "Korean") == ["안녕하세요", "여러분"]
    assert split_words(" ,. ", "English") == []


def test_fix_timestamps():
    assert fix_timestamps([0, 80, 160, 240]) == [0, 80, 160, 240]
    # a single outlier takes the value of its nearer neighbour
    assert fix_timestamps([0, 80, 9000, 240, 320]) == [0, 80, 80, 240, 320]
    # a run of three or more is interpolated between its neighbours
    assert fix_timestamps([0, 9000, 8000, 7000, 400, 500]) == [0, 100, 200, 300, 400, 500]
    assert fix_timestamps([]) == []


def test_audio_token_count():
    # 100 mel frames (1 s) -> 13 encoder outputs, as Qwen3-ASR's _get_feat_extract_output_lengths
    assert audio_token_count(100) == 13
    assert audio_token_count(3000) == 390
    assert audio_token_count(150) == 13 + 7


def test_word_offsets():
    text = "Hello, world! It's fine."
    words = [{"word": w} for w in ["Hello", "world", "It's", "fine"]]
    assert word_offsets(text, words) == [0, 7, 14, 19]
    assert word_offsets(text, [{"word": "nope"}]) is None


def _words(*spec):
    return [{"word": w, "start": s, "end": e} for w, s, e in spec]


def test_segments_keep_punctuation_and_break_at_pauses():
    words = _words(("你", 0.0, 0.2), ("好", 0.2, 0.4), ("再", 2.0, 2.2), ("见", 2.2, 2.4))
    segments = build_segments(words, "你好。再见！", 3.0)
    assert segments == [{"start": 0.0, "end": 0.4, "text": "你好。"}, {"start": 2.0, "end": 2.4, "text": "再见！"}]
    # text that doesn't match the words falls back to joining them
    assert build_segments(words, "something else", 3.0)[0]["text"] == "你好"
    assert build_segments([], "text", 3.0) == [{"start": 0.0, "end": 3.0, "text": "text"}]
    # segment texts are converted (OpenCC) after slicing the original text
    assert build_segments(words, "你好。再见！", 3.0, convert=str.upper)[0]["text"] == "你好。"


def test_speakers_split_segments():
    words = _words(("Hi", 0.1, 0.3), ("there.", 0.35, 0.6), ("Hello", 0.7, 0.9), ("back", 3.1, 3.2))
    for w in words:
        w["word"] = w["word"].rstrip(".")
    turns = [{"start": 0.0, "end": 0.62, "speaker": 0}, {"start": 0.65, "end": 1.0, "speaker": 1}]
    assign_speakers(words, turns)
    # "back" overlaps no turn and goes to the nearest one
    assert [w["speaker"] for w in words] == [0, 0, 1, 1]
    segments = build_segments(words, "Hi there. Hello back", 4.0)
    assert [(s["text"], s["speaker"]) for s in segments] == [("Hi there.", 0), ("Hello", 1), ("back", 1)]


def test_words_from_alignment_accepts_objects():
    class Item:
        def __init__(self, text, start, end):
            self.text, self.start_time, self.end_time = text, start, end

    class Result:
        items = [Item("a", np.float32(0.1), 0.2)]

    assert words_from_alignment(Result()) == [{"word": "a", "start": 0.1, "end": 0.2}]
    assert words_from_alignment({"items": [{"text": "b", "start_time": 1, "end_time": 2}]}) == [{"word": "b", "start": 1.0, "end": 2.0}]
    assert words_from_alignment(None) == []
