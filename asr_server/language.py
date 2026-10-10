"""Language codes -> Qwen3-ASR language names, and Traditional Chinese output via OpenCC."""

from typing import Callable, Dict, Optional, Tuple

from .config import OPENCC_HK_CONFIG, OPENCC_TW_CONFIG

SUPPORTED_LANGUAGES = [
    "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish", "Portuguese",
    "Indonesian", "Italian", "Korean", "Russian", "Thai", "Vietnamese", "Japanese", "Turkish",
    "Hindi", "Malay", "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino",
    "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
]

LANGUAGE_MAP = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "ru": "Russian", "pt": "Portuguese", "nl": "Dutch", "tr": "Turkish",
    "sv": "Swedish", "id": "Indonesian", "vi": "Vietnamese",
    "hi": "Hindi", "ar": "Arabic", "yue": "Cantonese", "th": "Thai",
    "ms": "Malay", "da": "Danish", "fi": "Finnish", "pl": "Polish",
    "cs": "Czech", "fil": "Filipino", "tl": "Filipino", "fa": "Persian",
    "el": "Greek", "ro": "Romanian", "hu": "Hungarian", "mk": "Macedonian",
}

# Chinese script variants -> OpenCC config (None = keep the model's Simplified output)
ZH_SCRIPT_MAP = {
    "zh-cn": None, "zh-sg": None, "zh-hans": None,
    "zh-tw": OPENCC_TW_CONFIG, "zh-hant": OPENCC_TW_CONFIG,
    "zh-hk": OPENCC_HK_CONFIG, "zh-mo": OPENCC_HK_CONFIG,
}


def map_language(lang_code: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Map a language code (ISO 639-1 or BCP-47, e.g. "en", "zh-CN", "zh-TW")
    to (Qwen full language name, OpenCC config or None).
    Qwen3-ASR has a single "Chinese" language, so Traditional output is
    produced by converting the Simplified transcript with OpenCC.
    Full language names (e.g. "Chinese") are accepted as well.
    Raises ValueError for languages the model does not support.
    """
    if not lang_code or not lang_code.strip():
        return None, None
    code = lang_code.strip().lower().replace("_", "-")
    if code in ZH_SCRIPT_MAP:
        return "Chinese", ZH_SCRIPT_MAP[code]
    base = code.split("-", 1)[0]
    if base in LANGUAGE_MAP:
        return LANGUAGE_MAP[base], None
    name = lang_code.strip().capitalize()
    if name in SUPPORTED_LANGUAGES:
        return name, None
    raise ValueError(f"Unsupported language: {lang_code}")


_opencc_converters: Dict[str, Callable[[str], str]] = {}


def get_converter(config: Optional[str]) -> Callable[[str], str]:
    """Return a cached text converter for the OpenCC config (identity if None)."""
    if config is None:
        return lambda text: text
    if config not in _opencc_converters:
        import opencc
        _opencc_converters[config] = opencc.OpenCC(config).convert
    return _opencc_converters[config]
