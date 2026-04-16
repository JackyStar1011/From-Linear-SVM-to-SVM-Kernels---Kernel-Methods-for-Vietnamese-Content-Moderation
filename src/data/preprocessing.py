from functools import partial
import re
import unicodedata
from typing import Any, Callable


WHITESPACE_RE = re.compile(r"\s+")
PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_text(
    text: Any,
    unicode_normalization: str = "NFC",
    collapse_whitespace: bool = True,
    strip: bool = True,
    lowercase: bool = False,
    preserve_punctuation: bool = True,
) -> str:
    value = "" if text is None else str(text)

    if unicode_normalization:
        value = unicodedata.normalize(unicode_normalization, value)
    if collapse_whitespace:
        value = WHITESPACE_RE.sub(" ", value)
    if strip:
        value = value.strip()
    if lowercase:
        value = value.lower()
    if not preserve_punctuation:
        value = PUNCTUATION_RE.sub("", value)
        if collapse_whitespace:
            value = WHITESPACE_RE.sub(" ", value)
        if strip:
            value = value.strip()
    return value


def build_preprocessor(preprocessing_config: dict[str, Any]) -> Callable[[Any], str]:
    return partial(normalize_text, **preprocessing_config)
