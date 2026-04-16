import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MULTISPACE_RE = re.compile(r"\s+")
REPEAT_PUNCT_RE = re.compile(r"([!?.,:;])\1{1,}")
REPEAT_CHAR_RE = re.compile(r"(.)\1{3,}")


def preprocess_light(text: str) -> str:
    text = str(text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def preprocess_medium(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def preprocess_strong(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = REPEAT_PUNCT_RE.sub(r"\1", text)
    text = REPEAT_CHAR_RE.sub(lambda m: m.group(1) * 2, text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


PREPROCESSORS = {
    "light": preprocess_light,
    "medium": preprocess_medium,
    "strong": preprocess_strong,
}


def preprocess_text(text: str, mode: str = "strong") -> str:
    if mode not in PREPROCESSORS:
        raise ValueError(f"unknown preprocessing mode: {mode}")
    return PREPROCESSORS[mode](text)
