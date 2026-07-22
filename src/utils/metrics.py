from __future__ import annotations

import re
import unicodedata
from math import sqrt
from typing import Iterable

import jiwer

# The paper specifies lowercasing and punctuation removal. Unicode letters,
# combining marks, whitespace, and digits are preserved for African orthographies.
_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = _PUNCTUATION.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def compute_metrics(references: Iterable[str], predictions: Iterable[str]) -> dict[str, float | bool]:
    references = [normalize_text(value) for value in references]
    predictions = [normalize_text(value) for value in predictions]
    wer = jiwer.wer(references, predictions)
    cer = jiwer.cer(references, predictions)
    return {"wer": wer, "cer": cer, "wer_percent": wer * 100, "cer_percent": cer * 100,
            "wer_over_100": wer > 1.0}


def wer_confidence_interval(wer: float, n: int = 300) -> float:
    """Paper-requested normal approximation, returned as a fraction."""
    return 1.96 * sqrt(wer * (1 - wer) / n) if 0 <= wer <= 1 else float("nan")
