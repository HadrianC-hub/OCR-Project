"""
metrics.py  —  CER, WER y exactitud por línea (sin dependencias externas)
"""

from typing import List


def _levenshtein(a: List, b: List) -> int:
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[n]


def cer(hypothesis: str, reference: str) -> float:
    """Tasa de Error de Carácter (Character Error Rate)."""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return _levenshtein(list(hypothesis), list(reference)) / len(reference)


def wer(hypothesis: str, reference: str) -> float:
    """Tasa de Error de Palabra (Word Error Rate)."""
    ref_w = reference.split()
    hyp_w = hypothesis.split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return _levenshtein(hyp_w, ref_w) / len(ref_w)


def line_accuracy(hypotheses: List[str], references: List[str]) -> float:
    """Proporción de líneas exactamente correctas."""
    if not references:
        return 0.0
    return sum(h == r for h, r in zip(hypotheses, references)) / len(references)