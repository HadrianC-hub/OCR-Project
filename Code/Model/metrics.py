"""
metrics.py  —  Métricas exhaustivas estado del arte para OCR

Métricas implementadas:
  - CER  (Character Error Rate) con desglose S/I/D
  - WER  (Word Error Rate) con desglose S/I/D
  - 1-NED (1 - Normalized Edit Distance)  ← estándar en papers modernos
  - Line Accuracy (exact match)
  - Character Accuracy / Word Accuracy
  - BLEU-4  (usado en OCR-as-sequence papers)
  - Carácter Precision / Recall / F1
  - Palabra   Precision / Recall / F1
  - Longitud media pred vs referencia
  - Histograma de errores por longitud de texto
"""

from typing import List, Tuple, Dict
from collections import Counter
import math


# ------------------------------------------------------------------ #
#  Levenshtein con desglose S/I/D                                     #
# ------------------------------------------------------------------ #

def _levenshtein_ops(a: List, b: List) -> Tuple[int, int, int, int]:
    """
    Distancia de Levenshtein con conteo de operaciones.

    Retorna
    -------
    (distance, substitutions, insertions, deletions)
    """
    m, n = len(a), len(b)
    # dp[i][j] = (dist, S, I, D)
    INF = (10**9, 0, 0, 0)
    prev = [(j, 0, j, 0) for j in range(n + 1)]   # solo inserciones

    for i in range(1, m + 1):
        curr = [(i, 0, 0, i)] + [INF] * n          # solo eliminaciones
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                # Match
                d, s, ins, d2 = prev[j - 1]
                curr[j] = (d, s, ins, d2)
            else:
                # Substitución (desde prev[j-1])
                sub   = (prev[j-1][0] + 1, prev[j-1][1] + 1, prev[j-1][2], prev[j-1][3])
                # Inserción (desde curr[j-1])
                ins_op = (curr[j-1][0] + 1, curr[j-1][1], curr[j-1][2] + 1, curr[j-1][3])
                # Eliminación (desde prev[j])
                del_op = (prev[j][0] + 1, prev[j][1], prev[j][2], prev[j][3] + 1)
                best = min(sub, ins_op, del_op, key=lambda x: x[0])
                curr[j] = best
        prev = curr

    dist, s, ins, d = prev[n]
    return dist, s, ins, d


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


# ------------------------------------------------------------------ #
#  Métricas básicas                                                    #
# ------------------------------------------------------------------ #

def cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate."""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return _levenshtein(list(hypothesis), list(reference)) / len(reference)


def wer(hypothesis: str, reference: str) -> float:
    """Word Error Rate."""
    ref_w = reference.split()
    hyp_w = hypothesis.split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return _levenshtein(hyp_w, ref_w) / len(ref_w)


def ned(hypothesis: str, reference: str) -> float:
    """
    Normalized Edit Distance (NED).
    NED = edit_distance / max(len(hyp), len(ref))
    Muy usado en papers post-2018 (e.g. STR benchmark).
    """
    if len(hypothesis) == 0 and len(reference) == 0:
        return 0.0
    denom = max(len(hypothesis), len(reference))
    return _levenshtein(list(hypothesis), list(reference)) / denom


def one_minus_ned(hypothesis: str, reference: str) -> float:
    """1-NED: métrica de similitud (1.0 = perfecto)."""
    return 1.0 - ned(hypothesis, reference)


def line_accuracy(hypotheses: List[str], references: List[str]) -> float:
    """Proporción de líneas exactamente correctas."""
    if not references:
        return 0.0
    return sum(h == r for h, r in zip(hypotheses, references)) / len(references)


# ------------------------------------------------------------------ #
#  BLEU-4 sin dependencias externas                                   #
# ------------------------------------------------------------------ #

def _ngrams(tokens: List, n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def bleu_score(
    hypotheses: List[str],
    references: List[str],
    max_n: int = 4,
    tokenize: str = "char",   # "char" | "word"
) -> float:
    """
    BLEU-N (corpus level) a nivel de carácter o palabra.
    Implementación propia, sin NLTK.
    """
    if tokenize == "char":
        hyp_tok = [list(h) for h in hypotheses]
        ref_tok = [list(r) for r in references]
    else:
        hyp_tok = [h.split() for h in hypotheses]
        ref_tok = [r.split() for r in references]

    # Brevity penalty
    hyp_len = sum(len(h) for h in hyp_tok)
    ref_len  = sum(len(r) for r in ref_tok)
    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / hyp_len)

    precisions = []
    for n in range(1, max_n + 1):
        clip_count, total_count = 0, 0
        for hyp, ref in zip(hyp_tok, ref_tok):
            hyp_ng  = _ngrams(hyp, n)
            ref_ng  = _ngrams(ref, n)
            clip_count  += sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
            total_count += max(0, len(hyp) - n + 1)
        if total_count == 0:
            precisions.append(0.0)
        else:
            precisions.append(clip_count / total_count)

    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / max_n
    return bp * math.exp(log_avg)


# ------------------------------------------------------------------ #
#  Precision / Recall / F1 a nivel carácter y palabra                 #
# ------------------------------------------------------------------ #

def _char_prf(hypothesis: str, reference: str) -> Tuple[float, float, float]:
    hyp_c = Counter(hypothesis)
    ref_c = Counter(reference)
    tp = sum((hyp_c & ref_c).values())
    precision = tp / max(len(hypothesis), 1)
    recall    = tp / max(len(reference),  1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _word_prf(hypothesis: str, reference: str) -> Tuple[float, float, float]:
    hyp_w = Counter(hypothesis.split())
    ref_w = Counter(reference.split())
    tp = sum((hyp_w & ref_w).values())
    precision = tp / max(sum(hyp_w.values()), 1)
    recall    = tp / max(sum(ref_w.values()), 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ------------------------------------------------------------------ #
#  Función maestra: calcula todo en un solo paso                      #
# ------------------------------------------------------------------ #

def compute_all_metrics(
    hypotheses: List[str],
    references: List[str],
) -> Dict:
    """
    Calcula todas las métricas sobre un corpus completo.

    Retorna un dict con claves organizadas por categoría.
    """
    assert len(hypotheses) == len(references), "Las listas deben tener el mismo tamaño"
    n = len(references)
    if n == 0:
        return {}

    # Acumuladores
    total_cer   = total_wer    = 0.0
    total_ned   = 0.0
    total_cp = total_cr = total_cf = 0.0
    total_wp = total_wr = total_wf = 0.0
    exact_match = 0
    total_chars_ref = total_chars_hyp = 0
    total_words_ref = total_words_hyp = 0

    # Desglose S/I/D a nivel carácter
    total_char_S = total_char_I = total_char_D = 0
    total_word_S = total_word_I = total_word_D = 0

    # Para histograma por longitud (buckets de 10 chars)
    bucket_cer: Dict[int, List[float]] = {}

    for hyp, ref in zip(hypotheses, references):
        # CER
        c_err = cer(hyp, ref)
        total_cer += c_err

        # WER
        total_wer += wer(hyp, ref)

        # NED
        total_ned += ned(hyp, ref)

        # Exact match
        if hyp == ref:
            exact_match += 1

        # Longitudes
        total_chars_ref += len(ref)
        total_chars_hyp += len(hyp)
        total_words_ref += len(ref.split())
        total_words_hyp += len(hyp.split())

        # S/I/D char
        _, s, ins, d = _levenshtein_ops(list(hyp), list(ref))
        total_char_S += s
        total_char_I += ins
        total_char_D += d

        # S/I/D word
        _, s, ins, d = _levenshtein_ops(hyp.split(), ref.split())
        total_word_S += s
        total_word_I += ins
        total_word_D += d

        # Char P/R/F1
        cp, cr, cf = _char_prf(hyp, ref)
        total_cp += cp; total_cr += cr; total_cf += cf

        # Word P/R/F1
        wp, wr, wf = _word_prf(hyp, ref)
        total_wp += wp; total_wr += wr; total_wf += wf

        # Bucket por longitud
        bucket = (len(ref) // 10) * 10
        bucket_cer.setdefault(bucket, []).append(c_err)

    # BLEU (corpus level)
    char_bleu = bleu_score(hypotheses, references, tokenize="char")
    word_bleu = bleu_score(hypotheses, references, tokenize="word")

    # Promedios
    avg = lambda x: x / n

    results = {
        # ---- Métricas principales ----
        "CER":          avg(total_cer),
        "WER":          avg(total_wer),
        "1-NED":        1.0 - avg(total_ned),
        "LineAcc":      exact_match / n,
        "CharAcc":      1.0 - avg(total_cer),    # = 1 - CER
        "WordAcc":      1.0 - avg(total_wer),

        # ---- BLEU ----
        "BLEU4_char":   char_bleu,
        "BLEU4_word":   word_bleu,

        # ---- Precision / Recall / F1 ----
        "Char_P":  avg(total_cp),
        "Char_R":  avg(total_cr),
        "Char_F1": avg(total_cf),
        "Word_P":  avg(total_wp),
        "Word_R":  avg(total_wr),
        "Word_F1": avg(total_wf),

        # ---- Desglose de errores (carácter) ----
        "Char_Substitutions": total_char_S,
        "Char_Insertions":    total_char_I,
        "Char_Deletions":     total_char_D,

        # ---- Desglose de errores (palabra) ----
        "Word_Substitutions": total_word_S,
        "Word_Insertions":    total_word_I,
        "Word_Deletions":     total_word_D,

        # ---- Estadísticas de corpus ----
        "N_samples":        n,
        "Exact_matches":    exact_match,
        "Avg_ref_len_char": total_chars_ref / n,
        "Avg_hyp_len_char": total_chars_hyp / n,
        "Avg_ref_len_word": total_words_ref / n,
        "Avg_hyp_len_word": total_words_hyp / n,

        # ---- CER por longitud de referencia ----
        "CER_by_length": {
            k: sum(v) / len(v)
            for k, v in sorted(bucket_cer.items())
        },
    }
    return results


# ------------------------------------------------------------------ #
#  Impresión formateada                                               #
# ------------------------------------------------------------------ #

def print_metrics(metrics: Dict, title: str = "Evaluación") -> None:
    """Imprime un reporte tabular completo en consola."""
    sep = "─" * 60
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

    def row(label, value, fmt=".4f", pct=False):
        v = f"{value:{fmt}}"
        suffix = " %" if pct else ""
        print(f"  {label:<30} {v}{suffix}")

    print(f"\n{'─'*60}")
    print("  MÉTRICAS PRINCIPALES")
    print(sep)
    row("CER  (↓ mejor)",        metrics["CER"])
    row("WER  (↓ mejor)",        metrics["WER"])
    row("1-NED  (↑ mejor)",      metrics["1-NED"])
    row("Line Accuracy  (↑)",    metrics["LineAcc"])
    row("Character Accuracy (↑)",metrics["CharAcc"])
    row("Word Accuracy (↑)",     metrics["WordAcc"])

    print(f"\n{sep}")
    print("  BLEU")
    print(sep)
    row("BLEU-4 char  (↑)",  metrics["BLEU4_char"])
    row("BLEU-4 word  (↑)",  metrics["BLEU4_word"])

    print(f"\n{sep}")
    print("  PRECISION / RECALL / F1")
    print(sep)
    row("Char Precision",  metrics["Char_P"])
    row("Char Recall",     metrics["Char_R"])
    row("Char F1",         metrics["Char_F1"])
    row("Word Precision",  metrics["Word_P"])
    row("Word Recall",     metrics["Word_R"])
    row("Word F1",         metrics["Word_F1"])

    print(f"\n{sep}")
    print("  DESGLOSE DE ERRORES — CARÁCTER")
    print(sep)
    row("Substituciones",  metrics["Char_Substitutions"], fmt="d")
    row("Inserciones",     metrics["Char_Insertions"],    fmt="d")
    row("Eliminaciones",   metrics["Char_Deletions"],     fmt="d")

    print(f"\n{sep}")
    print("  DESGLOSE DE ERRORES — PALABRA")
    print(sep)
    row("Substituciones",  metrics["Word_Substitutions"], fmt="d")
    row("Inserciones",     metrics["Word_Insertions"],    fmt="d")
    row("Eliminaciones",   metrics["Word_Deletions"],     fmt="d")

    print(f"\n{sep}")
    print("  CORPUS")
    print(sep)
    row("Muestras totales",    metrics["N_samples"],        fmt="d")
    row("Exact matches",       metrics["Exact_matches"],    fmt="d")
    row("Long. media ref (chars)", metrics["Avg_ref_len_char"], fmt=".1f")
    row("Long. media pred (chars)",metrics["Avg_hyp_len_char"], fmt=".1f")
    row("Long. media ref (words)", metrics["Avg_ref_len_word"], fmt=".1f")
    row("Long. media pred (words)",metrics["Avg_hyp_len_word"], fmt=".1f")

    print(f"\n{sep}")
    print("  CER POR LONGITUD DE REFERENCIA")
    print(sep)
    for length_bucket, bucket_cer in metrics["CER_by_length"].items():
        bar_len = int(bucket_cer * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  [{length_bucket:>3}-{length_bucket+9:<3}] {bar}  {bucket_cer:.4f}")

    print(f"{'═'*60}\n")