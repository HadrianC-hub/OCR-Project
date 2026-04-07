"""
metrics.py  —  Métricas OCR + Tests estadísticos para validación en tesis

Secciones:
  1. Métricas clásicas (CER, WER, NED, BLEU, P/R/F1)  — sin cambios
  2. Tests estadísticos:
       bootstrap_ci          — IC 95 % por remuestreo (10 000 iter.)
       shapiro_wilk          — normalidad de la distribución de CER
       binomial_exact        — LineAcc vs clasificador aleatorio H₀=0.5
       mcnemar_test          — greedy vs beam en corrección por línea
       wilcoxon_signed_rank  — CER greedy vs beam por muestra
       kruskal_wallis        — CER igual entre fuentes tipográficas?
       mannwhitney_posthoc   — pares de fuentes con corrección Bonferroni
       char_confusion_top_n  — top-N errores carácter con traceback
  3. compute_statistical_report — todo en uno, devuelve dict serializable
  4. print_statistical_report   — impresión formateada para consola/tesis
"""

from typing import List, Tuple, Dict
from collections import Counter, defaultdict
import math


# ══════════════════════════════════════════════════════════════════════
#  1. MÉTRICAS CLÁSICAS
# ══════════════════════════════════════════════════════════════════════

def _levenshtein_ops(a: List, b: List) -> Tuple[int, int, int, int]:
    m, n = len(a), len(b)
    INF = (10**9, 0, 0, 0)
    prev = [(j, 0, j, 0) for j in range(n + 1)]
    for i in range(1, m + 1):
        curr = [(i, 0, 0, i)] + [INF] * n
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                sub    = (prev[j-1][0] + 1, prev[j-1][1] + 1, prev[j-1][2], prev[j-1][3])
                ins_op = (curr[j-1][0] + 1, curr[j-1][1], curr[j-1][2] + 1, curr[j-1][3])
                del_op = (prev[j][0]   + 1, prev[j][1],   prev[j][2],   prev[j][3] + 1)
                curr[j] = min(sub, ins_op, del_op, key=lambda x: x[0])
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


def cer(hypothesis: str, reference: str) -> float:
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return _levenshtein(list(hypothesis), list(reference)) / len(reference)


def wer(hypothesis: str, reference: str) -> float:
    ref_w = reference.split()
    hyp_w = hypothesis.split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return _levenshtein(hyp_w, ref_w) / len(ref_w)


def ned(hypothesis: str, reference: str) -> float:
    if len(hypothesis) == 0 and len(reference) == 0:
        return 0.0
    denom = max(len(hypothesis), len(reference))
    return _levenshtein(list(hypothesis), list(reference)) / denom


def _ngrams(tokens: List, n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def bleu_score(hypotheses: List[str], references: List[str],
               max_n: int = 4, tokenize: str = "char") -> float:
    if tokenize == "char":
        hyp_tok = [list(h) for h in hypotheses]
        ref_tok = [list(r) for r in references]
    else:
        hyp_tok = [h.split() for h in hypotheses]
        ref_tok = [r.split() for r in references]

    hyp_len = sum(len(h) for h in hyp_tok)
    ref_len  = sum(len(r) for r in ref_tok)
    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / hyp_len)

    precisions = []
    for n in range(1, max_n + 1):
        clip_count, total_count = 0, 0
        for hyp, ref in zip(hyp_tok, ref_tok):
            hyp_ng = _ngrams(hyp, n)
            ref_ng = _ngrams(ref, n)
            clip_count  += sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
            total_count += max(0, len(hyp) - n + 1)
        precisions.append(clip_count / total_count if total_count else 0.0)

    if any(p == 0 for p in precisions):
        return 0.0
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)


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


def compute_all_metrics(hypotheses: List[str], references: List[str]) -> Dict:
    assert len(hypotheses) == len(references)
    n = len(references)
    if n == 0:
        return {}

    total_cer = total_wer = total_ned = 0.0
    total_cp = total_cr = total_cf = 0.0
    total_wp = total_wr = total_wf = 0.0
    exact_match = 0
    total_chars_ref = total_chars_hyp = 0
    total_words_ref = total_words_hyp = 0
    total_char_S = total_char_I = total_char_D = 0
    total_word_S = total_word_I = total_word_D = 0
    bucket_cer: Dict[int, List[float]] = {}

    for hyp, ref in zip(hypotheses, references):
        c_err = cer(hyp, ref)
        total_cer += c_err
        total_wer += wer(hyp, ref)
        total_ned += ned(hyp, ref)
        if hyp == ref:
            exact_match += 1

        total_chars_ref += len(ref)
        total_chars_hyp += len(hyp)
        total_words_ref += len(ref.split())
        total_words_hyp += len(hyp.split())

        _, s, ins, d = _levenshtein_ops(list(hyp), list(ref))
        total_char_S += s; total_char_I += ins; total_char_D += d

        _, s, ins, d = _levenshtein_ops(hyp.split(), ref.split())
        total_word_S += s; total_word_I += ins; total_word_D += d

        cp, cr, cf = _char_prf(hyp, ref)
        total_cp += cp; total_cr += cr; total_cf += cf

        wp, wr, wf = _word_prf(hyp, ref)
        total_wp += wp; total_wr += wr; total_wf += wf

        bucket = (len(ref) // 10) * 10
        bucket_cer.setdefault(bucket, []).append(c_err)

    char_bleu = bleu_score(hypotheses, references, tokenize="char")
    word_bleu = bleu_score(hypotheses, references, tokenize="word")
    avg = lambda x: x / n

    return {
        "CER":     avg(total_cer),
        "WER":     avg(total_wer),
        "1-NED":   1.0 - avg(total_ned),
        "LineAcc": exact_match / n,
        "CharAcc": 1.0 - avg(total_cer),
        "WordAcc": 1.0 - avg(total_wer),
        "BLEU4_char": char_bleu,
        "BLEU4_word": word_bleu,
        "Char_P":  avg(total_cp), "Char_R": avg(total_cr), "Char_F1": avg(total_cf),
        "Word_P":  avg(total_wp), "Word_R": avg(total_wr), "Word_F1": avg(total_wf),
        "Char_Substitutions": total_char_S,
        "Char_Insertions":    total_char_I,
        "Char_Deletions":     total_char_D,
        "Word_Substitutions": total_word_S,
        "Word_Insertions":    total_word_I,
        "Word_Deletions":     total_word_D,
        "N_samples":        n,
        "Exact_matches":    exact_match,
        "Avg_ref_len_char": total_chars_ref / n,
        "Avg_hyp_len_char": total_chars_hyp / n,
        "Avg_ref_len_word": total_words_ref / n,
        "Avg_hyp_len_word": total_words_hyp / n,
        "CER_by_length": {k: sum(v)/len(v) for k, v in sorted(bucket_cer.items())},
    }


def print_metrics(metrics: Dict, title: str = "Evaluación") -> None:
    sep = "─" * 60
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

    def row(label, value, fmt=".4f"):
        print(f"  {label:<30} {value:{fmt}}")

    print(f"\n{sep}\n  MÉTRICAS PRINCIPALES\n{sep}")
    row("CER  (↓ mejor)",         metrics["CER"])
    row("WER  (↓ mejor)",         metrics["WER"])
    row("1-NED  (↑ mejor)",       metrics["1-NED"])
    row("Line Accuracy  (↑)",     metrics["LineAcc"])
    row("Character Accuracy (↑)", metrics["CharAcc"])
    row("Word Accuracy (↑)",      metrics["WordAcc"])

    print(f"\n{sep}\n  BLEU\n{sep}")
    row("BLEU-4 char  (↑)", metrics["BLEU4_char"])
    row("BLEU-4 word  (↑)", metrics["BLEU4_word"])

    print(f"\n{sep}\n  PRECISION / RECALL / F1\n{sep}")
    row("Char Precision", metrics["Char_P"])
    row("Char Recall",    metrics["Char_R"])
    row("Char F1",        metrics["Char_F1"])
    row("Word Precision", metrics["Word_P"])
    row("Word Recall",    metrics["Word_R"])
    row("Word F1",        metrics["Word_F1"])

    print(f"\n{sep}\n  DESGLOSE DE ERRORES — CARÁCTER\n{sep}")
    row("Substituciones", metrics["Char_Substitutions"], fmt="d")
    row("Inserciones",    metrics["Char_Insertions"],    fmt="d")
    row("Eliminaciones",  metrics["Char_Deletions"],     fmt="d")

    print(f"\n{sep}\n  DESGLOSE DE ERRORES — PALABRA\n{sep}")
    row("Substituciones", metrics["Word_Substitutions"], fmt="d")
    row("Inserciones",    metrics["Word_Insertions"],    fmt="d")
    row("Eliminaciones",  metrics["Word_Deletions"],     fmt="d")

    print(f"\n{sep}\n  CORPUS\n{sep}")
    row("Muestras totales",         metrics["N_samples"],        fmt="d")
    row("Exact matches",            metrics["Exact_matches"],    fmt="d")
    row("Long. media ref (chars)",  metrics["Avg_ref_len_char"], fmt=".1f")
    row("Long. media pred (chars)", metrics["Avg_hyp_len_char"], fmt=".1f")
    row("Long. media ref (words)",  metrics["Avg_ref_len_word"], fmt=".1f")
    row("Long. media pred (words)", metrics["Avg_hyp_len_word"], fmt=".1f")

    print(f"\n{sep}\n  CER POR LONGITUD DE REFERENCIA\n{sep}")
    for length_bucket, bucket_cer_val in metrics["CER_by_length"].items():
        bar_len = int(bucket_cer_val * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  [{length_bucket:>3}-{length_bucket+9:<3}] {bar}  {bucket_cer_val:.4f}")

    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════
#  2. TESTS ESTADÍSTICOS
# ══════════════════════════════════════════════════════════════════════

def _require_scipy():
    try:
        import scipy.stats as _st
        import numpy as _np
        return _st, _np
    except ImportError:
        raise ImportError(
            "scipy y numpy son necesarios para los tests estadísticos.\n"
            "Instálalos con:  pip install scipy numpy"
        )


def bootstrap_ci(values, n_bootstrap: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> dict:
    """IC percentil al (1-alpha)*100% por remuestreo bootstrap."""
    _, np = _require_scipy()
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed=seed)
    n   = len(arr)
    boot = np.array([arr[rng.integers(0, n, n)].mean() for _ in range(n_bootstrap)])
    return {
        "mean":        float(arr.mean()),
        "std":         float(arr.std()),
        "ci_low":      float(np.percentile(boot, 100 * alpha / 2)),
        "ci_high":     float(np.percentile(boot, 100 * (1 - alpha / 2))),
        "n":           n,
        "n_bootstrap": n_bootstrap,
    }


def shapiro_wilk(values, max_n: int = 5_000, seed: int = 1) -> dict:
    """
    Test de Shapiro-Wilk sobre la distribución de valores.
    Si hay más de max_n muestras, usa submuestra aleatoria.
    H₀: los datos provienen de una distribución normal.
    """
    st, np = _require_scipy()
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed=seed)
    if len(arr) > max_n:
        arr = arr[rng.choice(len(arr), max_n, replace=False)]
    stat, p = st.shapiro(arr)
    return {
        "statistic":       float(stat),
        "p_value":         float(p),
        "n_used":          len(arr),
        "is_normal_alpha05": bool(p >= 0.05),
    }


def binomial_exact(n_correct: int, n_total: int, p0: float = 0.5) -> dict:
    """
    Test binomial exacto (alternativa: greater).
    H₀: P(línea correcta) = p0.
    """
    st, _ = _require_scipy()
    result = st.binomtest(n_correct, n_total, p=p0, alternative="greater")
    ci = result.proportion_ci(confidence_level=0.95)
    return {
        "n_correct":          n_correct,
        "n_total":            n_total,
        "p_observed":         n_correct / n_total,
        "p0_null":            p0,
        "p_value":            float(result.pvalue),
        "ci95_low":           float(ci.low),
        "ci95_high":          float(ci.high),
        "reject_H0_alpha001": bool(result.pvalue < 0.001),
    }


def mcnemar_test(correct_a, correct_b) -> dict:
    """
    Test de McNemar con corrección de continuidad de Edwards (b+c < 25).
    H₀: ambos clasificadores cometen el mismo número de errores.
    correct_a[i], correct_b[i] ∈ {0,1}  (1 = línea perfecta).
    """
    st, np = _require_scipy()
    ca = np.asarray(correct_a, dtype=int)
    cb = np.asarray(correct_b, dtype=int)
    b  = int(((1 - ca) * cb).sum())
    c  = int((ca * (1 - cb)).sum())
    n  = b + c
    if n == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": 0, "c": 0,
                "significant_alpha05": False, "note": "sin_discordancias"}
    chi2 = (abs(b - c) - (1 if n < 25 else 0)) ** 2 / n
    p    = float(1 - st.chi2.cdf(chi2, df=1))
    return {
        "chi2":                float(chi2),
        "p_value":             p,
        "b_A_wrong_B_right":   b,
        "c_A_right_B_wrong":   c,
        "significant_alpha05": p < 0.05,
    }


def wilcoxon_signed_rank(values_a, values_b) -> dict:
    """
    Test de Wilcoxon signed-rank (two-sided) sobre pares (a_i, b_i).
    H₀: mediana(a - b) = 0.
    """
    st, np = _require_scipy()
    a    = np.asarray(values_a, dtype=float)
    b    = np.asarray(values_b, dtype=float)
    diff = a - b
    if np.all(diff == 0):
        return {"p_value": 1.0, "note": "todas_las_diferencias_son_cero",
                "significant_alpha05": False}
    stat, p = st.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return {
        "statistic":           float(stat),
        "p_value":             float(p),
        "mean_diff":           float(diff.mean()),
        "median_diff":         float(np.median(diff)),
        "significant_alpha05": bool(p < 0.05),
    }


def kruskal_wallis(groups: dict) -> dict:
    """
    Test de Kruskal-Wallis sobre grupos (dict {nombre: [valores]}).
    H₀: igual distribución en todos los grupos.
    """
    st, np = _require_scipy()
    valid = {k: np.asarray(v, dtype=float) for k, v in groups.items() if len(v) > 1}
    if len(valid) < 2:
        return {"note": "menos_de_2_grupos_validos", "p_value": 1.0}
    stat, p = st.kruskal(*valid.values())
    return {
        "statistic":           float(stat),
        "p_value":             float(p),
        "n_groups":            len(valid),
        "groups":              list(valid.keys()),
        "significant_alpha05": bool(p < 0.05),
    }


def mannwhitney_posthoc(groups: dict, alpha: float = 0.05) -> list:
    """Mann-Whitney U por pares con corrección de Bonferroni."""
    st, np = _require_scipy()
    valid  = {k: np.asarray(v, dtype=float) for k, v in groups.items() if len(v) > 1}
    keys   = sorted(valid.keys())
    pairs  = [(keys[i], keys[j])
              for i in range(len(keys)) for j in range(i+1, len(keys))]
    alpha_b = alpha / max(len(pairs), 1)
    results = []
    for fa, fb in pairs:
        stat, p = st.mannwhitneyu(valid[fa], valid[fb], alternative="two-sided")
        results.append({
            "group_a":          fa,
            "group_b":          fb,
            "U":                float(stat),
            "p_value":          float(p),
            "alpha_bonferroni": alpha_b,
            "significant":      bool(p < alpha_b),
        })
    return results


def char_confusion_top_n(hyps: List[str], refs: List[str], top_n: int = 20) -> list:
    """
    Top-N errores de carácter con traceback de Levenshtein.
    Solo procesa pares donde hyp ≠ ref (eficiente con CER muy bajo).
    """
    subs = Counter()
    dels = Counter()
    ins  = Counter()

    for hyp, ref in zip(hyps, refs):
        if hyp == ref:
            continue
        m, n = len(ref), len(hyp)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1): dp[i][0] = i
        for j in range(n + 1): dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i-1] == hyp[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
                i -= 1; j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                subs[(ref[i-1], hyp[j-1])] += 1; i -= 1; j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                dels[ref[i-1]] += 1; i -= 1
            else:
                ins[hyp[j-1]] += 1; j -= 1

    result = []
    for (rc, hc), cnt in subs.most_common(top_n):
        result.append({"type": "SUB", "ref": rc, "hyp": hc, "count": cnt})
    for rc, cnt in dels.most_common(10):
        result.append({"type": "DEL", "ref": rc, "hyp": "",  "count": cnt})
    for hc, cnt in ins.most_common(10):
        result.append({"type": "INS", "ref": "",  "hyp": hc, "count": cnt})
    return result


# ══════════════════════════════════════════════════════════════════════
#  3. INFORME ESTADÍSTICO COMPLETO
# ══════════════════════════════════════════════════════════════════════

def compute_statistical_report(
    hyps_greedy:  List[str],
    hyps_beam:    List[str],
    refs:         List[str],
    font_labels:  List[str] = None,
    n_bootstrap:  int = 10_000,
) -> dict:
    """
    Ejecuta todos los tests y devuelve un dict 100 % serializable a JSON.

    Estructura del dict devuelto:
      n_samples, per_sample, bootstrap, shapiro_wilk, binomial,
      mcnemar, wilcoxon, font_analysis, char_errors
    """
    _, np = _require_scipy()

    N     = len(refs)
    cer_g = np.array([cer(h, r) for h, r in zip(hyps_greedy, refs)])
    cer_b = np.array([cer(h, r) for h, r in zip(hyps_beam,   refs)])
    wer_g = np.array([wer(h, r) for h, r in zip(hyps_greedy, refs)])
    wer_b = np.array([wer(h, r) for h, r in zip(hyps_beam,   refs)])
    corr_g = (cer_g == 0).astype(int)
    corr_b = (cer_b == 0).astype(int)

    report: dict = {"n_samples": N}

    report["per_sample"] = {
        "cer_greedy":    cer_g.tolist(),
        "cer_beam":      cer_b.tolist(),
        "wer_greedy":    wer_g.tolist(),
        "wer_beam":      wer_b.tolist(),
        "correct_greedy": corr_g.tolist(),
        "correct_beam":   corr_b.tolist(),
    }

    report["bootstrap"] = {
        "CER_greedy":     bootstrap_ci(cer_g,                  n_bootstrap),
        "CER_beam":       bootstrap_ci(cer_b,                  n_bootstrap),
        "WER_greedy":     bootstrap_ci(wer_g,                  n_bootstrap),
        "WER_beam":       bootstrap_ci(wer_b,                  n_bootstrap),
        "LineAcc_greedy": bootstrap_ci(corr_g.astype(float),   n_bootstrap),
        "LineAcc_beam":   bootstrap_ci(corr_b.astype(float),   n_bootstrap),
    }

    report["shapiro_wilk"] = {
        "greedy": shapiro_wilk(cer_g),
        "beam":   shapiro_wilk(cer_b),
    }

    report["binomial"] = binomial_exact(int(corr_g.sum()), N, p0=0.5)
    report["mcnemar"]  = mcnemar_test(corr_g, corr_b)
    report["wilcoxon"] = wilcoxon_signed_rank(cer_g, cer_b)

    if font_labels and any(f for f in font_labels):
        groups: dict = defaultdict(list)
        for fl, c in zip(font_labels, cer_g.tolist()):
            if fl:
                groups[fl].append(c)
        groups = dict(groups)
        per_font = {f: bootstrap_ci(np.array(v), min(n_bootstrap, 2000))
                    for f, v in groups.items()}
        kw      = kruskal_wallis(groups)
        posthoc = mannwhitney_posthoc(groups) if kw.get("significant_alpha05") else []
        report["font_analysis"] = {
            "kruskal_wallis":      kw,
            "per_font":            per_font,
            "posthoc_mannwhitney": posthoc,
        }
    else:
        report["font_analysis"] = {"note": "font_labels_no_disponibles"}

    report["char_errors"] = char_confusion_top_n(hyps_greedy, refs, top_n=20)
    return report


# ══════════════════════════════════════════════════════════════════════
#  4. IMPRESIÓN FORMATEADA
# ══════════════════════════════════════════════════════════════════════

def print_statistical_report(report: dict) -> None:
    """Imprime el informe estadístico completo en consola."""
    _, np = _require_scipy()

    def _sep(c="─"): print(c * 66)
    def _hdr(t, c="═"): print(f"\n{c*66}\n  {t}\n{c*66}")

    N = report["n_samples"]

    # ── Bootstrap ──────────────────────────────────────────────────
    nb = report["bootstrap"]["CER_greedy"]["n_bootstrap"]
    _hdr(f"BOOTSTRAP IC 95 %  (n_bootstrap = {nb:,})")
    print(f"\n  {'Métrica':<22} {'Media':>9} {'Std':>9} {'IC95 bajo':>11} {'IC95 alto':>11}")
    _sep()
    for key, label in [
        ("CER_greedy",     "CER greedy"),
        ("CER_beam",       "CER beam"),
        ("WER_greedy",     "WER greedy"),
        ("LineAcc_greedy", "LineAcc greedy"),
        ("LineAcc_beam",   "LineAcc beam"),
    ]:
        b = report["bootstrap"][key]
        print(f"  {label:<22} {b['mean']:>9.6f} {b['std']:>9.6f} "
              f"{b['ci_low']:>11.6f} {b['ci_high']:>11.6f}")

    # ── Shapiro-Wilk ───────────────────────────────────────────────
    _hdr("TEST DE SHAPIRO-WILK — NORMALIDAD DEL CER")
    print(f"\n  H₀: la distribución de CER por muestra es normal")
    print(f"\n  {'Decoder':<12} {'W':>10} {'p-valor':>12}  Conclusión")
    _sep()
    for dec, key in [("Greedy", "greedy"), ("Beam", "beam")]:
        s    = report["shapiro_wilk"][key]
        conc = "No rechazar H₀ (normal)" if s["is_normal_alpha05"] else "Rechazar H₀ (no normal)"
        print(f"  {dec:<12} {s['statistic']:>10.6f} {s['p_value']:>12.4e}  {conc}")
    sw_g = report["shapiro_wilk"]["greedy"]
    rec  = "no-paramétricos (Wilcoxon, Kruskal-Wallis)" if not sw_g["is_normal_alpha05"] \
           else "paramétricos o no-paramétricos"
    print(f"\n  → Tests recomendados: {rec}")

    # ── Binomial ───────────────────────────────────────────────────
    _hdr("TEST BINOMIAL EXACTO — LINE ACCURACY (GREEDY)")
    b = report["binomial"]
    print(f"\n  H₀: P(línea correcta) = {b['p0_null']}  (clasificador aleatorio)")
    print(f"  Alternativa: P(línea correcta) > {b['p0_null']}")
    print(f"\n  Líneas correctas : {b['n_correct']:,} / {b['n_total']:,}"
          f"  ({b['p_observed']*100:.4f} %)")
    print(f"  p-valor          : {b['p_value']:.4e}")
    print(f"  IC 95 % Clopper-Pearson: [{b['ci95_low']:.6f},  {b['ci95_high']:.6f}]")
    conc = "Rechazar H₀  (p < 0.001)" if b["reject_H0_alpha001"] else "No rechazar H₀"
    print(f"  Conclusión       : {conc}")

    # ── McNemar ────────────────────────────────────────────────────
    _hdr("TEST DE McNEMAR — GREEDY vs BEAM (LINE ACCURACY)")
    m = report["mcnemar"]
    if "note" in m and m.get("note") == "sin_discordancias":
        print("\n  Sin discordancias entre decoders — test no aplicable.")
    else:
        print(f"\n  H₀: greedy y beam cometen el mismo número de errores")
        print(f"\n  Greedy ✗, Beam ✓  (b) : {m['b_A_wrong_B_right']:,}")
        print(f"  Greedy ✓, Beam ✗  (c) : {m['c_A_right_B_wrong']:,}")
        print(f"  χ² (correc. continuidad): {m['chi2']:.4f}")
        print(f"  p-valor                : {m['p_value']:.4e}")
        conc = "Diferencia significativa" if m["significant_alpha05"] \
               else "Sin diferencia significativa"
        print(f"  Conclusión             : {conc}  (α = 0.05)")

    # ── Wilcoxon ───────────────────────────────────────────────────
    _hdr("TEST DE WILCOXON SIGNED-RANK — CER GREEDY vs BEAM")
    w = report["wilcoxon"]
    if "note" in w:
        print(f"\n  {w['note']}")
    else:
        print(f"\n  H₀: mediana(CER_greedy − CER_beam) = 0")
        print(f"  Estadístico W  : {w['statistic']:.4f}")
        print(f"  p-valor        : {w['p_value']:.4e}")
        print(f"  Diferencia media   (g − b): {w['mean_diff']:+.6f}")
        print(f"  Diferencia mediana (g − b): {w['median_diff']:+.6f}")
        mejor = "Greedy" if w["mean_diff"] < 0 else ("Beam" if w["mean_diff"] > 0 else "Igual")
        print(f"  Mejor decoder  : {mejor}")
        conc = "Diferencia significativa" if w["significant_alpha05"] \
               else "Sin diferencia significativa"
        print(f"  Conclusión     : {conc}  (α = 0.05)")

    # ── Análisis por fuente ────────────────────────────────────────
    fa = report.get("font_analysis", {})
    if "note" not in fa:
        _hdr("ANÁLISIS POR FUENTE TIPOGRÁFICA")
        pf = fa.get("per_font", {})
        if pf:
            print(f"\n  {'Fuente':<45} {'N':>5} {'Media':>8} {'Std':>8} "
                  f"{'IC95-':>10} {'IC95+':>10}")
            _sep()
            for font, b in sorted(pf.items()):
                lbl = (font[:43] + "…") if len(font) > 44 else font
                print(f"  {lbl:<45} {b['n']:>5} {b['mean']:>8.6f} {b['std']:>8.6f} "
                      f"{b['ci_low']:>10.6f} {b['ci_high']:>10.6f}")

        kw = fa.get("kruskal_wallis", {})
        if "statistic" in kw:
            print(f"\n  Kruskal-Wallis H = {kw['statistic']:.4f}  "
                  f"p = {kw['p_value']:.4e}")
            conc = "Diferencias significativas entre fuentes" \
                   if kw["significant_alpha05"] \
                   else "Sin diferencias significativas entre fuentes"
            print(f"  Conclusión: {conc}  (α = 0.05)")

        posthoc = fa.get("posthoc_mannwhitney", [])
        if posthoc:
            ab = posthoc[0]["alpha_bonferroni"]
            print(f"\n  Post-hoc Mann-Whitney U (Bonferroni α = {ab:.4f}):")
            print(f"  {'Par':<48} {'U':>10} {'p-valor':>12} {'Sig.':>6}")
            _sep()
            for r in posthoc:
                pair = f"{r['group_a'][:21]} vs {r['group_b'][:21]}"
                sig  = "✓" if r["significant"] else "✗"
                print(f"  {pair:<48} {r['U']:>10.1f} {r['p_value']:>12.4e} {sig:>6}")

    # ── Errores de carácter ────────────────────────────────────────
    _hdr("TOP ERRORES A NIVEL DE CARÁCTER (GREEDY)")
    errors = report.get("char_errors", [])
    if errors:
        print(f"\n  {'Tipo':<5} {'Ref':^6} {'Pred':^6} {'Veces':>8}")
        _sep()
        for e in errors[:25]:
            rd = repr(e["ref"]) if e["ref"] == " " else (f"'{e['ref']}'" if e["ref"] else "∅")
            hd = repr(e["hyp"]) if e["hyp"] == " " else (f"'{e['hyp']}'" if e["hyp"] else "∅")
            print(f"  {e['type']:<5} {rd:^6} {hd:^6} {e['count']:>8,}")
    else:
        print("\n  Sin errores de carácter detectados (CER = 0 en todas las muestras).")

    print(f"\n{'═'*66}\n")