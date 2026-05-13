"""
spell_correct.py — Corrección ortográfica post-OCR para español.

Dos motores combinables:

  • SymSpell + wordfreq          (rápido, sub-ms/palabra, sin contexto)
  • SymSpell + BETO via ONNX     (lento, con contexto semántico)

El segundo es opt-in por documento (campo Document.use_bert_correction).
SymSpell siempre genera los candidatos; BETO solo decide entre ellos
cuando hay más de uno.

Filosofía:
  - SymSpell propone, BETO escoge. NUNCA inventa palabras nuevas.
  - Solo se corrige palabras que SymSpell decidiría corregir.
  - Degradación elegante: si BETO no se puede cargar, cae a SymSpell solo;
    si SymSpell no se puede cargar, devuelve el texto sin tocar.

INSTALACIÓN:
    pip install symspellpy wordfreq            # base obligatoria
    pip install transformers onnxruntime       # opcional, para BERT

Plus el modelo ONNX colocado en models/beto/ del proyecto. Ver INSTALL.md
para el procedimiento de descarga y conversión (un único paso, una vez).
"""

import logging
import math
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Configuración ─────────────────────────────────────────────────────────────

SPELLCHECK_ENABLED = True
VOCAB_SIZE = 80_000
MAX_EDIT_DISTANCE = 3
MIN_WORD_LEN = 3

# Reranker BERT (vía ONNX)
BERT_TOP_K_CANDIDATES = 5     # cuántos candidatos pide a SymSpell para reranking
BERT_CONTEXT_WORDS = 20       # ventana de palabras a cada lado para contexto
BERT_ALPHA = 0.7              # peso del LM vs frecuencia (0.7·logprob + 0.3·logfreq)
AGGRESSIVE_BERT_MODE = True


def _resolve_model_dir() -> Path:
    """
    Devuelve la ruta donde buscar el modelo BETO ONNX. Prioridad:
      1. variable de entorno ARCHIVOOCR_BERT_PATH
      2. {BASE_DIR}/models/beto/   (Django BASE_DIR si está disponible)
      3. ./models/beto/            (relativo al CWD)
    """
    env = os.environ.get("ARCHIVOOCR_BERT_PATH")
    if env:
        return Path(env)
    try:
        from django.conf import settings  # noqa: PLC0415
        return Path(settings.BASE_DIR) / "models" / "beto"
    except Exception:
        return Path("models/beto")


_WORD_RE = re.compile(r"([^\W\d_]+)", re.UNICODE)


# ── Singletons (carga diferida thread-safe) ──────────────────────────────────
#
# Cargar SymSpell con el vocabulario español de wordfreq cuesta varios
# segundos y ~80 MB de RAM. Cargar BETO ONNX cuesta otro tanto. Para que
# dos threads del OCR no inicien la misma carga en paralelo (doble RAM,
# doble tiempo), envolvemos cada carga en un Lock. Patrón estándar de
# double-checked locking: fast path sin lock si ya está cargado/intentado,
# slow path con lock + re-check para inicializar.

import threading

_symspell = None
_symspell_load_attempted = False
_symspell_lock = threading.Lock()

_bert_session   = None       # onnxruntime InferenceSession
_bert_tokenizer = None
_bert_mask_id   = None
_bert_load_attempted = False
_bert_lock      = threading.Lock()


def _build_symspell():
    """Construye SymSpell con el diccionario español de wordfreq."""
    try:
        from symspellpy import SymSpell  # noqa: PLC0415
    except ImportError:
        logger.warning("symspellpy no instalado. Corrección desactivada. "
                       "pip install symspellpy")
        return None
    try:
        from wordfreq import top_n_list, word_frequency  # noqa: PLC0415
    except ImportError:
        logger.warning("wordfreq no instalado. pip install wordfreq")
        return None

    sym = SymSpell(max_dictionary_edit_distance=MAX_EDIT_DISTANCE,
                   prefix_length=7)
    n = 0
    for w in top_n_list("es", VOCAB_SIZE):
        if not w.isalpha():
            continue
        sym.create_dictionary_entry(w, max(1, int(word_frequency(w, "es") * 1e9)))
        n += 1
    logger.info("SymSpell cargado: %d palabras, distance=%d.",
                n, MAX_EDIT_DISTANCE)
    return sym


def _get_symspell():
    global _symspell, _symspell_load_attempted
    if _symspell_load_attempted:
        return _symspell
    with _symspell_lock:
        if _symspell_load_attempted:
            return _symspell
        _symspell_load_attempted = True
        try:
            _symspell = _build_symspell()
        except Exception as exc:
            logger.warning("Error inicializando SymSpell: %s", exc)
        return _symspell


def _build_bert():
    """
    Carga el modelo BETO ONNX y su tokenizador desde models/beto/.
    Devuelve (session, tokenizer, mask_token_id) o (None, None, None)
    si algo falta (modelo, librerías, etc.).
    """
    try:
        import onnxruntime as ort  # noqa: PLC0415
    except ImportError:
        logger.warning("onnxruntime no instalado. Reranker BERT desactivado. "
                       "pip install onnxruntime")
        return None, None, None
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except ImportError:
        logger.warning("transformers no instalado (necesario para tokenizar). "
                       "pip install transformers")
        return None, None, None

    model_dir = _resolve_model_dir()
    onnx_path = model_dir / "model.onnx"

    if not onnx_path.is_file():
        logger.warning(
            "No se encontró el modelo BETO ONNX en %s. "
            "Sigue las instrucciones de INSTALL.md para descargarlo y "
            "convertirlo. Reranker BERT desactivado, cae a SymSpell solo.",
            onnx_path,
        )
        return None, None, None

    try:
        logger.info("Cargando BETO ONNX desde %s ...", model_dir)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        mask_id = tokenizer.mask_token_id
        logger.info("BETO ONNX cargado. Reranker activo.")
        return session, tokenizer, mask_id
    except Exception as exc:
        logger.warning("No se pudo cargar BETO ONNX: %s. Cae a SymSpell.", exc)
        return None, None, None


def _get_bert():
    global _bert_session, _bert_tokenizer, _bert_mask_id, _bert_load_attempted
    if _bert_load_attempted:
        return _bert_session, _bert_tokenizer, _bert_mask_id
    with _bert_lock:
        if _bert_load_attempted:
            return _bert_session, _bert_tokenizer, _bert_mask_id
        _bert_load_attempted = True
        _bert_session, _bert_tokenizer, _bert_mask_id = _build_bert()
        return _bert_session, _bert_tokenizer, _bert_mask_id


# ── Heurística "vale la pena corregir" ────────────────────────────────────────

def _should_correct(word: str, sym, aggressive: bool = False) -> bool:
    if len(word) < MIN_WORD_LEN:
        return False
    lw = word.lower()
    if lw in sym.words:
        return False
    if not aggressive and word[0].isupper() and lw not in sym.words:
        return False
    return True


# ── Corrección con SymSpell solo (modo legacy) ───────────────────────────────

def _correct_word_symspell_only(word: str, sym) -> str:
    from symspellpy import Verbosity  # noqa: PLC0415
    if not _should_correct(word, sym):
        return word
    sugs = sym.lookup(word, Verbosity.CLOSEST,
                      max_edit_distance=MAX_EDIT_DISTANCE,
                      transfer_casing=True)
    if not sugs:
        return word
    best = sugs[0]
    if best.term.lower() == word.lower():
        return word
    return best.term


# ── Corrección con SymSpell + BERT (ONNX) reranker ───────────────────────────

def _bert_score_candidates(
    word_tokens: List[str],
    target_idx: int,
    candidates: List[str],
    session,
    tokenizer,
    mask_id: int,
) -> List[float]:
    """
    Para cada candidato, devuelve la log-prob de su PRIMER subword en
    la posición enmascarada. Una sola pasada de inferencia por palabra
    ambigua (no por candidato): se enmascara la posición y se lee logits
    sobre todo el vocabulario; luego se mira el id del primer subword
    de cada candidato.
    """
    import numpy as np  # noqa: PLC0415

    masked = list(word_tokens)
    masked[target_idx] = tokenizer.mask_token
    text = " ".join(masked)

    enc = tokenizer(text, truncation=True, max_length=512,
                    return_tensors="np")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    mask_positions = (input_ids == mask_id).nonzero()
    if len(mask_positions[1]) == 0:
        return [0.0] * len(candidates)
    mask_pos = int(mask_positions[1][0])

    inputs = {
        "input_ids": input_ids.astype(np.int64),
        "attention_mask": attention_mask.astype(np.int64),
    }
    sess_input_names = {i.name for i in session.get_inputs()}
    if "token_type_ids" in sess_input_names:
        inputs["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)

    outputs = session.run(None, inputs)
    logits = outputs[0]                         # [batch, seq, vocab]
    mask_logits = logits[0, mask_pos]            # [vocab]

    # log_softmax estable numéricamente, sin scipy
    m = mask_logits.max()
    log_probs = mask_logits - (m + np.log(np.exp(mask_logits - m).sum()))

    scores = []
    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if not ids:
            scores.append(float("-inf"))
            continue
        scores.append(float(log_probs[ids[0]]))
    return scores


def _build_context_window(parts: List[str], word_part_idx: int,
                          n_words: int = BERT_CONTEXT_WORDS
                          ) -> Tuple[List[str], int]:
    """
    Construye una ventana de palabras alrededor de parts[word_part_idx].
    parts es la salida de _WORD_RE.split: separadores en pares, palabras
    en impares. Devuelve (lista_palabras, indice_local_objetivo).
    """
    word_indices = list(range(1, len(parts), 2))
    target_pos = word_indices.index(word_part_idx)
    start = max(0, target_pos - n_words)
    end = min(len(word_indices), target_pos + n_words + 1)
    words = [parts[word_indices[k]] for k in range(start, end)]
    return words, target_pos - start


def _correct_word_with_bert(parts, i, sym, session, tokenizer, mask_id):
    from symspellpy import Verbosity  # noqa: PLC0415
    word = parts[i]
    if not _should_correct(word, sym, aggressive=AGGRESSIVE_BERT_MODE):
        return word

    sugs = sym.lookup(word, Verbosity.CLOSEST,
                      max_edit_distance=MAX_EDIT_DISTANCE,
                      transfer_casing=True)
    if not sugs:
        return word

    candidates = sugs[:BERT_TOP_K_CANDIDATES]
    if len(candidates) == 1:
        best = candidates[0]
        if best.term.lower() == word.lower():
            return word
        return best.term

    word_window, target_local = _build_context_window(parts, i)
    cand_terms = [c.term for c in candidates]
    bert_scores = _bert_score_candidates(
        word_window, target_local, cand_terms, session, tokenizer, mask_id,
    )

    best_idx, best_score = 0, float("-inf")
    for k, (cand, b_score) in enumerate(zip(candidates, bert_scores)):
        f_score = math.log(max(1, cand.count))
        combined = BERT_ALPHA * b_score + (1 - BERT_ALPHA) * f_score
        if combined > best_score:
            best_score = combined
            best_idx = k

    chosen = candidates[best_idx].term
    if chosen.lower() == word.lower():
        return word
    return chosen


# ── API pública ───────────────────────────────────────────────────────────────

def correct_text(text: str, use_bert: bool = False) -> str:
    """
    Corrige un texto preservando puntuación, espacios y saltos de línea.

    Parameters
    ----------
    text : str
        Texto OCR.
    use_bert : bool
        Si True, usa BETO ONNX como reranker. Si BETO no se puede cargar
        (modelo no instalado, librerías ausentes, etc.) cae a SymSpell.
    """
    if not SPELLCHECK_ENABLED or not text:
        return text
    sym = _get_symspell()
    if sym is None:
        return text

    session, tokenizer, mask_id = (None, None, None)
    if use_bert:
        session, tokenizer, mask_id = _get_bert()

    parts = _WORD_RE.split(text)
    word_indices = list(range(1, len(parts), 2))

    if session is None or tokenizer is None:
        for i in word_indices:
            parts[i] = _correct_word_symspell_only(parts[i], sym)
    else:
        for i in word_indices:
            parts[i] = _correct_word_with_bert(parts, i, sym,
                                               session, tokenizer, mask_id)
    return "".join(parts)


def is_available() -> bool:
    return SPELLCHECK_ENABLED and _get_symspell() is not None


def is_bert_available() -> bool:
    """True si BETO ONNX está cargado y operativo."""
    if not is_available():
        return False
    session, tok, _ = _get_bert()
    return session is not None and tok is not None
