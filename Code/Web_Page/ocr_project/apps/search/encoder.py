"""
apps/search/encoder.py — Carga del modelo E5 + tokenizador + encoding.

El modelo y el tokenizador se cargan de forma diferida (la primera vez
que se necesitan) y se cachean en memoria. Si las librerías o el modelo
no están disponibles, todas las funciones devuelven None y el sistema
degrada elegantemente a búsqueda por metadatos solo.

INSTALACIÓN:
    pip install sentence-transformers   # arrastra transformers + torch

Plus el modelo en `models/multilingual-e5-small/` (ver INSTALL.md).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Configuración ─────────────────────────────────────────────────────────

DEFAULT_BATCH_SIZE = 32
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX   = "query: "

# Tope duro del modelo. E5-small acepta hasta 512 tokens en input;
# dejamos margen para los tokens de control (CLS, SEP) y el prefijo.
MAX_INPUT_TOKENS    = 512
PREFIX_TOKEN_BUDGET = 8


def _resolve_model_dir() -> Path:
    """
    Devuelve la ruta donde buscar el modelo E5. Prioridad:
      1. variable de entorno ARCHIVOOCR_E5_PATH
      2. {BASE_DIR}/models/multilingual-e5-small/  (Django BASE_DIR)
      3. ./models/multilingual-e5-small/  (relativo a CWD)
    """
    env = os.environ.get("ARCHIVOOCR_E5_PATH")
    if env:
        return Path(env)
    try:
        from django.conf import settings  # noqa: PLC0415
        return Path(settings.BASE_DIR) / "models" / "multilingual-e5-small"
    except Exception:
        return Path("models/multilingual-e5-small")


# ── Singletons cargados de forma diferida ─────────────────────────────────

_model = None
_tokenizer = None
_load_attempted = False
# Flag separado para el path "lite" de get_tokenizer(). Se necesita
# aparte porque get_tokenizer puede tener éxito cuando get_model_and_tokenizer
# falla (modelo corrupto pero tokenizador OK), y al revés. Sin este flag,
# cada llamada a get_tokenizer() reintenta la carga y duplica logs.
_tokenizer_load_attempted = False

# Locks para serializar la carga ENTRE threads. El modelo E5 pesa ~140 MB
# y tarda 3-5 segundos en cargar; sin el lock dos threads concurrentes
# (background OCR + petición HTTP de búsqueda, p.ej.) lo cargarían en
# paralelo, duplicando memoria temporalmente.
import threading as _threading
_model_lock     = _threading.Lock()
_tokenizer_lock = _threading.Lock()


def _build_model_and_tokenizer():
    """
    Carga el modelo SentenceTransformer y el tokenizador (Hugging Face
    AutoTokenizer) desde el directorio del modelo. El tokenizador se
    carga aparte porque lo necesitamos para chunkear sin cargar el
    modelo completo (140 MB) si solo queremos chunkear.

    Devuelve (model, tokenizer) o (None, None) si algo falla.
    """
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        from transformers import AutoTokenizer                  # noqa: PLC0415
    except ImportError as exc:
        logger.warning(
            "sentence-transformers/transformers no instalados. "
            "Búsqueda semántica desactivada. Detalle: %s", exc,
        )
        return None, None

    model_dir = _resolve_model_dir()
    if not model_dir.is_dir():
        logger.warning(
            "Modelo E5 no encontrado en %s. "
            "Búsqueda semántica desactivada; sólo se usará metadatos.",
            model_dir,
        )
        return None, None

    try:
        logger.info("Cargando E5 desde %s ...", model_dir)
        m = SentenceTransformer(str(model_dir))
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        logger.info("E5 cargado. dim=%d", m.get_sentence_embedding_dimension())
        return m, tok
    except Exception as exc:
        logger.warning("No se pudo cargar E5: %s", exc)
        return None, None


def get_model_and_tokenizer():
    """Devuelve (model, tokenizer) cargados o (None, None) si falta algo."""
    global _model, _tokenizer, _load_attempted
    if _load_attempted:
        return _model, _tokenizer
    with _model_lock:
        if _load_attempted:
            return _model, _tokenizer
        _load_attempted = True
        _model, _tokenizer = _build_model_and_tokenizer()
        return _model, _tokenizer


def get_tokenizer():
    """Devuelve solo el tokenizador. Útil para chunkear sin cargar el modelo entero."""
    global _tokenizer, _tokenizer_load_attempted

    if _tokenizer is not None:
        return _tokenizer
    if _tokenizer_load_attempted:
        return None

    with _tokenizer_lock:
        if _tokenizer is not None:
            return _tokenizer
        if _tokenizer_load_attempted:
            return None
        _tokenizer_load_attempted = True

        try:
            from transformers import AutoTokenizer  # noqa: PLC0415
        except ImportError:
            return None
        model_dir = _resolve_model_dir()
        if not model_dir.is_dir():
            return None
        try:
            _tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            return _tokenizer
        except Exception as exc:
            logger.warning("No se pudo cargar el tokenizador: %s", exc)
            return None


def is_available() -> bool:
    """True si el modelo está disponible y operativo."""
    m, t = get_model_and_tokenizer()
    return m is not None and t is not None


def model_signature() -> str:
    """Devuelve un identificador del modelo cargado (para validación del store)."""
    return "intfloat/multilingual-e5-small"


# ── Encoding ──────────────────────────────────────────────────────────────

def encode_passages(texts: list[str],
                    batch_size: int = DEFAULT_BATCH_SIZE,
                    show_progress: bool = False) -> Optional[np.ndarray]:
    """
    Codifica una lista de pasajes (texto del documento) en una matriz
    (N, D) de embeddings normalizados L2.

    Usa batching para velocidad. El prefijo "passage: " se aplica
    automáticamente (requisito de E5).

    Returns:
        np.ndarray (N, D) float32, normalizado L2; None si el modelo
        no está disponible.
    """
    model, _tok = get_model_and_tokenizer()
    if model is None:
        return None
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    prefixed = [PASSAGE_PREFIX + t for t in texts]
    embs = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,    # L2 normalizado al vuelo
        convert_to_numpy=True,
    )
    return embs.astype(np.float32, copy=False)


def encode_query(text: str) -> Optional[np.ndarray]:
    """
    Codifica una consulta del usuario en un vector (D,) normalizado L2.

    El prefijo "query: " se aplica automáticamente.

    Returns:
        np.ndarray (D,) float32 normalizado; None si no hay modelo.
    """
    model, _tok = get_model_and_tokenizer()
    if model is None:
        return None
    emb = model.encode(
        QUERY_PREFIX + text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32, copy=False)
