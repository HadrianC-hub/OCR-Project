"""
apps/search/chunker.py — Sub-chunking adaptativo por página, token-aware.

Filosofía:
  - Cada página del documento se intenta indexar como UN sub-chunk.
  - Si la página excede el límite de tokens del modelo (E5: 512),
    se parte en ventanas deslizantes solapadas.
  - Los sub-chunks de una misma página comparten `page_order` y se
    diferencian con `sub_chunk_index`.
  - El chunking usa el tokenizador real del modelo (no conteo de palabras)
    para garantizar que ninguna ventana exceda el límite.

Esto preserva la unidad lógica "página" en los resultados — el usuario
ve "doc X, página 17, score Y" — mientras evita el truncamiento
silencioso que pasaría si embedeáramos páginas largas enteras.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ── Configuración ─────────────────────────────────────────────────────────

# Tokens efectivos disponibles para texto (descontando especiales + prefijo)
MAX_CHUNK_TOKENS = 400
OVERLAP_TOKENS   = 50

# Si una página tiene menos texto que MIN_CHARS, no se indexa (ruido).
MIN_CHARS = 20


@dataclass
class PageChunk:
    """Un sub-chunk de una página. Si la página cabe entera, hay solo uno."""
    page_order: int
    sub_chunk_index: int      # 0 si la página entró en un único chunk
    text: str                 # el texto del sub-chunk (no truncado)


def chunk_page_text(text: str,
                    page_order: int,
                    tokenizer,
                    max_tokens: int = MAX_CHUNK_TOKENS,
                    overlap_tokens: int = OVERLAP_TOKENS) -> list[PageChunk]:
    """
    Trocea el texto de una página en uno o varios sub-chunks.

    Si la página entera cabe en `max_tokens` tokens del tokenizador,
    devuelve una lista de UN solo PageChunk. Si no, devuelve una
    ventana deslizante con solapamiento.

    Args:
        text:           texto OCR de la página
        page_order:     número de página dentro del documento
        tokenizer:      AutoTokenizer del modelo (lo devuelve encoder.get_tokenizer())
        max_tokens:     tope de tokens por sub-chunk (default 400)
        overlap_tokens: solapamiento entre sub-chunks consecutivos (default 50)

    Returns:
        Lista de PageChunk. Vacía si el texto es ruido (< MIN_CHARS).
    """
    if not text or len(text.strip()) < MIN_CHARS:
        return []

    text = text.strip()

    # Si no hay tokenizador, fallback por palabras (peor pero funcional)
    if tokenizer is None:
        return _chunk_by_words_fallback(text, page_order, max_tokens, overlap_tokens)

    # Tokenizar SIN add_special_tokens — los añadirá el encoder solo
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    input_ids = enc["input_ids"]
    offsets   = enc["offset_mapping"]   # [(start, end), ...] en chars

    total = len(input_ids)

    # Caso simple: cabe entero
    if total <= max_tokens:
        return [PageChunk(page_order=page_order, sub_chunk_index=0, text=text)]

    # Caso ventana deslizante
    step = max_tokens - overlap_tokens
    if step <= 0:
        raise ValueError("max_tokens debe ser mayor que overlap_tokens.")

    chunks: list[PageChunk] = []
    sub_idx = 0
    i = 0
    while i < total:
        j = min(i + max_tokens, total)
        # Convertimos índices de tokens a slice de caracteres
        char_start = offsets[i][0]
        char_end   = offsets[j - 1][1]
        sub_text = text[char_start:char_end].strip()
        if sub_text:
            chunks.append(PageChunk(
                page_order=page_order,
                sub_chunk_index=sub_idx,
                text=sub_text,
            ))
            sub_idx += 1
        if j == total:
            break
        i += step

    return chunks


def _chunk_by_words_fallback(text: str, page_order: int,
                             max_tokens: int, overlap_tokens: int) -> list[PageChunk]:
    """
    Fallback cuando no hay tokenizador disponible. Aproxima 1 token ≈ 0.7
    palabras (en español con E5), con margen conservador.
    """
    words_per_chunk   = int(max_tokens * 0.65)
    words_per_overlap = int(overlap_tokens * 0.65)
    words = text.split()

    if len(words) <= words_per_chunk:
        return [PageChunk(page_order=page_order, sub_chunk_index=0, text=text)]

    step = words_per_chunk - words_per_overlap
    chunks: list[PageChunk] = []
    sub_idx = 0
    for i in range(0, len(words), step):
        ws = words[i : i + words_per_chunk]
        sub_text = " ".join(ws).strip()
        if sub_text:
            chunks.append(PageChunk(
                page_order=page_order, sub_chunk_index=sub_idx, text=sub_text,
            ))
            sub_idx += 1
        if i + words_per_chunk >= len(words):
            break
    return chunks
