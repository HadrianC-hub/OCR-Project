"""
apps/search/hybrid.py — Re-ranking híbrido BM25 + E5 con RRF.

Por qué híbrido:
  - E5 captura semántica (sinónimos, paráfrasis): mejor recall en
    consultas naturales.
  - BM25 gana en consultas literales (nombre propio, expresión exacta).
  - Combinarlos con Reciprocal Rank Fusion (RRF) suele subir el MAP
    5-10 puntos sin tunear nada.

Por qué BM25 (Best Matching 25, Robertson & Walker, 1994) en lugar de TF-IDF:
  - Saturación de TF: el aporte de un término no crece linealmente al
    repetirse — controlado por el parámetro k1 (típico 1.2-2.0).
  - Normalización por longitud: documentos cortos no se ven favorecidos
    artificialmente — controlado por b (típico 0.75).
  - Es el estándar de facto: lo que usan Lucene, Elasticsearch,
    OpenSearch y prácticamente todos los motores de búsqueda léxica
    modernos.

RRF score = sum_over_rankers( 1 / (k + rank) ).
  - k=60 es el valor estándar de la literatura (Cormack 2009).
  - Rank empieza en 1.
  - Documentos que NO aparecen en un ranker no aportan score de ese.

El índice BM25 se mantiene en memoria del proceso, reconstruido la
primera vez que se hace una búsqueda híbrida, y se invalida cuando
cambia el número de chunks del store.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

RRF_K = 60

# Hiperparámetros BM25 — los defaults de Lucene/Elasticsearch.
# k1 controla la saturación de la frecuencia de término: valores más
# altos hacen que repetir un término aporte más score; 1.5 es un
# compromiso razonable. b controla cuánto se penaliza la longitud del
# documento; 0.75 es el estándar.
BM25_K1 = 1.5
BM25_B  = 0.75


# ── Tokenización ──────────────────────────────────────────────────────────

# Patrón Unicode: \w incluye letras acentuadas, ñ, dígitos.
# Filtramos tokens de 1 sola letra: aportan poco a la recuperación
# léxica y suelen ser ruido OCR.
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize_for_bm25(text: str) -> list[str]:
    """
    Tokenizador simple para BM25:
      - minúsculas
      - split en cualquier no-alfanumérico (Unicode-aware: conserva
        tildes y ñ)
      - descarta tokens de longitud 1

    Nota: no aplicamos stemming ni eliminación de stop-words. Stemming
    podría subir el recall (gato/gatos/gata serían el mismo token) a
    costa de algo de precisión y de añadir una dependencia adicional
    (snowballstemmer). Si en el futuro se quiere añadir, basta con
    importar SnowballStemmer('spanish') y aplicarlo aquí.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if len(t) > 1]


# ── BM25 (rank_bm25) ──────────────────────────────────────────────────────

_bm25_state = {
    "bm25":     None,
    "records":  None,   # snapshot de chunks asociados
    "version":  -1,     # versión del store usada para construir esta cache
}
_bm25_lock = threading.Lock()


def _build_bm25(texts: list[str]):
    """Construye el índice BM25 desde la lista de textos de chunks."""
    try:
        from rank_bm25 import BM25Okapi  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "rank_bm25 no instalado. Re-ranking híbrido desactivado. "
            "pip install rank_bm25",
        )
        return None

    tokenized = [_tokenize_for_bm25(t) for t in texts]
    # Si la tokenización dejó todos los docs vacíos, BM25 no tiene
    # nada que indexar; salimos limpiamente.
    if not any(tokenized):
        return None
    return BM25Okapi(tokenized, k1=BM25_K1, b=BM25_B)


def _get_or_build_bm25(store):
    """Devuelve (bm25, records) o (None, None).

    La invalidación usa `store.version`, no `store.num_chunks`. Esto
    importa porque un swap de un doc por otro del mismo tamaño deja
    num_chunks intacto pero cambia el contenido: con num_chunks como
    llave, la cache devolvía chunks del doc eliminado.
    """
    with _bm25_lock:
        # Tomamos el snapshot bajo el lock del store (sin tocar privados)
        records, store_version = store.snapshot_records()
        if _bm25_state["version"] != store_version:
            texts = [r.text for r in records]
            if not texts:
                _bm25_state.update(bm25=None, records=None,
                                   version=store_version)
                return None, None
            bm25 = _build_bm25(texts)
            _bm25_state.update(
                bm25=bm25, records=records, version=store_version,
            )
        return _bm25_state["bm25"], _bm25_state["records"]


def search_bm25(query: str, store, top_k: int = 50) -> list[dict]:
    """
    Búsqueda BM25 sobre los chunks del store. Devuelve los top_k
    chunks por score BM25, agrupados por (doc_id, page_order) como
    hace la búsqueda semántica.
    """
    bm25, records = _get_or_build_bm25(store)
    if bm25 is None or not records:
        return []

    q_tokens = _tokenize_for_bm25(query)
    if not q_tokens:
        return []

    scores = np.asarray(bm25.get_scores(q_tokens), dtype=np.float32)

    n_consider = min(len(scores), max(top_k * 5, 100))
    idx = np.argpartition(-scores, n_consider - 1)[:n_consider]
    idx = idx[np.argsort(-scores[idx])]

    seen: set[tuple[int, int]] = set()
    results: list[dict] = []
    for i in idx:
        # Score 0 = ningún término del query aparece en el chunk;
        # no aporta nada al ranking BM25, lo cortamos aquí.
        if scores[i] <= 0:
            break
        rec = records[i]
        key = (rec.doc_id, rec.page_order)
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "doc_id":          rec.doc_id,
            "page_order":      rec.page_order,
            "sub_chunk_index": rec.sub_chunk_index,
            "score":           float(scores[i]),
            "snippet":         rec.text,
        })
        if len(results) >= top_k:
            break
    return results


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────

def rrf_merge(semantic_results: list[dict],
              bm25_results: list[dict],
              top_k: int = 20,
              k: int = RRF_K) -> list[dict]:
    """
    Fusión RRF de dos rankings. Cada documento+página recibe un score:
        sum_rankers 1 / (k + rank_in_ranker)
    Las posiciones empiezan en 1. Documentos que NO aparecen en un
    ranker simplemente no aportan score de ese ranker.

    Devuelve la lista combinada ordenada por score RRF descendente.
    Conserva snippet y scores originales para que la UI pueda mostrar
    ambos si quiere.
    """
    combined: dict[tuple[int, int], dict] = {}

    for rank, r in enumerate(semantic_results, start=1):
        key = (r["doc_id"], r["page_order"])
        combined.setdefault(key, {
            "doc_id":     r["doc_id"],
            "page_order": r["page_order"],
            "sub_chunk_index": r.get("sub_chunk_index", 0),
            "snippet":    r["snippet"],
            "score_semantic": r["score"],
            "score_bm25":     None,
            "score_rrf":      0.0,
        })
        combined[key]["score_rrf"] += 1.0 / (k + rank)
        # Preferimos el snippet semántico
        combined[key]["snippet"] = r["snippet"]

    for rank, r in enumerate(bm25_results, start=1):
        key = (r["doc_id"], r["page_order"])
        if key not in combined:
            combined[key] = {
                "doc_id":     r["doc_id"],
                "page_order": r["page_order"],
                "sub_chunk_index": r.get("sub_chunk_index", 0),
                "snippet":    r["snippet"],
                "score_semantic": None,
                "score_bm25":     r["score"],
                "score_rrf":      0.0,
            }
        else:
            combined[key]["score_bm25"] = r["score"]
        combined[key]["score_rrf"] += 1.0 / (k + rank)

    merged = list(combined.values())
    merged.sort(key=lambda x: x["score_rrf"], reverse=True)
    return merged[:top_k]
