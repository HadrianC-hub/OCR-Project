"""
apps/search/store.py — Almacén vectorial con persistencia atómica.

Mejoras sobre el SRI original:
  • Granularidad por (doc_id, page_order, sub_chunk_index) — soporta
    múltiples sub-chunks por página (cuando la página no cabe en 1)
  • Embeddings ya normalizados L2 al añadir → búsqueda = dot product puro
  • Save atómico (tmp + rename) → resistente a crashes en mitad del save
  • Sidecar JSON con metadata del modelo y versión → detecta inconsistencias
  • Add / remove a nivel de documento (todos sus chunks de golpe)
  • Top-k consciente de páginas: agrupa por (doc_id, page_order) y
    devuelve la mejor sub-chunk de cada página
  • Contador `version` interno que se incrementa en cada mutación —
    permite a caches externas (BM25) detectar invalidación correctamente
    incluso cuando num_chunks no cambia (swap de docs del mismo tamaño)
  • snapshot_records() expone un copy thread-safe de los records sin
    obligar a los consumidores a tocar atributos privados

Formato en disco:
  {path}.json   → lista de [doc_id, page_order, sub_chunk_index, text]
  {path}.npy    → matriz (N, D) float32 normalizada L2
  {path}.meta.json → {"model": "...", "version": 1, "indexed_at": "..."}
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

STORE_FORMAT_VERSION = 1


# ── Modelo de datos ───────────────────────────────────────────────────────

@dataclass
class ChunkRecord:
    """Un sub-chunk indexado. El embedding vive en la matriz aparte."""
    doc_id: int               # Document.pk
    page_order: int           # Page.order dentro del documento
    sub_chunk_index: int      # 0 si la página entró en un chunk
    text: str                 # texto del sub-chunk (para snippet)


# ── VectorStore ───────────────────────────────────────────────────────────

class VectorStore:
    """
    Repositorio vectorial. Thread-safe vía un Lock; los embeddings se
    asumen ya normalizados L2 al añadir.
    """

    def __init__(self, store_path: Path, model_signature: str = ""):
        """
        Args:
            store_path: ruta base (sin extensión); se crearán .json, .npy, .meta.json
            model_signature: identificador del modelo (para validar al cargar)
        """
        self.store_path = Path(store_path)
        self.model_signature = model_signature
        self._records: list[ChunkRecord] = []
        self._matrix: Optional[np.ndarray] = None  # (N, D) normalizada L2
        self._lock = threading.Lock()
        self._dirty = False
        self._dim: Optional[int] = None
        # Contador monotónico de mutaciones. Sirve a las caches externas
        # (por ejemplo BM25 en hybrid.py) como llave de invalidación.
        # num_chunks no basta: un remove+add con el mismo tamaño no
        # cambiaría num_chunks y la cache devolvería resultados rancios.
        self._version: int = 0

    # ── Propiedades ────────────────────────────────────────────────────

    @property
    def num_chunks(self) -> int:
        return len(self._records)

    @property
    def num_docs(self) -> int:
        return len({r.doc_id for r in self._records})

    @property
    def doc_ids(self) -> set[int]:
        return {r.doc_id for r in self._records}

    @property
    def dim(self) -> Optional[int]:
        return self._dim

    @property
    def version(self) -> int:
        """
        Contador de mutaciones. Se incrementa en cada add_chunks,
        remove_document, clear o load. NO se persiste a disco; se
        resetea a 0 al re-instanciar el VectorStore.
        """
        return self._version

    def __repr__(self) -> str:
        return (f"VectorStore(docs={self.num_docs}, chunks={self.num_chunks}, "
                f"dim={self._dim}, version={self._version}, "
                f"dirty={self._dirty})")

    # ── Add / Remove ───────────────────────────────────────────────────

    def add_chunks(self, doc_id: int,
                   chunks: list[ChunkRecord],
                   embeddings: np.ndarray,
                   replace_existing: bool = True) -> None:
        """
        Añade chunks de un documento. Los embeddings deben venir ya
        normalizados L2 (encoder.encode_passages los devuelve así).

        Args:
            doc_id: id del documento
            chunks: lista de ChunkRecord (sin embedding, solo metadatos+texto)
            embeddings: matriz (len(chunks), D) float32 normalizada L2
            replace_existing: si True (default), elimina cualquier chunk
                previo del mismo doc_id antes de añadir; es lo que quiere
                index_page cuando re-indexa una página tras una corrección
                OCR. Si False, los nuevos chunks se APPENDEAN a los
                existentes del mismo doc: lo que necesita reindex_all
                cuando un documento queda repartido entre varios batches
                de encoding (sin esto, el segundo flush_batch borraría
                lo que añadió el primero).
        """
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"len(chunks)={len(chunks)} != len(embeddings)={len(embeddings)}"
            )

        with self._lock:
            # Si ya existía y nos piden reemplazo, limpiar primero
            if replace_existing and doc_id in self.doc_ids:
                self._remove_unlocked(doc_id)

            self._records.extend(chunks)

            new_mat = np.asarray(embeddings, dtype=np.float32)
            if self._matrix is None or self._matrix.shape[0] == 0:
                self._matrix = new_mat
                self._dim = new_mat.shape[1]
            else:
                if new_mat.shape[1] != self._matrix.shape[1]:
                    raise ValueError(
                        f"Dimensión de embeddings inconsistente: "
                        f"esperaba {self._matrix.shape[1]}, recibió {new_mat.shape[1]}"
                    )
                self._matrix = np.vstack([self._matrix, new_mat])
            self._dirty = True
            self._version += 1

    def remove_document(self, doc_id: int) -> int:
        """Elimina todos los chunks de un documento. Devuelve cuántos quitó."""
        with self._lock:
            n_removed = self._remove_unlocked(doc_id)
            if n_removed > 0:
                self._version += 1
            return n_removed

    def remove_page(self, doc_id: int, page_order: int) -> int:
        """
        Elimina los chunks de UNA página específica preservando las
        demás páginas del documento. Devuelve cuántos chunks quitó.

        Útil para `index_page`, que re-indexa una sola página tras una
        corrección OCR y no quiere tocar las otras páginas del doc.
        """
        with self._lock:
            if not self._records or self._matrix is None:
                return 0
            keep_mask = np.array(
                [not (r.doc_id == doc_id and r.page_order == page_order)
                 for r in self._records],
                dtype=bool,
            )
            n_removed = int((~keep_mask).sum())
            if n_removed == 0:
                return 0
            self._records = [r for r, k in zip(self._records, keep_mask) if k]
            self._matrix = self._matrix[keep_mask]
            if self._matrix.shape[0] == 0:
                self._matrix = None
            self._dirty = True
            self._version += 1
            return n_removed

    def _remove_unlocked(self, doc_id: int) -> int:
        """Sin lock — llamar solo dentro de un with self._lock."""
        keep_mask = np.array([r.doc_id != doc_id for r in self._records],
                             dtype=bool)
        n_removed = int((~keep_mask).sum())
        if n_removed == 0:
            return 0
        self._records = [r for r, k in zip(self._records, keep_mask) if k]
        if self._matrix is not None:
            self._matrix = self._matrix[keep_mask]
            if self._matrix.shape[0] == 0:
                self._matrix = None
        self._dirty = True
        return n_removed

    def clear(self) -> None:
        with self._lock:
            had_something = bool(self._records) or self._matrix is not None
            self._records = []
            self._matrix = None
            self._dirty = True
            if had_something:
                self._version += 1

    # ── Acceso seguro a los records (para caches externas) ────────────

    def snapshot_records(self) -> tuple[list[ChunkRecord], int]:
        """
        Devuelve (copia_de_records, version) bajo el lock interno.

        Pensado para que caches externas (la cache BM25 en hybrid.py)
        reconstruyan su índice sin tener que tocar atributos privados
        del store ni manipular el lock por sí mismas. La copia es
        superficial — los ChunkRecord son dataclasses inmutables en
        la práctica, así que compartirlos es seguro.
        """
        with self._lock:
            return list(self._records), self._version

    # ── Search ─────────────────────────────────────────────────────────

    def search(self, query_emb: np.ndarray,
               top_k: int = 20,
               max_per_doc: int = 1) -> list[dict]:
        """
        Busca los chunks más similares y agrupa por (doc_id, page_order).

        Args:
            query_emb: vector (D,) normalizado L2 (encoder.encode_query)
            top_k: número de resultados finales (a nivel de página, no chunk)
            max_per_doc: cuántas páginas como máximo por documento en los
                resultados. 1 = el sistema devuelve la página más relevante
                de cada doc. 3 = puede devolver hasta 3 páginas del mismo doc.

        Returns:
            Lista de dicts: {doc_id, page_order, sub_chunk_index, score, snippet}
            ordenada por score descendente.
        """
        with self._lock:
            if self._matrix is None or self._matrix.shape[0] == 0:
                return []
            # Como ambos están normalizados, cosine = dot product
            scores = self._matrix @ np.asarray(query_emb, dtype=np.float32)

            # Tomamos top candidatos a nivel de chunk para luego dedupar por página
            n_consider = min(len(scores), max(top_k * 5, 50))
            # argpartition es O(N) frente al O(N log N) del sort completo
            idx = np.argpartition(-scores, n_consider - 1)[:n_consider]
            idx = idx[np.argsort(-scores[idx])]

            # Dedup por (doc_id, page_order), respetando max_per_doc
            seen_pages: set[tuple[int, int]] = set()
            per_doc_count: dict[int, int] = {}
            results: list[dict] = []
            for i in idx:
                rec = self._records[i]
                key = (rec.doc_id, rec.page_order)
                if key in seen_pages:
                    continue
                if per_doc_count.get(rec.doc_id, 0) >= max_per_doc:
                    continue
                seen_pages.add(key)
                per_doc_count[rec.doc_id] = per_doc_count.get(rec.doc_id, 0) + 1
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

    # ── Persistencia ───────────────────────────────────────────────────

    def save(self) -> None:
        """
        Persiste el store en disco de forma atómica: escribimos a
        ficheros .tmp y al final renombramos. Si el proceso muere a
        media escritura, los ficheros viejos sobreviven intactos.
        """
        with self._lock:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)

            meta_path  = self.store_path.with_suffix(".json")
            emb_path   = self.store_path.with_suffix(".npy")
            sig_path   = self.store_path.with_suffix(".meta.json")
            meta_tmp = meta_path.with_suffix(".json.tmp")
            emb_tmp  = emb_path.with_suffix(".npy.tmp")
            sig_tmp  = sig_path.with_suffix(".meta.json.tmp")

            # 1. Metadatos de chunks
            meta = [
                [r.doc_id, r.page_order, r.sub_chunk_index, r.text]
                for r in self._records
            ]
            meta_tmp.write_text(json.dumps(meta, ensure_ascii=False),
                                encoding="utf-8")

            # 2. Matriz de embeddings — np.save añade '.npy' si el path no
            # lo lleva, así que abrimos el fichero nosotros para escribir
            # al nombre exacto que necesitamos para el rename atómico.
            with open(emb_tmp, "wb") as f:
                if self._matrix is not None and len(self._matrix) > 0:
                    np.save(f, self._matrix, allow_pickle=False)
                else:
                    np.save(f, np.empty((0,), dtype=np.float32),
                            allow_pickle=False)

            # 3. Sidecar de metadata
            sig = {
                "model":      self.model_signature,
                "version":    STORE_FORMAT_VERSION,
                "dim":        self._dim,
                "num_chunks": self.num_chunks,
                "num_docs":   self.num_docs,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
            sig_tmp.write_text(json.dumps(sig, ensure_ascii=False, indent=2),
                               encoding="utf-8")

            # 4. Renombrado atómico (en POSIX y NTFS un rename es atómico)
            os.replace(meta_tmp, meta_path)
            os.replace(emb_tmp,  emb_path)
            os.replace(sig_tmp,  sig_path)

            self._dirty = False
            logger.info("VectorStore guardado: %s", self)

    def load(self) -> bool:
        """
        Carga el store desde disco. Devuelve True si lo cargó, False si
        no había nada que cargar O si los ficheros estaban corruptos.

        Si el modelo del store no coincide con el modelo actual,
        AVISA con WARNING y carga igualmente — pero la búsqueda dará
        resultados malos hasta que se ejecute reindex_search.

        Tolerancia a fallos: cada paso (metadata, matriz, sidecar) se
        envuelve en try/except. Si el JSON de metadatos o el .npy de
        embeddings están corruptos, el store queda VACÍO (no en estado
        intermedio inconsistente) y se loguea el detalle del error.
        """
        meta_path = self.store_path.with_suffix(".json")
        emb_path  = self.store_path.with_suffix(".npy")
        sig_path  = self.store_path.with_suffix(".meta.json")

        if not meta_path.is_file():
            logger.info("No existe el store en %s — vacío.", meta_path)
            return False

        with self._lock:
            # 1. Cargar metadatos de records.
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                new_records = [
                    ChunkRecord(
                        doc_id=int(m[0]), page_order=int(m[1]),
                        sub_chunk_index=int(m[2]), text=str(m[3]),
                    )
                    for m in meta
                ]
            except (json.JSONDecodeError, ValueError, TypeError,
                    IndexError, KeyError) as exc:
                logger.error(
                    "Metadata corrupta en %s (%s: %s). "
                    "Store queda vacío; ejecutá `reindex_search` para "
                    "reconstruir.",
                    meta_path, type(exc).__name__, exc,
                )
                self._records = []
                self._matrix = None
                self._dim = None
                self._dirty = False
                self._version += 1
                return False

            # 2. Cargar matriz de embeddings (si existe). Una matriz
            # corrupta NO debería invalidar los records — pero como
            # search() depende de la matriz y BM25 sólo necesita los
            # records, dejamos los records cargados y la matriz a None.
            new_matrix: Optional[np.ndarray] = None
            new_dim: Optional[int] = None
            if emb_path.is_file():
                try:
                    mat = np.load(emb_path)
                    if mat.ndim == 2 and mat.shape[0] > 0:
                        new_matrix = mat.astype(np.float32, copy=False)
                        new_dim = mat.shape[1]
                except (ValueError, OSError, EOFError) as exc:
                    logger.error(
                        "Embeddings corruptos en %s (%s: %s). "
                        "Records cargados pero búsqueda semántica "
                        "deshabilitada hasta `reindex_search`.",
                        emb_path, type(exc).__name__, exc,
                    )

            # 3. Si los embeddings cargaron pero la cuenta no cuadra
            # con los records, hay un desincronizado serio en disco —
            # mejor no usar la matriz.
            if new_matrix is not None and len(new_records) != new_matrix.shape[0]:
                logger.error(
                    "Inconsistencia: %d records vs %d embeddings. "
                    "Deshabilitando búsqueda semántica.",
                    len(new_records), new_matrix.shape[0],
                )
                new_matrix = None
                new_dim = None

            # 4. Commit del estado cargado
            self._records = new_records
            self._matrix = new_matrix
            self._dim = new_dim

            # 5. Sidecar (validación, no afecta al estado funcional)
            if sig_path.is_file():
                try:
                    sig = json.loads(sig_path.read_text(encoding="utf-8"))
                    if (self.model_signature
                            and sig.get("model")
                            and sig["model"] != self.model_signature):
                        logger.warning(
                            "El índice fue creado con modelo %r pero el actual "
                            "es %r. Ejecutá `python manage.py reindex_search` "
                            "para reindexar.",
                            sig["model"], self.model_signature,
                        )
                    if sig.get("version") != STORE_FORMAT_VERSION:
                        logger.warning(
                            "Versión del store desactualizada (%s vs %s).",
                            sig.get("version"), STORE_FORMAT_VERSION,
                        )
                except Exception as exc:
                    logger.warning("No se pudo leer el sidecar %s: %s",
                                   sig_path, exc)

            self._dirty = False
            self._version += 1
            logger.info("VectorStore cargado: %s", self)
            return True
