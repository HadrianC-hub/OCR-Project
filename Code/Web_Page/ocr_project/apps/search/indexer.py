"""
apps/search/indexer.py — Operaciones de alto nivel sobre el índice.

Tres operaciones principales:
  • index_page(page):       indexa UNA página recién OCR'ada
  • remove_document(id):    elimina TODOS los chunks de un documento
  • reindex_all():          reconstruye el store desde cero

`index_page` se llama desde apps/ocr/tasks.py cuando una página completa
su OCR. Es síncrono pero rápido (50-200 ms por página típica). El
modelo se carga una sola vez en memoria por proceso (lazy).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from apps.search import encoder
from apps.search.chunker import chunk_page_text
from apps.search.service import get_store
from apps.search.store import ChunkRecord

if TYPE_CHECKING:
    from apps.documents.models import Page

logger = logging.getLogger(__name__)


def index_page(page: "Page", save: bool = True) -> int:
    """
    Indexa una página: lee su texto, chunkea, encode, añade al store.

    Si el modelo no está disponible, no hace nada (degrada limpiamente).
    Si la página no tenía un texto previo (o es ruido), tampoco hace
    nada.

    Args:
        page: instancia de apps.documents.models.Page
        save: si True, persiste el store en disco después (default True).
              Pasar False si vas a indexar muchas páginas seguidas y
              guardar al final manualmente, para no escribir el .npy
              entero N veces.

    Returns:
        Número de chunks indexados (0 si nada).
    """
    if not encoder.is_available():
        return 0

    text = page.get_text()
    if not text or not text.strip():
        return 0

    tokenizer = encoder.get_tokenizer()
    chunks_local = chunk_page_text(text, page.order, tokenizer)
    if not chunks_local:
        return 0

    # Eliminamos sub-chunks previos de ESTA página antes de añadir los nuevos.
    # No basta con remove_document, queremos preservar las OTRAS páginas.
    store = get_store()
    store.remove_page(page.document_id, page.order)

    # Encode batched (aunque hay 1-3 chunks típicamente, el batching da igual)
    embeddings = encoder.encode_passages([c.text for c in chunks_local])
    if embeddings is None:
        return 0

    records = [
        ChunkRecord(
            doc_id=page.document_id,
            page_order=c.page_order,
            sub_chunk_index=c.sub_chunk_index,
            text=c.text[:500],   # limitamos snippet en disco
        )
        for c in chunks_local
    ]
    # CRÍTICO: replace_existing=False. Esta es la razón:
    #   1. remove_page() arriba borró los chunks DE ESTA PÁGINA
    #      preservando los chunks de las OTRAS páginas del documento.
    #   2. Si llamáramos add_chunks() con replace_existing=True (default),
    #      como el doc_id sigue presente en el store (por las otras
    #      páginas), add_chunks haría _remove_unlocked(doc_id) y borraría
    #      todas las páginas — quedaría solo la página que estamos
    #      re-indexando.
    #   3. Bug previo: con el default puesto, cada llamada a index_page
    #      durante el OCR asíncrono iba dejando el índice con sólo la
    #      última página procesada de cada documento.
    store.add_chunks(page.document_id, records, embeddings,
                     replace_existing=False)

    if save:
        try:
            store.save()
        except Exception as exc:
            logger.warning("Error guardando el store: %s", exc)

    return len(chunks_local)


def remove_document(doc_id: int, save: bool = True) -> int:
    """
    Elimina todos los chunks de un documento del store. Se llama desde
    delete_document. Devuelve cuántos chunks quitó.
    """
    store = get_store()
    n = store.remove_document(doc_id)
    if save and n > 0:
        try:
            store.save()
        except Exception as exc:
            logger.warning("Error guardando el store tras delete: %s", exc)
    return n


def reindex_all(verbose: bool = False) -> dict:
    """
    Reconstruye el store entero desde cero. Lee todos los documentos
    de la BD, toma sus páginas con texto, chunkea, encode con batching
    grande, persiste.

    Args:
        verbose: imprime progreso por stdout (útil para management cmd).

    Returns:
        dict con estadísticas: n_docs, n_pages, n_chunks, elapsed_s
    """
    import time
    from apps.documents.models import Document  # noqa: PLC0415

    if not encoder.is_available():
        raise RuntimeError(
            "El modelo E5 no está disponible. Comprueba la instalación "
            "(sentence-transformers + carpeta models/multilingual-e5-small)."
        )

    t0 = time.time()
    store = get_store()
    store.clear()  # vaciado total

    tokenizer = encoder.get_tokenizer()

    # Acumulamos chunks de varias páginas y embedeamos por lotes grandes.
    # OJO: el contador es de CHUNKS, no de páginas — el nombre antiguo
    # `BATCH_PAGES` era engañoso porque se comparaba con
    # len(pending_texts), que cuenta sub-chunks. Una página que se
    # parte en 4 sub-chunks contribuye 4 al contador.
    BATCH_CHUNKS = 50
    pending_records: list[ChunkRecord] = []
    pending_texts:   list[str]         = []
    pending_by_doc:  dict[int, list[int]] = {}  # doc_id -> índices en pending_records

    def flush_batch():
        nonlocal pending_records, pending_texts, pending_by_doc
        if not pending_texts:
            return
        if verbose:
            print(f"  Embeddeando lote de {len(pending_texts)} chunks...")
        embs = encoder.encode_passages(pending_texts, batch_size=32)
        if embs is None:
            return
        # Añadir por documento. CRÍTICO: replace_existing=False.
        # Si un documento queda repartido entre varios flush_batch
        # (cosa habitual para libros largos), el primer flush añade sus
        # chunks iniciales y el segundo añade los siguientes. Con
        # replace_existing=True (default) el segundo flush borraría
        # los chunks que metió el primero antes de añadir los nuevos
        # — pérdida silenciosa de datos. Aquí sabemos que store.clear()
        # se llamó al inicio de reindex_all, así que nunca hay chunks
        # previos "legítimos" que conservar de iteraciones anteriores;
        # appendear es lo correcto.
        for doc_id, idxs in pending_by_doc.items():
            doc_records = [pending_records[i] for i in idxs]
            doc_embs    = embs[idxs]
            store.add_chunks(doc_id, doc_records, doc_embs,
                             replace_existing=False)
        pending_records = []
        pending_texts = []
        pending_by_doc = {}

    docs = Document.objects.all()
    n_docs = 0
    n_pages = 0
    n_chunks = 0

    for doc in docs.iterator():
        n_docs += 1
        if verbose:
            print(f"[{n_docs}] {doc.id} '{doc.title[:50]}'")
        for page in doc.pages.order_by('order'):
            text = page.get_text()
            if not text or not text.strip():
                continue
            chunks_local = chunk_page_text(text, page.order, tokenizer)
            if not chunks_local:
                continue
            n_pages += 1
            for ch in chunks_local:
                pending_records.append(ChunkRecord(
                    doc_id=doc.id, page_order=ch.page_order,
                    sub_chunk_index=ch.sub_chunk_index,
                    text=ch.text[:500],
                ))
                pending_texts.append(ch.text)
                pending_by_doc.setdefault(doc.id, []).append(
                    len(pending_records) - 1
                )
                n_chunks += 1
            if len(pending_texts) >= BATCH_CHUNKS:
                flush_batch()

    flush_batch()
    store.save()
    elapsed = time.time() - t0

    stats = {
        "n_docs":     n_docs,
        "n_pages":    n_pages,
        "n_chunks":   n_chunks,
        "elapsed_s":  round(elapsed, 2),
    }
    if verbose:
        print(f"\n✓ Reindex completo. {stats}")
    return stats
