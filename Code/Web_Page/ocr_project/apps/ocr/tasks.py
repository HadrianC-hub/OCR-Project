"""
apps/ocr/tasks.py — Procesado OCR en background.

Permite que el endpoint de subida retorne inmediatamente y que el OCR
de las páginas se haga en un thread aparte, página a página. El frontend
puede consultar el estado y mostrar/desbloquear cada página a medida que
se completa.

Diseño:
  - Un thread por documento. Si llega una segunda petición para el mismo
    documento mientras el thread anterior sigue vivo, se ignora.
  - El thread recorre las páginas en `pending` por orden, las marca como
    `processing`, ejecuta OCR, guarda el texto y las marca como `done`
    (o `error`).
  - Al terminar todas las pendientes, el thread se cierra y se desregistra.

Limitaciones conocidas (mencionadas en INSTALL/README):
  - Usa threading.Thread, NO una cola externa tipo Celery. Si el proceso
    Django muere a media procesado (crash, reinicio, etc.), las páginas
    en `processing` se quedan así huérfanas. Hay una función de
    `recover_orphans()` que se puede llamar al arranque (apps.py.ready)
    para resucitar esas páginas como `pending` y reintentar.
  - El thread comparte el GIL con las peticiones HTTP. OCR es CPU-bound
    así que en runserver/gunicorn-sync esto NO compite mucho con las
    peticiones (que son I/O-bound). Con muchos workers la cosa empeora;
    para volumen alto, conviene migrar a Celery.
"""

from __future__ import annotations

import logging
import threading
from datetime import timedelta
from pathlib  import Path as pathlib_path

from django.db import close_old_connections
from django.utils import timezone

logger = logging.getLogger(__name__)


# ── Registro de threads en marcha ────────────────────────────────────────

_threads_lock = threading.Lock()
_running_threads: dict[int, threading.Thread] = {}
# Conjunto de doc_ids marcados como "cancelar lo antes posible". El thread
# los consulta entre página y página y sale del bucle si su doc_id está
# aquí. NO interrumpe el OCR de la página actual (eso requeriría matar
# al subproceso de Tesseract o similar, que no soportamos).
_cancelled_docs: set[int] = set()


def start_document_ocr(doc_id: int) -> bool:
    """
    Arranca el OCR background para un documento. Si ya hay un thread vivo
    procesándolo, no hace nada.

    Returns:
        True si arrancó un thread nuevo, False si ya había uno.
    """
    with _threads_lock:
        existing = _running_threads.get(doc_id)
        if existing is not None and existing.is_alive():
            logger.info("OCR doc %s: ya hay un thread en marcha; ignoro.", doc_id)
            return False

        # Si había una cancelación pendiente para este doc_id (de un
        # uso anterior del mismo id, por ej. tras un reinicio), la
        # limpiamos antes de arrancar.
        _cancelled_docs.discard(doc_id)

        t = threading.Thread(
            target=_run_document_ocr,
            args=(doc_id,),
            daemon=True,
            name=f"ocr-doc-{doc_id}",
        )
        _running_threads[doc_id] = t
        t.start()
        logger.info("OCR doc %s: thread arrancado.", doc_id)
        return True


def cancel_document_ocr(doc_id: int) -> bool:
    """
    Marca un documento para que su thread de OCR salga del bucle en la
    próxima iteración entre páginas. NO mata la página que esté en curso.

    Llamar desde `abandon_document` para que el thread no gaste CPU
    procesando páginas que el usuario ya ha decidido borrar.

    Returns:
        True si había un thread vivo al que cancelar, False si no.
    """
    with _threads_lock:
        had_thread = (doc_id in _running_threads
                      and _running_threads[doc_id].is_alive())
        _cancelled_docs.add(doc_id)
        if had_thread:
            logger.info("OCR doc %s: marcado para cancelar.", doc_id)
        return had_thread


def is_document_processing(doc_id: int) -> bool:
    """True si hay un thread vivo procesando este documento."""
    with _threads_lock:
        t = _running_threads.get(doc_id)
        return t is not None and t.is_alive()


# ── El runner ────────────────────────────────────────────────────────────

def _run_document_ocr(doc_id: int) -> None:
    """Procesa secuencialmente todas las páginas en estado `pending`."""
    from apps.documents.models import Document, Page  # noqa: PLC0415
    from apps.ocr.views import _ocr_for_document         # noqa: PLC0415

    try:
        try:
            doc = Document.objects.get(pk=doc_id)
        except Document.DoesNotExist:
            logger.warning("OCR background: documento %s no existe.", doc_id)
            return

        ocr_fn   = _ocr_for_document(doc)
        use_bert = bool(doc.use_bert_correction)
        doc_type = doc.document_type
        logger.info("OCR doc %s: arrancando (use_bert=%s, type=%s).",
                    doc_id, use_bert, doc_type)

        while True:
            # Comprobación de cancelación
            with _threads_lock:
                if doc_id in _cancelled_docs:
                    logger.info("OCR doc %s: cancelado por abandono.", doc_id)
                    break

            if not Document.objects.filter(pk=doc_id).exists():
                logger.info("OCR doc %s: el documento ya no existe; paro.",
                            doc_id)
                break

            page = (Page.objects
                    .filter(document_id=doc_id, ocr_status=Page.OCR_PENDING)
                    .order_by('order')
                    .first())
            if page is None:
                break

            Page.objects.filter(pk=page.pk).update(
                ocr_status=Page.OCR_PROCESSING,
                ocr_error='',
            )
            logger.info("OCR doc %s p.%s: processing...", doc_id, page.order)

            # Pre-cache de la visualización de segmentación.
            # Lo hacemos AQUÍ, dentro del thread OCR, por dos motivos:
            #   1. La detección de líneas que hace el OCR es la misma
            #      que necesita el frontend para mostrar las cajas; si
            #      no pre-cacheamos, cuando el navegador pida la imagen
            #      vía <img src="line_segmentation_image">, ejecutaría
            #      el pipeline una segunda vez en paralelo, monopolizando
            #      la CPU mientras el OCR también la usa. Eso "colgaba"
            #      la respuesta HTTP durante 30 s en manuscritos grandes.
            #   2. El usuario ve las cajas en cuanto la página termina,
            #      sin un segundo retraso al cargar la imagen.
            # No es crítico si falla — la viz se podría regenerar después.
            try:
                _precompute_segmentation(page, doc_type)
            except Exception:
                logger.exception(
                    "Error pre-cacheando segmentación de doc %s p.%s "
                    "(no crítico).", doc_id, page.order,
                )

            # Ejecutar OCR (puede tardar segundos o minutos)
            try:
                text = ocr_fn(page.image.path, use_bert=use_bert)
            except Exception as exc:
                logger.exception("OCR doc %s p.%s falló.", doc_id, page.order)
                Page.objects.filter(pk=page.pk).update(
                    ocr_status=Page.OCR_ERROR,
                    ocr_error=str(exc)[:500],
                )
                close_old_connections()
                continue

            try:
                page.refresh_from_db()
                page.text = text
                Page.objects.filter(pk=page.pk).update(
                    ocr_status=Page.OCR_DONE,
                    ocr_error='',
                )
                logger.info("OCR doc %s p.%s: done.", doc_id, page.order)
            except Exception as exc:
                logger.exception("OCR doc %s p.%s: error guardando texto.",
                                 doc_id, page.order)
                Page.objects.filter(pk=page.pk).update(
                    ocr_status=Page.OCR_ERROR,
                    ocr_error=f"Guardando texto: {exc}"[:500],
                )
                close_old_connections()
                continue

            # Indexar para búsqueda semántica.
            if Page.objects.filter(pk=page.pk).exists():
                try:
                    from apps.search.indexer import index_page  # noqa: PLC0415
                    n_chunks = index_page(page)
                    if n_chunks:
                        logger.info("OCR doc %s p.%s: indexada (%d chunks).",
                                    doc_id, page.order, n_chunks)
                except Exception:
                    logger.exception("Error indexando doc %s p.%s (no crítico).",
                                     doc_id, page.order)
            else:
                logger.info("OCR doc %s p.%s: página eliminada antes del "
                            "indexado; salto.", doc_id, page.order)

            close_old_connections()

        logger.info("OCR doc %s: completado.", doc_id)
    finally:
        close_old_connections()
        with _threads_lock:
            _running_threads.pop(doc_id, None)
            _cancelled_docs.discard(doc_id)


def _precompute_segmentation(page, doc_type: str) -> None:
    """
    Genera y guarda en cache la imagen de segmentación de `page`.

    Reusa `pipeline.run` (para impresos) o `detect_manuscript_lines`
    (para manuscritos), produciendo el mismo resultado que más tarde
    consumiría `apps/ocr/views.line_segmentation_image`. Al guardar la
    cache aquí, el request del frontend simplemente sirve un fichero
    estático sin tocar pipeline alguno.

    Si la cache ya existe y es más nueva que el facsimilar, no hace
    nada (idempotente).
    """
    from apps.ocr import segmentation  # noqa: PLC0415
    out = segmentation.viz_path(page.document_id, page.order)
    src = pathlib_path(page.image.path)
    if out.is_file() and src.is_file():
        try:
            if out.stat().st_mtime >= src.stat().st_mtime:
                return  # ya está fresca
        except OSError:
            pass

    segmentation.get_or_generate(
        page.image.path, page.document_id, page.order,
        doc_type=doc_type, allow_block=True, force=True,
    )


# ── Recuperación de páginas huérfanas ────────────────────────────────────

def recover_orphans(stale_minutes: int = 30) -> int:
    """
    Si Django se reinició dejando páginas en `processing`, las resucita
    como `pending` para que un próximo `start_document_ocr` las
    reintente. Se llama al arranque en apps.py.ready().

    Args:
        stale_minutes: cuántos minutos de "antigüedad" en estado processing
            consideramos suficiente para asumir que el thread original ya
            no existe.

    Returns:
        Número de páginas resucitadas.
    """
    from apps.documents.models import Page  # noqa: PLC0415

    # Como no tenemos timestamp por estado, resucitamos TODAS las que
    # estén en processing al arranque. En la práctica, justo después del
    # arranque no hay threads en marcha, así que cualquier `processing`
    # es huérfana.
    n = (Page.objects
         .filter(ocr_status=Page.OCR_PROCESSING)
         .update(ocr_status=Page.OCR_PENDING))
    if n:
        logger.warning("Recuperadas %d páginas huérfanas en `processing`.", n)
    return n
