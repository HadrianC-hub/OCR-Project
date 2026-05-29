"""
apps/ocr/views.py — Vistas OCR.

Endpoints:
  - ocr_process(doc_id):           OCR de toda la página, página por página.
  - ocr_single_page(page_id):      AJAX: re-ejecuta OCR sobre una página.
  - ocr_regions_page(page_id):     AJAX: ejecuta OCR sólo sobre las regiones
                                   definidas por el usuario en la página.
  - line_segmentation_image(p_id): GET que devuelve (o genera bajo demanda)
                                   el JPG con las cajas de líneas detectadas.
  - line_segmentation_boxes(p_id): GET JSON con las cajas brutas de líneas
                                   y bloques (útil para el frontend).
"""

from django.shortcuts import redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse, FileResponse, Http404
from django.views.decorators.http import require_POST, require_GET

from apps.accounts.decorators import worker_required
from apps.documents.models import Document, Page
from apps.ocr.ocr_engine import ocr_printed, ocr_manuscript, ocr_regions
from apps.ocr import segmentation


# ── Selección de función OCR según tipo de documento ─────────────────────

def _ocr_for_document(doc: Document):
    if doc.document_type == Document.MANUSCRIPT:
        return ocr_manuscript
    return ocr_printed


# ── Vistas existentes ────────────────────────────────────────────────────

@worker_required
def ocr_process(request, doc_id):
    """
    Arranca el OCR de un documento en background. Marca como `pending`
    todas las páginas que no estén ya en `done`, arranca un thread y
    redirige inmediatamente a edit_document. El frontend muestra el
    progreso y desbloquea cada página según se completa.
    """
    from apps.ocr.tasks import start_document_ocr  # noqa: PLC0415
    doc = get_object_or_404(Document, pk=doc_id)

    # Marcar como pending las páginas no-listas para que el thread las coja.
    # Las páginas que ya estaban `done` (con texto) no se re-procesan.
    Page.objects.filter(document_id=doc_id).exclude(
        ocr_status=Page.OCR_DONE,
    ).update(ocr_status=Page.OCR_PENDING, ocr_error='')

    start_document_ocr(doc_id)
    return redirect('edit_document', doc_id=doc_id)


@worker_required
@require_POST
def ocr_single_page(request, page_id):
    """AJAX: re-ejecuta OCR sobre una página entera, actualizando estado."""
    page = get_object_or_404(Page, pk=page_id)

    if not request.user.is_worker_or_above:
        return JsonResponse({'error': 'Forbidden'}, status=403)

    doc      = page.document
    ocr_fn   = _ocr_for_document(doc)
    use_bert = bool(doc.use_bert_correction)

    Page.objects.filter(pk=page.pk).update(
        ocr_status=Page.OCR_PROCESSING, ocr_error='',
    )

    try:
        ocr_text = ocr_fn(page.image.path, use_bert=use_bert)
    except Exception as exc:
        Page.objects.filter(pk=page.pk).update(
            ocr_status=Page.OCR_ERROR, ocr_error=str(exc)[:500],
        )
        return JsonResponse({'error': str(exc)}, status=500)

    page.refresh_from_db()
    page.text = ocr_text
    Page.objects.filter(pk=page.pk).update(
        ocr_status=Page.OCR_DONE, ocr_error='',
    )

    # Reindexar esta página (idempotente: si ya existía, se sustituye)
    try:
        from apps.search.indexer import index_page  # noqa: PLC0415
        index_page(page)
    except Exception:
        import logging
        logging.getLogger(__name__).exception(
            "Error reindexando p.%s tras rerun (no crítico).", page.order,
        )

    return JsonResponse({'text': ocr_text, 'ocr_status': Page.OCR_DONE})


# ── Endpoint de estado para el frontend (polling) ────────────────────────

@require_GET
def document_ocr_status(request, doc_id):
    """
    GET → JSON con el estado OCR de todas las páginas del documento.

    Respuesta:
        {
          "doc_id": 42,
          "all_done": false,
          "thread_alive": true,
          "pages": [
            {"id": 1, "order": 1, "ocr_status": "done",       "ocr_error": ""},
            {"id": 2, "order": 2, "ocr_status": "processing", "ocr_error": ""},
            {"id": 3, "order": 3, "ocr_status": "pending",    "ocr_error": ""},
            ...
          ]
        }

    Lo usa el JS del edit_document para refrescar los thumbnails y
    recargar la vista cuando la página actual se complete.

    `thread_alive` indica si hay un thread vivo procesando este
    documento. Si hay páginas pendientes Y `thread_alive=False`, el
    thread se murió (por reinicio o crash) y conviene re-arrancar
    `ocr_process` para reanudar. Esto se hace AUTOMÁTICAMENTE aquí:
    si detectamos páginas pendientes sin thread, relanzamos. Así el
    usuario nunca se queda con un doc en "pending" para siempre, ni
    aunque el servidor se haya reiniciado entre medias.
    """
    doc = get_object_or_404(Document, pk=doc_id)
    if not doc.can_access(request.user):
        return JsonResponse({'error': 'Forbidden'}, status=403)

    from apps.ocr.tasks import is_document_processing, start_document_ocr  # noqa: PLC0415

    pages = list(doc.pages.order_by('order').values(
        'id', 'order', 'ocr_status', 'ocr_error',
    ))
    all_done = all(p['ocr_status'] == Page.OCR_DONE for p in pages)
    thread_alive = is_document_processing(doc_id)

    # Auto-recovery: páginas pendientes/procesando sin thread vivo →
    # el thread murió (reinicio, crash). Relanzamos transparentemente.
    has_unfinished = any(
        p['ocr_status'] in (Page.OCR_PENDING, Page.OCR_PROCESSING)
        for p in pages
    )
    if has_unfinished and not thread_alive:
        # Resucitar PROCESSING → PENDING (lo que estaba a medias se
        # rehace; no podemos asumir que terminó parcialmente).
        from django.db.models import Q  # noqa: PLC0415
        Page.objects.filter(
            document_id=doc_id,
            ocr_status=Page.OCR_PROCESSING,
        ).update(ocr_status=Page.OCR_PENDING, ocr_error='')
        start_document_ocr(doc_id)
        thread_alive = True

    return JsonResponse({
        'doc_id':       doc_id,
        'all_done':     all_done,
        'thread_alive': thread_alive,
        'pages':        pages,
    })


# ── Nuevas vistas: segmentación de líneas ────────────────────────────────

@worker_required
@require_GET
def line_segmentation_image(request, page_id):
    """Devuelve el JPG con las cajas de segmentación de líneas y bloques.

    Si la página todavía está siendo procesada por el OCR y la cache aún
    no existe, devuelve 503 + Retry-After para que el cliente reintente.

    Acepta ?force=1 para forzar la regeneración aunque exista cache.
    """
    page = get_object_or_404(Page, pk=page_id)
    if not request.user.is_worker_or_above:
        raise Http404

    force = request.GET.get('force') == '1'

    if page.ocr_status in (Page.OCR_PENDING, Page.OCR_PROCESSING) and not force:
        out_path = segmentation.get_or_generate(
            page.image.path, page.document_id, page.order,
            doc_type=page.document.document_type,
            allow_block=False,
        )
        if out_path is None or not out_path.is_file():
            response = HttpResponse(
                'Segmentación pendiente; reintenta en unos segundos.',
                status=503,
                content_type='text/plain; charset=utf-8',
            )
            response['Retry-After'] = '3'
            response['Cache-Control'] = 'no-store'
            return response
        response = FileResponse(open(out_path, 'rb'),
                                content_type='image/jpeg')
        response['Cache-Control'] = 'private, max-age=10'
        return response

    out_path = segmentation.get_or_generate(
        page.image.path, page.document_id, page.order,
        force=force,
        doc_type=page.document.document_type,
        allow_block=True,
    )
    if out_path is None or not out_path.is_file():
        return HttpResponse(
            'No se pudo generar la imagen de segmentación.',
            status=500,
            content_type='text/plain; charset=utf-8',
        )
    response = FileResponse(open(out_path, 'rb'), content_type='image/jpeg')
    response['Cache-Control'] = 'private, max-age=30'
    return response


@worker_required
@require_GET
def line_segmentation_boxes(request, page_id):
    """Devuelve el JSON de cajas de la última segmentación cacheada.

    Si la página está en proceso de OCR, devuelve 503 con Retry-After.
    """
    page = get_object_or_404(Page, pk=page_id)
    if not request.user.is_worker_or_above:
        return JsonResponse({'error': 'Forbidden'}, status=403)

    json_path = segmentation.boxes_json_path(page.document_id, page.order)
    if not json_path.is_file():
        allow_block = page.ocr_status not in (Page.OCR_PENDING, Page.OCR_PROCESSING)
        out = segmentation.get_or_generate(
            page.image.path, page.document_id, page.order,
            doc_type=page.document.document_type,
            allow_block=allow_block,
        )
        if (out is None or not json_path.is_file()) and not allow_block:
            response = JsonResponse(
                {'error': 'Segmentación pendiente, reintenta en unos segundos.'},
                status=503,
            )
            response['Retry-After'] = '3'
            return response
        if out is None or not json_path.is_file():
            return JsonResponse(
                {'error': 'No se pudo generar la segmentación.'},
                status=500,
            )

    return HttpResponse(
        json_path.read_text(encoding='utf-8'),
        content_type='application/json',
    )


# ── Nueva vista: OCR sobre regiones definidas por el usuario ─────────────

@worker_required
@require_POST
def ocr_regions_page(request, page_id):
    """
    AJAX: lee las regiones guardadas en el XML de la página y ejecuta
    OCR únicamente sobre esos recortes (en el orden indicado).

    El cliente debería haber guardado primero las regiones via
    POST /pages/<page_id>/regions/. Si la página no tiene regiones
    devolvemos 400.
    """
    page = get_object_or_404(Page, pk=page_id)
    if not request.user.is_worker_or_above:
        return JsonResponse({'error': 'Forbidden'}, status=403)

    region_objs = page.get_regions()
    if not region_objs:
        return JsonResponse(
            {'error': 'No hay regiones definidas para esta página.'},
            status=400,
        )

    doc_type = page.document.document_type
    use_bert = bool(page.document.use_bert_correction)
    try:
        text = ocr_regions(
            page.image.path,
            [r.to_dict() for r in region_objs],
            doc_type=doc_type,
            use_bert=use_bert,
        )
    except Exception as exc:
        return JsonResponse({'error': str(exc)}, status=500)

    page.text = text
    return JsonResponse({'text': text, 'n_regions': len(region_objs)})
