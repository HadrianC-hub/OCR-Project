from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import Http404, JsonResponse
from django.db.models import Q
from django.urls import reverse
from django.views.decorators.http import require_POST

from apps.accounts.decorators import worker_required
from .models import Document, Page, OperationLog
from .forms import DocumentMetadataForm, SearchForm
from .utils import generate_epub, generate_pdf
from . import transcripts


# ── Helpers ────────────────────────────────────────────────────────────────

def _log(user, action, description):
    if user and user.is_authenticated:
        OperationLog.objects.create(user=user, action=action, description=description)


def _base_qs(user):
    """Return documents accessible by the given user."""
    qs = Document.objects.prefetch_related('pages')
    if not (user and user.is_authenticated and user.is_worker_or_above):
        qs = qs.filter(is_public=True)
    return qs


def _doc_matches_text(doc, query_lower: str) -> bool:
    """
    Devuelve True si el texto transcrito de cualquier página de `doc`
    contiene `query_lower` (búsqueda case-insensitive).

    """
    for page in doc.pages.all():
        if query_lower in page.get_text().lower():
            return True
    return False


# ── Home / Presentation ────────────────────────────────────────────────────

def home(request):
    return render(request, 'documents/home.html')


# ── Search ─────────────────────────────────────────────────────────────────

def search(request):
    """
    Búsqueda con dos modos seleccionables:

      mode='metadata' (default): SQL directo sobre title/author/description.
          Sin embeddings ni IO de disco. Instantáneo (~1 ms).

      mode='content': búsqueda semántica sobre el texto OCR de las
          páginas via embeddings E5 + re-ranking híbrido con BM25.
          Devuelve documentos + página exacta donde matcheó.

    Los filtros adicionales (año, tipo, autor) se aplican en ambos modos.
    """
    form     = SearchForm(request.GET or None)
    results  = []
    page_hits = {}   # doc_id -> [{page_order, score, snippet}, ...]
    searched = False
    mode     = request.GET.get('mode', 'metadata')
    if mode not in ('metadata', 'content'):
        mode = 'metadata'

    if request.GET:
        searched = True
        qs = _base_qs(request.user)

        q = request.GET.get('q', '').strip()
        year_from = request.GET.get('year_from')
        year_to   = request.GET.get('year_to')
        doc_type  = request.GET.get('document_type')
        author    = request.GET.get('author', '').strip()

        # Filtros adicionales (igual en ambos modos)
        if year_from:
            qs = qs.filter(year__gte=year_from)
        if year_to:
            qs = qs.filter(year__lte=year_to)
        if doc_type:
            qs = qs.filter(document_type=doc_type)
        if author:
            qs = qs.filter(author__icontains=author)

        if q:
            if mode == 'metadata':
                # Búsqueda directa en SQL. No tocamos disco. Rápido.
                qs = qs.filter(
                    Q(title__icontains=q) |
                    Q(author__icontains=q) |
                    Q(description__icontains=q)
                )
            else:
                # Búsqueda semántica + híbrida sobre contenido OCR
                try:
                    from apps.search.service import get_store
                    from apps.search import encoder, hybrid

                    if not encoder.is_available():
                        messages.warning(
                            request,
                            "La búsqueda semántica no está disponible "
                            "(falta el modelo E5). Cambia a búsqueda por "
                            "metadatos o instálalo (ver INSTALL_SEARCH.md)."
                        )
                        qs = qs.none()
                    else:
                        store = get_store()
                        # E5 semántico
                        q_emb = encoder.encode_query(q)
                        sem_results = store.search(q_emb, top_k=50,
                                                   max_per_doc=3) if q_emb is not None else []
                        # BM25 sobre los mismos chunks
                        bm25_results = hybrid.search_bm25(q, store, top_k=50)
                        # Fusión RRF
                        merged = hybrid.rrf_merge(sem_results, bm25_results, top_k=30)

                        # Aplicamos el control de acceso ANTES de poblar
                        # page_hits — si no, el dict acabaría conteniendo
                        # snippets de documentos privados que el usuario
                        # no debería ver. Hoy el template sólo accede a
                        # page_hits[doc.id] para docs ya filtrados, así
                        # que el leak no es visible; pero cualquier
                        # endpoint JSON o iteración directa del contexto
                        # lo expondría.
                        candidate_ids: list[int] = []
                        seen_ids: set[int] = set()
                        for r in merged:
                            did = r['doc_id']
                            if did not in seen_ids:
                                seen_ids.add(did)
                                candidate_ids.append(did)

                        qs = qs.filter(id__in=candidate_ids)
                        docs_by_id = {d.id: d for d in qs}
                        ordered_ids = [i for i in candidate_ids if i in docs_by_id]

                        # Ahora sí, poblar page_hits sólo para docs accesibles
                        for r in merged:
                            did = r['doc_id']
                            if did not in docs_by_id:
                                continue
                            page_hits.setdefault(did, []).append({
                                'page_order': r['page_order'],
                                'snippet':    r['snippet'],
                                'score':      round(r['score_rrf'], 4),
                                'score_sem':  r.get('score_semantic'),
                                'score_bm25': r.get('score_bm25'),
                            })
                        # Devolver lista ordenada (no QuerySet) — para preservar
                        # el orden por score, que un QuerySet no garantiza.
                        results = [docs_by_id[i] for i in ordered_ids]

                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).exception(
                        "Error en búsqueda semántica."
                    )
                    messages.error(request,
                                   f"Error en búsqueda semántica: {exc}")
                    qs = qs.none()

        if mode == 'metadata' or not q:
            # En metadatos siempre, y en content si no hay q (sólo filtros),
            # el QuerySet final es el queryset filtrado.
            results = list(qs.distinct())

    return render(request, 'documents/search.html', {
        'form':      form,
        'results':   results,
        'page_hits': page_hits,
        'searched':  searched,
        'mode':      mode,
    })


# ── Visualisation ──────────────────────────────────────────────────────────

def view_document(request, doc_id):
    doc = get_object_or_404(Document, pk=doc_id)

    if not doc.can_access(request.user):
        raise Http404

    pages = list(doc.pages.all())
    if not pages:
        messages.warning(request, 'Este documento no tiene páginas.')

    try:
        page_num = max(1, int(request.GET.get('page', 1)))
    except (ValueError, TypeError):
        page_num = 1

    page_num     = min(page_num, len(pages)) if pages else 1
    current_page = pages[page_num - 1] if pages else None

    Document.objects.filter(pk=doc_id).update(total_views=doc.total_views + 1)
    _log(request.user, OperationLog.VIEW_DOC, f'Visualizó "{doc.title}"')

    return render(request, 'documents/view_document.html', {
        'doc':          doc,
        'pages':        pages,
        'current_page': current_page,
        'page_num':     page_num,
        'total_pages':  len(pages),
    })


# ── Download ───────────────────────────────────────────────────────────────

def download_document(request, doc_id):
    doc = get_object_or_404(Document, pk=doc_id)

    if not doc.can_access(request.user):
        raise Http404

    fmt = request.GET.get('format', 'pdf').lower()
    include_facsimile = request.GET.get('facsimile') == '1'

    if fmt == 'epub':
        response = generate_epub(doc, include_facsimile=include_facsimile)
    else:
        response = generate_pdf(doc, include_facsimile=include_facsimile)

    if response is None:
        messages.error(request, 'Error generando el archivo. Contacta al administrador.')
        return redirect('view_document', doc_id=doc_id)

    Document.objects.filter(pk=doc_id).update(total_downloads=doc.total_downloads + 1)
    suffix = ' (con facsimilar)' if include_facsimile else ''
    _log(request.user, OperationLog.DOWNLOAD_DOC,
         f'Descargó "{doc.title}" en formato {fmt.upper()}{suffix}')
    return response


# ── Insert ─────────────────────────────────────────────────────────────────

@worker_required
def insert_document(request):
    if request.method == 'POST':
        meta_form = DocumentMetadataForm(request.POST)
        images    = request.FILES.getlist('images')
        order_str = request.POST.get('order', '')

        if not images:
            messages.error(request, 'Debes subir al menos una imagen.')
            return render(request, 'documents/insert_document.html', {'form': meta_form})

        if not meta_form.is_valid():
            messages.error(request, 'Corrige los errores del formulario.')
            return render(request, 'documents/insert_document.html', {'form': meta_form})

        doc = meta_form.save(commit=False)
        doc.created_by = request.user
        doc.save()

        # Reorder images according to user-specified order (JS sends indices)
        try:
            order_indices = [int(i) for i in order_str.split(',') if i.strip().isdigit()]
            if len(order_indices) == len(images):
                ordered_images = [images[i] for i in order_indices]
            else:
                ordered_images = images
        except Exception:
            ordered_images = images

        for idx, img in enumerate(ordered_images, start=1):
            Page.objects.create(
                document=doc, order=idx, image=img,
                ocr_status=Page.OCR_PENDING,
            )

        Document.objects.filter(pk=doc.pk).update(total_edits=doc.total_edits + 1)
        _log(request.user, OperationLog.INSERT_DOC,
             f'{request.user.username} insertó el documento "{doc.title}"')
        messages.success(request, f'Documento "{doc.title}" insertado. Revisa el texto del OCR.')

        return redirect('ocr_process', doc_id=doc.id)

    else:
        form = DocumentMetadataForm()
    return render(request, 'documents/insert_document.html', {'form': form})


# ── Edit ───────────────────────────────────────────────────────────────────

@worker_required
def edit_document(request, doc_id):
    doc       = get_object_or_404(Document, pk=doc_id)
    pages     = list(doc.pages.all())
    meta_form = DocumentMetadataForm(request.POST or None, instance=doc)

    try:
        page_num = max(1, int(request.GET.get('page', 1)))
    except (ValueError, TypeError):
        page_num = 1

    page_num     = min(page_num, len(pages)) if pages else 1
    current_page = pages[page_num - 1] if pages else None

    # Detección de estado "en proceso": hay alguna página pendiente o
    # corriendo. Mientras esto sea True, el guardado está bloqueado y
    # el frontend pide confirmación si el usuario intenta navegar fuera.
    from apps.documents.models import Page  # noqa: PLC0415
    is_processing = any(
        p.ocr_status in (Page.OCR_PENDING, Page.OCR_PROCESSING, Page.OCR_ERROR)
        for p in pages
    )

    if request.method == 'POST':
        # Si está en proceso, NO guardamos nada: solo se permite la
        # navegación entre páginas mediante el botón _nav. Esto evita
        # sobrescribir el texto OCR generado por el thread con datos
        # vacíos del formulario.
        nav_page = request.POST.get('_nav')

        if is_processing:
            if not nav_page:
                messages.warning(
                    request,
                    'No se puede guardar mientras el OCR está en proceso. '
                    'Espera a que termine o abandona el documento.',
                )
            target = nav_page if nav_page else request.GET.get('page', 1)
            base_url = reverse('edit_document', args=[doc_id])
            return redirect(f'{base_url}?page={target}')

        # Flujo normal (sin processing): guardar texto y metadatos.
        # Detectamos qué páginas cambiaron de texto para reindexar solo
        # esas en el store semántico al final (evita reindexar páginas
        # intactas y mantiene la búsqueda BM25/E5 sincronizada con el XML).
        modified_pages = []
        for page in pages:
            key = f'text_page_{page.id}'
            if key in request.POST:
                new_text = request.POST[key]
                if new_text != page.text:
                    page.text = new_text
                    modified_pages.append(page)

        if meta_form.is_valid():
            updated_doc = meta_form.save(commit=False)
            updated_doc.last_modified_by = request.user
            updated_doc.save()
            Document.objects.filter(pk=doc.pk).update(total_edits=doc.total_edits + 1)
            _log(request.user, OperationLog.EDIT_DOC,
                 f'{request.user.username} editó el documento "{doc.title}"')
            if not nav_page:
                messages.success(request, 'Cambios guardados correctamente.')
        else:
            if not nav_page:
                messages.error(request, 'Corrige los errores de los metadatos.')

        # Reindexar las páginas modificadas en el store semántico. Se hace
        # después de guardar el texto (set_text ya escribió al XML que
        # index_page lee vía page.get_text()). Una sola llamada save por
        # página es aceptable porque el flujo de edición no es masivo.
        # Errores aquí son no-críticos: el texto queda guardado aunque
        # falle la reindexación; el siguiente reindex_all corrige.
        if modified_pages:
            try:
                from apps.search.indexer import index_page  # noqa: PLC0415
                for p in modified_pages:
                    index_page(p)
            except Exception:
                import logging
                logging.getLogger(__name__).exception(
                    "Error reindexando páginas editadas del doc %s "
                    "(no crítico, el texto está guardado).", doc_id,
                )

        target = nav_page if nav_page else request.GET.get('page', 1)
        base_url = reverse('edit_document', args=[doc_id])
        return redirect(f'{base_url}?page={target}')

    # Para el panel de regiones del frontend, exponemos las regiones
    # actuales serializadas como JSON (se embebe en un atributo data-).
    import json as _json
    current_regions_json = "[]"
    if current_page:
        current_regions_json = _json.dumps(
            [r.to_dict() for r in current_page.get_regions()],
            ensure_ascii=False,
        )

    return render(request, 'documents/edit_document.html', {
        'doc':                  doc,
        'pages':                pages,
        'current_page':         current_page,
        'page_num':             page_num,
        'total_pages':          len(pages),
        'meta_form':            meta_form,
        'current_regions_json': current_regions_json,
        'is_processing':        is_processing,
        'is_first_time':        doc.total_edits == 0,
    })


# ── Abandono de documento en proceso ───────────────────────────────────────

@worker_required
@require_POST
def abandon_document(request, doc_id):
    """
    Elimina un documento que está siendo procesado (OCR pendiente).

    Endpoint solo POST llamado desde el modal del edit_document cuando el
    usuario confirma "salir y eliminar" mientras hay páginas en cola, o
    desde sendBeacon al cerrar la pestaña. Solo borra si el documento
    está efectivamente en proceso — para borrados de documentos completos
    existe la vista normal `delete_document`.
    """
    doc = get_object_or_404(Document, pk=doc_id)

    if not (request.user.is_authenticated and request.user.is_worker_or_above):
        return JsonResponse({'error': 'Forbidden'}, status=403)

    from apps.documents.models import Page  # noqa: PLC0415
    is_processing = doc.pages.filter(
        ocr_status__in=(Page.OCR_PENDING, Page.OCR_PROCESSING, Page.OCR_ERROR)
    ).exists()

    if not is_processing:
        # Seguridad: no permitimos abandonar (=borrar) documentos
        # completos por error; para esos hay `delete_document`.
        return JsonResponse({
            'error': 'El documento no está en proceso; usa la vista de '
                     'eliminación normal.',
        }, status=400)

    # ── Decisión: ¿borrar el doc o solo dejar salir al usuario? ──────────
    #
    # `total_edits == 0`: el usuario nunca ha guardado nada vía form de
    #   edición. Es la primera vez que entra al doc después del upload.
    #   El doc "no tiene estado anterior" — si se va, no hay nada que
    #   preservar, lo borramos.
    #
    # `total_edits > 0`: ya hay al menos un save previo. El doc tiene una
    #   versión anterior que el usuario querrá mantener. NO borramos.
    #   Solo confirmamos que sale; el OCR sigue corriendo en background
    #   y, al volver, encontrará las páginas en el estado en que
    #   terminaron.
    is_first_time = (doc.total_edits == 0)

    if is_first_time:
        title = doc.title

        # Avisamos al thread del OCR ANTES de borrar el doc. El thread
        # consultará el flag en la siguiente iteración y saldrá del bucle,
        # liberando CPU. La página que tenga entre manos ahora mismo terminará
        # (no podemos matar al subproceso de Tesseract), pero al menos el doc
        # no seguirá ocupando recursos en sus páginas restantes.
        try:
            from apps.ocr.tasks import cancel_document_ocr  # noqa: PLC0415
            cancel_document_ocr(doc_id)
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "Error cancelando thread OCR para doc %s (no crítico).", doc_id,
            )

        doc.delete()  # CASCADE → pages → signals → ficheros del disco

        # Quitar del índice semántico (no crítico si falla)
        try:
            from apps.search.indexer import remove_document as _idx_remove
            _idx_remove(doc_id)
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "Error quitando doc %s del índice tras abandono.", doc_id,
            )

        _log(request.user, OperationLog.DELETE_DOC,
             f'{request.user.username} abandonó (eliminó) el documento '
             f'"{title}" en proceso')

        return JsonResponse({
            'ok':      True,
            'deleted': True,
            'message': f'Documento "{title}" eliminado.',
        })

    # Doc con histórico: solo confirmamos la salida. NO cancelamos el OCR
    # (el usuario querrá que las páginas pendientes terminen). NO borramos.
    # Las ediciones no guardadas en buffer del navegador se pierden, pero
    # eso es lo esperado al "salir sin guardar".
    return JsonResponse({
        'ok':      True,
        'deleted': False,
        'message': 'Saliendo sin guardar. El OCR sigue en segundo plano.',
    })


# ── Delete ─────────────────────────────────────────────────────────────────

@worker_required
def delete_document(request, doc_id):
    """
    El borrado de ficheros (facsimiles, XML, segmentación) lo gestionan
    los signals post_delete (apps/documents/signals.py). Aquí sólo
    disparamos el delete y registramos la operación.
    """
    doc = get_object_or_404(Document, pk=doc_id)

    if request.method == 'POST':
        title = doc.title
        doc.delete()  # CASCADE → Page → signals borran ficheros del disco

        # Quitar los chunks de este documento del índice semántico
        try:
            from apps.search.indexer import remove_document as _remove_from_index
            _remove_from_index(doc_id)
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "Error quitando doc %s del índice (no crítico).", doc_id,
            )

        _log(request.user, OperationLog.DELETE_DOC,
             f'{request.user.username} eliminó el documento "{title}"')
        messages.success(request, f'Documento "{title}" eliminado correctamente.')
        return redirect('home')

    return render(request, 'documents/confirm_delete_document.html', {'doc': doc})
