"""
apps/documents/region_views.py — Endpoints AJAX para la persistencia de
regiones definidas por el usuario en la pantalla de edición.

El cómputo OCR sobre esas regiones vive en apps/ocr/views.py
(endpoint ocr_regions). Aquí sólo guardamos/leemos la lista.
"""

from __future__ import annotations

import json
import uuid

from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_POST

from apps.accounts.decorators import worker_required
from .models import Page
from .transcripts import Region


@worker_required
@require_POST
def save_regions(request, page_id: int):
    """
    Guarda la lista de regiones del usuario para una página.

    Body JSON:
        {"regions": [
            {"id": "r1", "order": 1, "x": 120, "y": 80,
             "width": 900, "height": 220},
            ...
        ]}

    Coordenadas en píxeles del facsimilar ORIGINAL.
    Devuelve el listado normalizado (con ids garantizados y order saneado).
    """
    page = get_object_or_404(Page, pk=page_id)

    if not request.user.is_worker_or_above:
        return JsonResponse({'error': 'Forbidden'}, status=403)

    try:
        payload = json.loads(request.body or b'{}')
    except json.JSONDecodeError:
        return HttpResponseBadRequest('JSON inválido')

    raw = payload.get('regions') or []
    if not isinstance(raw, list):
        return HttpResponseBadRequest("Esperaba 'regions' como lista")

    regions = []
    for idx, item in enumerate(raw, start=1):
        try:
            x = max(0, int(item.get('x', 0)))
            y = max(0, int(item.get('y', 0)))
            w = max(1, int(item.get('width', 0)))
            h = max(1, int(item.get('height', 0)))
        except (TypeError, ValueError):
            continue
        rid = str(item.get('id') or '').strip() or f"r{uuid.uuid4().hex[:8]}"
        try:
            order = int(item.get('order', idx))
        except (TypeError, ValueError):
            order = idx
        regions.append(Region(
            id=rid, order=order, x=x, y=y, width=w, height=h,
        ))

    # Renumeramos `order` 1..N por si el cliente lo envió mal.
    regions.sort(key=lambda r: r.order)
    for i, r in enumerate(regions, start=1):
        r.order = i

    page.set_regions(regions)
    return JsonResponse({'regions': [r.to_dict() for r in regions]})
