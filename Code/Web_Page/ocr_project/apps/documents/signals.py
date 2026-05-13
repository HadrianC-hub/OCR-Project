"""
apps/documents/signals.py — Limpieza de ficheros al borrar entidades.

Django no borra automáticamente los ficheros de un FileField/ImageField al
borrar el modelo (lo hacía hasta la 1.2; lo retiraron por riesgo de
inconsistencias en transacciones revertidas). Aquí lo añadimos a mano,
junto con la limpieza de las transcripciones XML y las visualizaciones
cacheadas de la segmentación de líneas.

Flujo:
  - delete_page_files (post_delete sobre Page): borra el facsimilar,
    el transcript XML y la visualización cacheada de líneas.
  - cleanup_document_dirs (post_delete sobre Document): tras el CASCADE
    de las páginas, borra los directorios facsimiles/<id>/,
    transcripts/<id>/ y segmentation/<id>/ por si quedó algún huérfano.

Si un Page se borra individualmente (no por CASCADE), el handler de
post_delete sigue funcionando: sólo necesita document_id y order, que
están en el snapshot del objeto borrado.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from django.conf import settings
from django.db.models.signals import post_delete
from django.dispatch import receiver

from .models import Document, Page
from . import transcripts

logger = logging.getLogger(__name__)


# ── Page: borrar facsimilar + transcript + cache de segmentación ──────────────

@receiver(post_delete, sender=Page)
def delete_page_files(sender, instance: Page, **kwargs):
    # 1) Imagen del facsimilar
    if instance.image:
        try:
            instance.image.delete(save=False)
        except Exception as exc:
            logger.warning(
                "No se pudo borrar el facsimilar de Page %s: %s",
                instance.pk, exc,
            )

    # 2) Transcript XML
    try:
        transcripts.delete_page_transcript(instance.document_id, instance.order)
    except Exception as exc:
        logger.warning(
            "No se pudo borrar el transcript de Page %s (doc=%s, order=%s): %s",
            instance.pk, instance.document_id, instance.order, exc,
        )

    # 3) Visualización cacheada de la segmentación de líneas
    try:
        seg_dir = Path(settings.MEDIA_ROOT) / "segmentation" / str(instance.document_id)
        for fname in (f"page_{instance.order:03d}_lines.jpg",
                      f"page_{instance.order:03d}_lines.json"):
            f = seg_dir / fname
            if f.is_file():
                f.unlink()
    except Exception as exc:
        logger.warning(
            "No se pudo borrar la visualización de Page %s: %s",
            instance.pk, exc,
        )


# ── Document: limpiar directorios del documento ───────────────────────────────

@receiver(post_delete, sender=Document)
def cleanup_document_dirs(sender, instance: Document, **kwargs):
    """
    Tras el CASCADE de las páginas (y sus handlers), eliminamos los
    directorios del documento. Borramos el directorio entero (más
    robusto que comprobar si está vacío, porque podrían quedar ficheros
    huérfanos de procesos previos).
    """
    doc_id = instance.pk
    media_root = Path(settings.MEDIA_ROOT)

    for sub in ("facsimiles", "transcripts", "segmentation"):
        d = media_root / sub / str(doc_id)
        if d.is_dir():
            try:
                shutil.rmtree(d)
            except Exception as exc:
                logger.warning(
                    "No se pudo eliminar %s tras borrar Document %s: %s",
                    d, doc_id, exc,
                )
