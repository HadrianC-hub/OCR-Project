"""
Migración 0003: traslada el contenido de Page.text a ficheros XML
y elimina la columna text de la tabla Page.

Pasos:
  1) RunPython.export_text_to_files
        Para cada Page con text no vacío, escribe el XML correspondiente
        en {MEDIA_ROOT}/transcripts/<doc_id>/page_<order:03d>.xml.
        Idempotente: si el XML ya existe lo respeta.
  2) RemoveField Page.text
        Elimina la columna de la base de datos.

Reverso (manage.py migrate documents 0002):
  - Re-crea la columna text vacía. Los ficheros XML permanecen en
    media/transcripts/, así que NO se pierden datos por revertir.
    Para rehidratar el campo desde los XML, ejecuta después:
        python manage.py shell -c "from apps.documents.models import Page; \\
            from apps.documents.transcripts import get_text; \\
            [setattr(p, '_text_cache', get_text(p.document_id, p.order)) \\
             or Page.objects.filter(pk=p.pk).update(text=p._text_cache) \\
             for p in Page.objects.all()]"
    (Ver también docs/transcript_format.md.)
"""

import os
from django.conf import settings
from django.db import migrations
from pathlib import Path


# ── Helpers (auto-contenidos: la migración no debe depender de que
#    apps.documents.transcripts siga existiendo o no cambie en el futuro) ──

def _build_xml_bytes(*, doc_id, page_order, facsimile, title, author, year,
                     doc_type, lines):
    from datetime import datetime, timezone
    from xml.etree import ElementTree as ET
    from xml.dom import minidom

    root = ET.Element("Transcript", attrib={"version": "1"})

    doc_el = ET.SubElement(root, "Document", attrib={"id": str(doc_id)})
    ET.SubElement(doc_el, "Title").text  = title or ""
    ET.SubElement(doc_el, "Author").text = author or ""
    if year is not None:
        ET.SubElement(doc_el, "Year").text = str(year)
    ET.SubElement(doc_el, "Type").text   = doc_type or ""

    ET.SubElement(root, "Page", attrib={
        "order":     str(page_order),
        "facsimile": facsimile or "",
    })

    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    ts_el = ET.SubElement(root, "Timestamps")
    ET.SubElement(ts_el, "Created").text  = now
    ET.SubElement(ts_el, "Modified").text = now

    body_el = ET.SubElement(root, "Body")
    for line in lines:
        ln = ET.SubElement(body_el, "Line")
        ln.text = line if line else ""

    raw = ET.tostring(root, encoding="utf-8")
    return minidom.parseString(raw).toprettyxml(indent="  ", encoding="utf-8")


def _transcript_path(doc_id, page_order):
    return (Path(settings.MEDIA_ROOT) / "transcripts" / str(doc_id)
            / f"page_{page_order:03d}.xml")


# ── Forward: text → ficheros XML ──────────────────────────────────────────

def export_text_to_files(apps, schema_editor):
    Page = apps.get_model("documents", "Page")
    n_total = n_written = n_skipped = 0
    for page in Page.objects.select_related("document").iterator():
        n_total += 1
        text = (page.text or "").strip()
        if not text:
            continue

        path = _transcript_path(page.document_id, page.order)
        if path.exists():
            n_skipped += 1
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        facsimile = ""
        try:
            if page.image and page.image.name:
                facsimile = page.image.name.split("/")[-1]
        except Exception:
            pass

        xml_bytes = _build_xml_bytes(
            doc_id     = page.document_id,
            page_order = page.order,
            facsimile  = facsimile,
            title      = page.document.title,
            author     = page.document.author,
            year       = page.document.year,
            doc_type   = page.document.document_type,
            lines      = (page.text or "").splitlines(),
        )
        path.write_bytes(xml_bytes)
        n_written += 1

    print(f"\n  [migration 0003] Páginas procesadas: {n_total} | "
          f"XML escritos: {n_written} | ya existentes (omitidos): {n_skipped}")


# ── Migración ────────────────────────────────────────────────────────────

class Migration(migrations.Migration):

    dependencies = [
        ('documents', '0002_alter_operationlog_action'),
    ]

    operations = [
        # Paso 1: copiar el texto a ficheros XML.
        # Reverso: noop. Los XML ya existen y no se borran al revertir,
        # así que no se pierden datos. Para rehidratar el campo text
        # tras revertir, ver el docstring de este módulo.
        migrations.RunPython(
            export_text_to_files,
            reverse_code=migrations.RunPython.noop,
        ),
        # Paso 2: eliminar la columna text.
        # Al revertir, Django re-crea el campo vacío.
        migrations.RemoveField(
            model_name='page',
            name='text',
        ),
    ]
