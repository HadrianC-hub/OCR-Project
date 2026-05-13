"""
apps/documents/transcripts.py — Lectura/escritura de transcripciones en XML.

El contenido transcrito de cada página se guarda en
    {MEDIA_ROOT}/transcripts/{document_id}/page_{order:03d}.xml
en un formato XML simple y legible (ver docs/transcript_format.md).

Esta capa es la *única* que toca el sistema de ficheros para transcripciones.
El resto del proyecto interactúa siempre a través de los métodos de Page
(get_text, set_text, save_regions, etc.).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional
from xml.dom import minidom
from xml.etree import ElementTree as ET

from django.conf import settings


# ── Constantes ────────────────────────────────────────────────────────────────

TRANSCRIPT_VERSION  = "1"
TRANSCRIPT_DIR_NAME = "transcripts"


# ── Modelos ligeros ───────────────────────────────────────────────────────────

@dataclass
class Region:
    """
    Caja de texto definida por el usuario en la pantalla de edición.

    Coordenadas en píxeles del *facsimilar original* (no de la versión
    deskewed). El pipeline de OCR aplicará deskew local al recorte.
    """
    id:     str
    order:  int
    x:      int
    y:      int
    width:  int
    height: int

    def to_dict(self) -> dict:
        return {
            "id": self.id, "order": self.order,
            "x": self.x, "y": self.y,
            "width": self.width, "height": self.height,
        }


@dataclass
class TranscriptData:
    """Estructura en memoria de un fichero de transcripción."""
    document_id:    int
    page_order:     int
    facsimile:      str = ""
    title:          str = ""
    author:         str = ""
    year:           Optional[int] = None
    doc_type:       str = ""
    lines:          List[str] = field(default_factory=list)   # texto, una entrada por línea
    regions:        List[Region] = field(default_factory=list)
    created_at:     Optional[str] = None  # ISO 8601
    modified_at:    Optional[str] = None  # ISO 8601


# ── Rutas ─────────────────────────────────────────────────────────────────────

def transcript_dir(document_id: int) -> Path:
    return Path(settings.MEDIA_ROOT) / TRANSCRIPT_DIR_NAME / str(document_id)


def transcript_path(document_id: int, page_order: int) -> Path:
    return transcript_dir(document_id) / f"page_{page_order:03d}.xml"


# ── Lectura ───────────────────────────────────────────────────────────────────

def load(document_id: int, page_order: int) -> Optional[TranscriptData]:
    """
    Devuelve el TranscriptData del fichero correspondiente, o None si
    aún no existe en disco. Nunca lanza excepciones por XML malformado;
    en ese caso emite un warning y devuelve None.
    """
    path = transcript_path(document_id, page_order)
    if not path.is_file():
        return None

    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Transcript corrupto en '%s': %s. Se ignora.", path, exc,
        )
        return None

    root = tree.getroot()

    data = TranscriptData(
        document_id=document_id,
        page_order=page_order,
    )

    # Document metadata
    doc_el = root.find("Document")
    if doc_el is not None:
        data.title    = (doc_el.findtext("Title")  or "").strip()
        data.author   = (doc_el.findtext("Author") or "").strip()
        data.doc_type = (doc_el.findtext("Type")   or "").strip()
        year_txt      = (doc_el.findtext("Year")   or "").strip()
        if year_txt.isdigit():
            data.year = int(year_txt)

    # Page reference
    page_el = root.find("Page")
    if page_el is not None:
        data.facsimile = page_el.get("facsimile", "")

    # Timestamps
    ts_el = root.find("Timestamps")
    if ts_el is not None:
        data.created_at  = (ts_el.findtext("Created")  or "").strip() or None
        data.modified_at = (ts_el.findtext("Modified") or "").strip() or None

    # Regions
    regions_el = root.find("Regions")
    if regions_el is not None:
        for r in regions_el.findall("Region"):
            try:
                data.regions.append(Region(
                    id     = r.get("id", ""),
                    order  = int(r.get("order", "0")),
                    x      = int(r.get("x", "0")),
                    y      = int(r.get("y", "0")),
                    width  = int(r.get("width", "0")),
                    height = int(r.get("height", "0")),
                ))
            except (TypeError, ValueError):
                continue
        data.regions.sort(key=lambda r: r.order)

    # Body / lines
    body_el = root.find("Body")
    if body_el is not None:
        data.lines = [(line_el.text or "") for line_el in body_el.findall("Line")]

    return data


def get_text(document_id: int, page_order: int) -> str:
    """Devuelve el texto plano (líneas unidas con '\\n'). '' si no hay fichero."""
    data = load(document_id, page_order)
    if data is None:
        return ""
    return "\n".join(data.lines)


def get_regions(document_id: int, page_order: int) -> List[Region]:
    """Devuelve las regiones definidas (vacío si no hay fichero o no hay regiones)."""
    data = load(document_id, page_order)
    if data is None:
        return []
    return data.regions


# ── Escritura ─────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _build_xml(data: TranscriptData) -> bytes:
    """Serializa un TranscriptData a XML pretty-printed UTF-8."""
    root = ET.Element("Transcript", attrib={"version": TRANSCRIPT_VERSION})

    doc_el = ET.SubElement(root, "Document", attrib={"id": str(data.document_id)})
    ET.SubElement(doc_el, "Title").text  = data.title
    ET.SubElement(doc_el, "Author").text = data.author
    if data.year is not None:
        ET.SubElement(doc_el, "Year").text = str(data.year)
    ET.SubElement(doc_el, "Type").text   = data.doc_type

    ET.SubElement(root, "Page", attrib={
        "order":     str(data.page_order),
        "facsimile": data.facsimile,
    })

    ts_el = ET.SubElement(root, "Timestamps")
    ET.SubElement(ts_el, "Created").text  = data.created_at  or _now_iso()
    ET.SubElement(ts_el, "Modified").text = data.modified_at or _now_iso()

    if data.regions:
        regions_el = ET.SubElement(root, "Regions")
        for r in sorted(data.regions, key=lambda x: x.order):
            ET.SubElement(regions_el, "Region", attrib={
                "id":     r.id,
                "order":  str(r.order),
                "x":      str(r.x),
                "y":      str(r.y),
                "width":  str(r.width),
                "height": str(r.height),
            })

    body_el = ET.SubElement(root, "Body")
    for line in data.lines:
        ln = ET.SubElement(body_el, "Line")
        ln.text = line if line else ""

    raw    = ET.tostring(root, encoding="utf-8")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ", encoding="utf-8")
    return pretty


def save(data: TranscriptData) -> Path:
    """
    Escribe el transcript a disco. Crea directorios intermedios si hace falta.
    Si ya existe, conserva created_at y actualiza modified_at.
    Escritura atómica: escribe a .tmp y renombra.
    """
    path = transcript_path(data.document_id, data.page_order)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = load(data.document_id, data.page_order)
    if existing and existing.created_at and not data.created_at:
        data.created_at = existing.created_at
    if not data.created_at:
        data.created_at = _now_iso()
    data.modified_at = _now_iso()

    tmp_path = path.with_suffix(".xml.tmp")
    tmp_path.write_bytes(_build_xml(data))
    os.replace(tmp_path, path)
    return path


def save_text(
    document_id: int,
    page_order:  int,
    text:        str,
    *,
    facsimile:   str = "",
    title:       str = "",
    author:      str = "",
    year:        Optional[int] = None,
    doc_type:    str = "",
    regions:     Optional[Iterable[Region]] = None,
) -> Path:
    """
    Atajo: actualiza el texto de una página conservando regiones existentes
    salvo que se pasen explícitamente.
    """
    existing = load(document_id, page_order)
    data = TranscriptData(
        document_id=document_id,
        page_order=page_order,
        facsimile=facsimile or (existing.facsimile if existing else ""),
        title=title or (existing.title if existing else ""),
        author=author or (existing.author if existing else ""),
        year=year if year is not None else (existing.year if existing else None),
        doc_type=doc_type or (existing.doc_type if existing else ""),
        lines=text.splitlines(),
        regions=list(regions) if regions is not None else (existing.regions if existing else []),
        created_at=(existing.created_at if existing else None),
    )
    return save(data)


def save_regions(
    document_id: int,
    page_order:  int,
    regions:     Iterable[Region],
) -> Path:
    """Atajo: actualiza únicamente las regiones, conservando el texto."""
    existing = load(document_id, page_order) or TranscriptData(
        document_id=document_id, page_order=page_order,
    )
    existing.regions = list(regions)
    return save(existing)


# ── Borrado ───────────────────────────────────────────────────────────────────

def delete_page_transcript(document_id: int, page_order: int) -> bool:
    """Borra el fichero XML de una página. Devuelve True si se borró algo."""
    path = transcript_path(document_id, page_order)
    if path.is_file():
        try:
            path.unlink()
            return True
        except OSError:
            pass
    return False


def delete_document_transcripts(document_id: int) -> bool:
    """Borra el directorio entero de transcripciones del documento."""
    d = transcript_dir(document_id)
    if d.is_dir():
        try:
            shutil.rmtree(d)
            return True
        except OSError:
            pass
    return False
