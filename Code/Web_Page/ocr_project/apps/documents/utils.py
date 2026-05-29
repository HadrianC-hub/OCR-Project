"""
apps/documents/utils.py — Generación de PDF y EPUB descargables.

Ambos exports comparten la misma cadena de procesado de texto:

    OCR → reflow_paragraphs → smart_typography → add_soft_hyphens
                                                ↓
                                         render(PDF | EPUB)

PDF: maquetación tipo libro (B5, márgenes asimétricos, folio en pie,
encabezado vivo con título, cubierta + portadilla + TOC + colofón,
cuerpo en serif Times-Roman, sangría francesa, justificado).

EPUB: hoja de estilos serif con justificado y sangría, capitular en
inicio de capítulo, metadatos enriquecidos, cubierta como imagen.

Ambos soportan modo facsímil (include_facsimile=True): cada página
incluye primero la imagen original y después la transcripción.
"""

from __future__ import annotations

import io
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from django.http import HttpResponse

from . import typography

logger = logging.getLogger(__name__)


# Cadena estándar para procesar el texto OCR antes de componerlo.
def _process_text(text: str, lang: str = "es", *,
                  hyphenate: bool = True) -> List[str]:
    """OCR → párrafos limpios listos para componer (une líneas contiguas)."""
    paragraphs = typography.reflow_paragraphs(text or "")
    paragraphs = [typography.smart_typography(p, lang=lang) for p in paragraphs]
    if hyphenate:
        paragraphs = [typography.add_soft_hyphens(p, lang=lang) for p in paragraphs]
    return paragraphs


def _process_blocks(text: str, lang: str = "es", *,
                    hyphenate: bool = True) -> List[List[str]]:
    """
    OCR → bloques de líneas preservando saltos dentro del bloque.

    Devuelve List[List[str]] donde cada sub-lista es un bloque separado
    por línea en blanco.  Los saltos de línea simples se conservan como
    líneas distintas, lo que permite renderizar correctamente verso y
    poesía tanto en EPUB como en PDF.
    """
    blocks = typography.split_blocks(text or "")
    result = []
    for block in blocks:
        processed = [typography.smart_typography(line, lang=lang) for line in block]
        if hyphenate:
            processed = [typography.add_soft_hyphens(line, lang=lang) for line in processed]
        result.append(processed)
    return result


# ─────────────────────────────────────────────────────────────────────────
#                                EPUB
# ─────────────────────────────────────────────────────────────────────────

def generate_epub(document, include_facsimile: bool = False):
    """
    Genera un EPUB con tipografía cuidada.

    Args:
        document: instancia de Document.
        include_facsimile: si es True, cada página incluye la imagen
                           original antes de la transcripción.
    """
    try:
        from ebooklib import epub
    except ImportError:
        logger.error("ebooklib no está instalado")
        return None

    book = epub.EpubBook()
    book.set_identifier(f"archivoocr-{document.id}")
    book.set_title(document.title or "Documento")
    book.set_language("es")

    if document.author:
        book.add_author(document.author)
    book.add_metadata("DC", "publisher", "ArchivoOCR")
    book.add_metadata(
        "DC", "date",
        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        {"event": "publication"},
    )
    if document.year:
        book.add_metadata("DC", "date", f"{document.year}", {"event": "original-publication"})
    if document.description:
        book.add_metadata("DC", "description", document.description)
    book.add_metadata(
        "DC", "rights",
        "Transcripción generada mediante reconocimiento óptico de caracteres.",
    )
    book.add_metadata(
        "DC", "contributor",
        "Transcripción OCR — ArchivoOCR",
        {"role": "trc"},
    )

    # CSS principal
    css_item = epub.EpubItem(
        uid="style_main", file_name="style/main.css",
        media_type="text/css", content=_epub_css(),
    )
    book.add_item(css_item)

    # Cubierta — usamos el primer facsimilar normalizado a JPG
    pages = list(document.pages.all())
    if pages and pages[0].image:
        cover_data = _normalize_cover_image(pages[0].image.path)
        if cover_data:
            book.set_cover("cover.jpg", cover_data)

    # Página de portada con texto (separada de la imagen)
    cover_text = epub.EpubHtml(
        title="Portada", file_name="title.xhtml", lang="es",
    )
    cover_text.add_item(css_item)
    cover_text.content = _epub_cover_xhtml(document)
    book.add_item(cover_text)

    # Capítulos — uno por página
    chapters = []
    for page in pages:
        ch = epub.EpubHtml(
            title=f"Página {page.order}",
            file_name=f"page_{page.order:03d}.xhtml",
            lang="es",
        )
        ch.add_item(css_item)

        # Imagen del facsimilar (si procede)
        figure_html = ""
        if include_facsimile and page.image:
            embedded = _epub_embed_image(book, page.image.path, page.order)
            if embedded:
                figure_html = (
                    f'<figure class="facsimile">\n'
                    f'  <img src="{embedded}" alt="Facsimilar página {page.order}"/>\n'
                    f'  <figcaption>Facsimilar — página {page.order}</figcaption>\n'
                    f'</figure>\n'
                )

        # Texto reflujado preservando saltos de línea originales
        blocks = _process_blocks(page.text, hyphenate=True)
        if blocks:
            body_parts = []
            for block in blocks:
                if len(block) == 1:
                    # Párrafo normal: una sola línea en el bloque
                    body_parts.append(f"  <p>{_xml_escape(block[0])}</p>")
                else:
                    # Bloque multilínea (verso/poesía): preservar cada línea
                    lines_html = "<br/>\n    ".join(_xml_escape(l) for l in block)
                    body_parts.append(f'  <p class="verse">{lines_html}</p>')
            body_html = "\n".join(body_parts)
        else:
            body_html = '  <p class="empty"><em>(Sin texto transcrito)</em></p>'

        ch.content = (
            '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="es" lang="es">\n'
            "<head>\n"
            f"  <title>Página {page.order}</title>\n"
            '  <link rel="stylesheet" type="text/css" href="style/main.css"/>\n'
            "</head>\n"
            "<body>\n"
            f"  <h2>Página {page.order}</h2>\n"
            f"  {figure_html}"
            f'  <section class="page-body">\n{body_html}\n  </section>\n'
            "</body>\n"
            "</html>"
        )
        book.add_item(ch)
        chapters.append(ch)

    # TOC y spine
    book.toc = (
        epub.Link("title.xhtml", "Portada", "title"),
        (epub.Section("Páginas"), tuple(chapters)),
    )
    book.spine = ["nav", cover_text] + chapters
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    buf = io.BytesIO()
    epub.write_epub(buf, book)
    buf.seek(0)

    safe_name = _safe_filename(document.title)
    suffix = "_facsimilar" if include_facsimile else ""
    response = HttpResponse(buf.read(), content_type="application/epub+zip")
    response["Content-Disposition"] = f'attachment; filename="{safe_name}{suffix}.epub"'
    return response


def _epub_cover_xhtml(document) -> str:
    title = _xml_escape(document.title or "")
    parts = [
        '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="es" lang="es">',
        "<head><title>Portada</title>",
        '<link rel="stylesheet" type="text/css" href="style/main.css"/>',
        "</head>",
        '<body><div class="cover-page">',
        f'<h1 class="title">{title}</h1>',
    ]
    if document.author:
        parts.append(f'<p class="author">{_xml_escape(document.author)}</p>')
    if document.year:
        parts.append(f'<p class="year">{document.year}</p>')
    parts.append('<div class="ornament">❦</div>')
    parts.append('<p class="imprint">A R C H I V O · O C R</p>')
    parts.append("</div></body></html>")
    return "\n".join(parts)


def _epub_css() -> str:
    """Hoja de estilos para el EPUB — pensada para lectores tipo Kindle/Books."""
    return """\
@charset "utf-8";

body {
  font-family: Georgia, "Times New Roman", "DejaVu Serif", serif;
  line-height: 1.55;
  margin: 0;
  padding: 0 1em;
  color: #1a1a1a;
  hyphens: auto;
  -webkit-hyphens: auto;
  -epub-hyphens: auto;
}

h1, h2 {
  font-family: Georgia, "Times New Roman", serif;
  font-weight: normal;
  text-align: center;
}

h1 {
  font-size: 1.8em;
  margin: 2em 0 0.5em;
}

h2 {
  font-size: 1.1em;
  font-style: italic;
  color: #555;
  margin-top: 2.5em;
  margin-bottom: 1.2em;
  letter-spacing: 0.05em;
}

p {
  text-align: justify;
  text-indent: 1.5em;
  margin: 0;
  line-height: 1.55;
  hyphens: auto;
  -webkit-hyphens: auto;
  -epub-hyphens: auto;
}

/* Sin sangría en el primer párrafo de cada sección */
section.page-body > p:first-child,
h2 + p,
figure + section.page-body > p:first-child,
figure + p {
  text-indent: 0;
}

/* Capitular discreta al inicio de cada capítulo */
section.page-body > p:first-child::first-letter {
  font-size: 2.6em;
  float: left;
  line-height: 0.85;
  margin-right: 0.06em;
  margin-top: 0.06em;
  font-family: Georgia, "Times New Roman", serif;
  color: #2a2a2a;
}

p.empty {
  text-align: center;
  color: #999;
  font-style: italic;
  text-indent: 0;
}

/* Verso / poesía: sin sangría, alineado a la izquierda, sin justificar */
p.verse {
  text-indent: 0;
  text-align: left;
  hyphens: none;
  -webkit-hyphens: none;
  -epub-hyphens: none;
  margin: 0 0 0.9em 0;
  line-height: 1.6;
}

figure.facsimile {
  text-align: center;
  margin: 1.5em 0;
  page-break-inside: avoid;
}

figure.facsimile img {
  max-width: 100%;
  max-height: 70vh;
  border: 1px solid #ddd;
}

figure.facsimile figcaption {
  font-size: 0.85em;
  color: #777;
  font-style: italic;
  margin-top: 0.4em;
}

/* Portada */
.cover-page {
  text-align: center;
  margin-top: 6em;
}

.cover-page .title {
  font-size: 2.2em;
  font-style: normal;
  letter-spacing: 0.04em;
  font-weight: normal;
}

.cover-page .author {
  font-size: 1.2em;
  font-style: italic;
  margin-top: 1.5em;
  color: #444;
  text-indent: 0;
}

.cover-page .year {
  font-size: 1em;
  margin-top: 1em;
  color: #666;
  text-indent: 0;
}

.cover-page .ornament {
  margin: 2.5em auto;
  font-size: 1.4em;
  color: #999;
}

.cover-page .imprint {
  margin-top: 6em;
  font-size: 0.85em;
  color: #888;
  letter-spacing: 0.2em;
  text-indent: 0;
}
"""


def _normalize_cover_image(path: str) -> Optional[bytes]:
    """Carga una imagen y la devuelve como JPEG (para usar como cubierta EPUB)."""
    try:
        from PIL import Image as PILImage
    except ImportError:
        logger.warning("Pillow no está instalado; omitiendo cubierta")
        return None
    try:
        with PILImage.open(path) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            # Limita tamaño máximo a 1600px para no inflar el EPUB
            im.thumbnail((1600, 1600), PILImage.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=88, optimize=True)
            return buf.getvalue()
    except Exception as exc:
        logger.warning("No se pudo cargar la cubierta: %s", exc)
        return None


def _epub_embed_image(book, src_path: str, page_order: int) -> Optional[str]:
    """
    Convierte una imagen a JPEG, la añade al EPUB como item, y devuelve
    la ruta interna para referenciarla desde el XHTML.

    Limita el tamaño a 1600px de ancho para mantener el EPUB usable.
    """
    try:
        from ebooklib import epub
        from PIL import Image as PILImage
    except ImportError:
        return None
    try:
        with PILImage.open(src_path) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            im.thumbnail((1600, 1600), PILImage.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85, optimize=True)
            data = buf.getvalue()
    except Exception as exc:
        logger.warning("No se pudo embeber facsimilar p.%d: %s", page_order, exc)
        return None

    name = f"images/page_{page_order:03d}.jpg"
    item = epub.EpubItem(
        uid=f"img_p{page_order:03d}",
        file_name=name,
        media_type="image/jpeg",
        content=data,
    )
    book.add_item(item)
    return name


# ─────────────────────────────────────────────────────────────────────────
#                                 PDF
# ─────────────────────────────────────────────────────────────────────────

def generate_pdf(document, include_facsimile: bool = False):
    """
    Genera un PDF con maquetación tipo libro (B5).

    Args:
        document: instancia de Document.
        include_facsimile: si es True, cada página inserta primero la
                           imagen original (escalada a página completa)
                           y la transcripción reflujada empieza en la
                           página siguiente.
    """
    try:
        from reportlab.lib.pagesizes import B5
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
        from reportlab.lib import colors
        from reportlab.platypus import (
            BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
            PageBreak, NextPageTemplate, HRFlowable, Flowable,
        )
    except ImportError:
        logger.error("reportlab no está instalado")
        return None

    PAGE_W, PAGE_H = B5  # B5 ISO en puntos: ~498.9 × 708.7

    # Márgenes (en mm). El interior es más ancho para dejar espacio al
    # lomo cuando el libro se imprime y encuaderna a doble cara.
    INNER  = 25 * mm   # margen del lado del lomo
    OUTER  = 18 * mm   # margen del lado exterior
    TOP    = 22 * mm
    BOTTOM = 25 * mm   # extra para dejar aire al folio

    # ── Frames ──────────────────────────────────────────────────────────
    def _frame(left, right, fid):
        return Frame(
            left, BOTTOM,
            PAGE_W - left - right, PAGE_H - TOP - BOTTOM,
            id=fid,
            leftPadding=0, rightPadding=0,
            topPadding=0, bottomPadding=0,
        )

    body_recto_frame = _frame(INNER, OUTER, "body_recto")  # impar: lomo a la izquierda
    body_verso_frame = _frame(OUTER, INNER, "body_verso")  # par:    lomo a la derecha
    front_frame      = _frame(OUTER, OUTER, "front")       # cubierta/cortesía: simétrico

    # El frame de facsimilar usa márgenes generosos para que la imagen
    # quepa entera (frame de cuerpo dejaba 203mm de alto, ajustado).
    FACS_MARGIN = 12 * mm
    facsimile_frame = Frame(
        FACS_MARGIN, FACS_MARGIN,
        PAGE_W - 2 * FACS_MARGIN,
        PAGE_H - 2 * FACS_MARGIN,
        id="facsimile",
        leftPadding=0, rightPadding=0,
        topPadding=0, bottomPadding=0,
    )

    # ── DocTemplate que recuerda en qué página real empezó el cuerpo ──
    # _draw_running captura automáticamente esa página la primera vez
    # que se invoca (ver más abajo); no necesitamos hooks adicionales.
    class _BookDoc(BaseDocTemplate):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._first_body_page: Optional[int] = None
            self._book_title = (document.title or "")[:80]

    # ── Decoración por tipo de página ──────────────────────────────────
    def _on_front(canvas, doc):
        """Cubierta / cortesía / TOC / colofón: limpio."""
        return

    def _draw_running(canvas, doc):
        """Encabezado vivo con título + folio en pie (esquina exterior)."""
        if doc._first_body_page is None:
            doc._first_body_page = canvas.getPageNumber()
        folio = canvas.getPageNumber() - doc._first_body_page + 1
        is_recto = (folio % 2) == 1

        canvas.saveState()
        # Encabezado vivo
        canvas.setFont("Times-Italic", 9)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawCentredString(PAGE_W / 2, PAGE_H - 12 * mm, doc._book_title)
        canvas.setStrokeColor(colors.HexColor("#cccccc"))
        canvas.setLineWidth(0.4)
        canvas.line(
            min(INNER, OUTER), PAGE_H - 14 * mm,
            PAGE_W - min(INNER, OUTER), PAGE_H - 14 * mm,
        )
        # Folio en esquina exterior del pie
        canvas.setFont("Times-Roman", 10)
        canvas.setFillColor(colors.HexColor("#444444"))
        if is_recto:
            canvas.drawRightString(PAGE_W - OUTER, BOTTOM - 10 * mm, str(folio))
        else:
            canvas.drawString(OUTER, BOTTOM - 10 * mm, str(folio))
        canvas.restoreState()

    def _on_facsimile(canvas, doc):
        """Páginas con la imagen del facsimilar: pie discreto, sin folio."""
        canvas.saveState()
        canvas.setFont("Times-Italic", 8)
        canvas.setFillColor(colors.HexColor("#999999"))
        canvas.drawCentredString(
            PAGE_W / 2, BOTTOM - 10 * mm,
            "Facsimilar — la transcripción comienza en la página siguiente",
        )
        canvas.restoreState()

    # ── Plantillas de página ────────────────────────────────────────────
    pt_front     = PageTemplate(id="Front",     frames=[front_frame],     onPage=_on_front)
    pt_recto     = PageTemplate(id="Recto",     frames=[body_recto_frame], onPage=_draw_running)
    pt_verso     = PageTemplate(id="Verso",     frames=[body_verso_frame], onPage=_draw_running)
    pt_facsimile = PageTemplate(id="Facsimile", frames=[facsimile_frame],  onPage=_on_facsimile)

    buf = io.BytesIO()
    doc = _BookDoc(
        buf, pagesize=B5,
        title=document.title or "Documento",
        author=document.author or "",
        creator="ArchivoOCR",
        subject=(document.description or "")[:400],
        pageTemplates=[pt_front, pt_recto, pt_verso, pt_facsimile],
    )

    # ── Estilos ─────────────────────────────────────────────────────────
    body_style = ParagraphStyle(
        "Body", fontName="Times-Roman",
        fontSize=10.5, leading=14,
        firstLineIndent=14, alignment=TA_JUSTIFY,
        spaceBefore=0, spaceAfter=0,
    )
    body_first_style = ParagraphStyle(
        "BodyFirst", parent=body_style, firstLineIndent=0,
    )
    page_head_style = ParagraphStyle(
        "PageHead", fontName="Times-Italic",
        fontSize=11, textColor=colors.HexColor("#555"),
        alignment=TA_CENTER, spaceBefore=8, spaceAfter=14,
    )
    cover_title_style = ParagraphStyle(
        "CoverTitle", fontName="Times-Roman",
        fontSize=26, leading=32, alignment=TA_CENTER,
    )
    cover_author_style = ParagraphStyle(
        "CoverAuthor", fontName="Times-Italic",
        fontSize=14, leading=18, alignment=TA_CENTER,
        textColor=colors.HexColor("#444"),
    )
    cover_year_style = ParagraphStyle(
        "CoverYear", fontName="Times-Roman",
        fontSize=12, alignment=TA_CENTER,
        textColor=colors.HexColor("#666"),
    )
    imprint_style = ParagraphStyle(
        "Imprint", fontName="Times-Italic",
        fontSize=9, alignment=TA_CENTER,
        textColor=colors.HexColor("#888"),
    )
    half_title_style = ParagraphStyle(
        "HalfTitle", fontName="Times-Italic",
        fontSize=18, leading=22, alignment=TA_CENTER,
        textColor=colors.HexColor("#444"),
    )
    toc_head_style = ParagraphStyle(
        "TOCHead", fontName="Times-Roman",
        fontSize=18, alignment=TA_CENTER, spaceAfter=18,
    )
    toc_entry_style = ParagraphStyle(
        "TOCEntry", fontName="Times-Roman",
        fontSize=11, leading=18, alignment=TA_LEFT,
        leftIndent=14, firstLineIndent=-14,
    )
    colophon_style = ParagraphStyle(
        "Colophon", fontName="Times-Roman",
        fontSize=9.5, leading=14, alignment=TA_CENTER,
        textColor=colors.HexColor("#555"),
        spaceBefore=4, spaceAfter=4,
    )
    empty_page_style = ParagraphStyle(
        "Empty", parent=body_style, alignment=TA_CENTER,
        textColor=colors.HexColor("#999"),
        firstLineIndent=0,
    )
    # Estilo verso: sin sangría, alineado a la izquierda, sin justificar
    verse_style = ParagraphStyle(
        "Verse", fontName="Times-Roman",
        fontSize=10.5, leading=14,
        firstLineIndent=0, alignment=TA_LEFT,
        spaceBefore=0, spaceAfter=0,
    )

    # ── Construcción del story ──────────────────────────────────────────
    story = []

    # 1. CUBIERTA -------------------------------------------------------
    story.append(NextPageTemplate("Front"))
    story.append(Spacer(1, 60 * mm))
    story.append(Paragraph(_pdf_escape(document.title or "Documento"),
                           cover_title_style))
    if document.author:
        story.append(Spacer(1, 14 * mm))
        story.append(Paragraph(_pdf_escape(document.author), cover_author_style))
    if document.year:
        story.append(Spacer(1, 8 * mm))
        story.append(Paragraph(str(document.year), cover_year_style))
    story.append(Spacer(1, 20 * mm))
    story.append(HRFlowable(
        width=30 * mm, thickness=0.6, color=colors.HexColor("#999999"),
        hAlign="CENTER", spaceBefore=0, spaceAfter=0,
    ))
    story.append(Spacer(1, PAGE_H * 0.18))
    story.append(Paragraph("A R C H I V O · O C R", imprint_style))
    story.append(PageBreak())

    # 2. PORTADILLA -----------------------------------------------------
    story.append(Spacer(1, PAGE_H * 0.35))
    story.append(Paragraph(_pdf_escape(document.title or ""), half_title_style))

    # 3. ÍNDICE (sólo si hay > 1 página) --------------------------------
    pages = list(document.pages.all())
    if len(pages) > 1:
        story.append(PageBreak())  # cierra la portadilla
        story.append(Spacer(1, 12 * mm))
        story.append(Paragraph("Tabla de páginas", toc_head_style))
        for p in pages:
            story.append(Paragraph(f"Página {p.order}", toc_entry_style))
        # No añadimos PageBreak aquí — el del cuerpo se encarga.

    # 4. CUERPO ---------------------------------------------------------
    # Cada iteración abre sus propias páginas con el PageBreak inicial.
    # El _draw_running detecta automáticamente la primera página con
    # plantilla Recto/Verso y fija ahí el origen del folio (folio=1).
    for i, page in enumerate(pages):
        # 4a. Página facsimilar (en su propia plantilla, sin folio)
        img = None
        if include_facsimile and page.image:
            try:
                img = _make_pdf_image(
                    page.image.path,
                    max_width_mm=150, max_height_mm=220,
                )
            except Exception as exc:
                logger.warning("Imagen p.%d falló: %s", page.order, exc)
                img = None

        if img is not None:
            story.append(NextPageTemplate("Facsimile"))
            story.append(PageBreak())
            story.append(img)
            # Volvemos al cuerpo para la transcripción
            story.append(NextPageTemplate(["Recto", "Verso"]))
            story.append(PageBreak())
        else:
            # Sin facsimilar: simplemente abrimos página de cuerpo
            story.append(NextPageTemplate(["Recto", "Verso"]))
            story.append(PageBreak())

        # 4b. Encabezado de la página-fuente
        story.append(Paragraph(f"— Página {page.order} —", page_head_style))

        # 4c. Texto preservando estructura de líneas originales
        blocks = _process_blocks(page.text, hyphenate=True)
        if not blocks:
            story.append(Paragraph("(Sin texto transcrito)", empty_page_style))
        else:
            first_block = True
            for block in blocks:
                if len(block) == 1:
                    # Párrafo de una sola línea: estilo normal con sangría
                    style = body_first_style if first_block else body_style
                    story.append(Paragraph(_pdf_escape(block[0]), style))
                else:
                    # Bloque multilínea (verso/poesía): una línea por Paragraph
                    # con estilo sin sangría y sin justificar
                    for k, line in enumerate(block):
                        # Primera línea del bloque: sin sangría siempre
                        story.append(Paragraph(_pdf_escape(line), verse_style))
                    # Espacio extra entre estrofas/bloques multilínea
                    story.append(Spacer(1, 4))
                first_block = False

    # 5. COLOFÓN --------------------------------------------------------
    story.append(NextPageTemplate("Front"))
    story.append(PageBreak())
    story.append(Spacer(1, PAGE_H * 0.32))
    story.append(Paragraph("Colofón", colophon_style))
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(
        width=20 * mm, thickness=0.4, color=colors.HexColor("#bbbbbb"),
        hAlign="CENTER",
    ))
    story.append(Spacer(1, 6 * mm))

    n_pages = len(pages)
    doc_type = ("manuscrito" if document.document_type == "manuscript"
                else "impreso")
    bits = [f"«{_pdf_escape(document.title or '')}»"]
    if document.author:
        bits.append(f"de {_pdf_escape(document.author)}")
    if document.year:
        bits.append(str(document.year))
    head_line = ", ".join(bits) + "."

    colophon_html = (
        f"{head_line}<br/>"
        f"Tipo: {doc_type}. "
        f"{n_pages} página{'s' if n_pages != 1 else ''}.<br/>"
        f"Transcripción generada por OCR.<br/>"
        f"PDF compuesto el "
        f"{datetime.now(timezone.utc).strftime('%d-%m-%Y')} con ArchivoOCR."
    )
    story.append(Paragraph(colophon_html, colophon_style))

    doc.build(story)
    buf.seek(0)

    safe_name = _safe_filename(document.title)
    suffix = "_facsimilar" if include_facsimile else ""
    response = HttpResponse(buf.read(), content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{safe_name}{suffix}.pdf"'
    return response


# ─────────────────────────────────────────────────────────────────────────
#                              Helpers PDF
# ─────────────────────────────────────────────────────────────────────────

def _make_pdf_image(image_path: str, max_width_mm: int = 146,
                    max_height_mm: int = 200):
    """
    Crea un Image flowable escalado para entrar en max_width × max_height
    (mm). Mantiene relación de aspecto y centra horizontalmente.
    """
    try:
        from reportlab.platypus import Image as RLImage
        from reportlab.lib.units import mm
        from PIL import Image as PILImage
    except ImportError:
        return None

    try:
        with PILImage.open(image_path) as im:
            iw, ih = im.size
    except Exception:
        return None

    max_w = max_width_mm * mm
    max_h = max_height_mm * mm
    scale = min(max_w / iw, max_h / ih, 1.0)
    img = RLImage(image_path, width=iw * scale, height=ih * scale)
    img.hAlign = "CENTER"
    return img


def _pdf_escape(s: str) -> str:
    """Escapa entidades XML reservadas por el mini-lenguaje de Paragraph."""
    if not s:
        return ""
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))


def _xml_escape(s: str) -> str:
    """Escapa entidades XML para XHTML del EPUB (mismo conjunto)."""
    return _pdf_escape(s)


def _safe_filename(name, max_len: int = 60) -> str:
    """Genera un nombre de fichero seguro (sin extensión)."""
    safe = re.sub(r"[^\w\s-]", "", name or "", flags=re.UNICODE)
    safe = re.sub(r"\s+", "_", safe.strip())
    return safe[:max_len] or "documento"
