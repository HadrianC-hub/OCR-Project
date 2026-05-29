"""
apps/documents/typography.py — Utilidades de tipografía para los exports.

Funciones puras (sin estado) para transformar texto OCR en prosa
publicable. Las funciones públicas son:

    reflow_paragraphs(text)       → list[str]   párrafos reconstruidos
    smart_typography(text, lang)  → str         microtipografía aplicada
    add_soft_hyphens(text, lang)  → str         soft hyphens (necesita pyphen)

Las tres se diseñan para encadenarse en este orden.
"""

from __future__ import annotations

import re
from typing import List


# ── Reflujo de párrafos ─────────────────────────────────────────────────

def reflow_paragraphs(text: str) -> List[str]:
    """
    Convierte texto con saltos físicos del OCR en una lista de párrafos.

    Reglas:
      - Una o más líneas en blanco marcan frontera de párrafo.
      - Las líneas físicas dentro de un párrafo se unen con un espacio.
      - Si una línea termina en '-' y la siguiente empieza en minúscula,
        se asume guión blando (corte de fin de línea de imprenta) y se
        une sin guión ni espacio.

    Ejemplo:
        >>> reflow_paragraphs("En un lugar\\nde la Mancha\\n\\nde cuyo\\nnombre")
        ['En un lugar de la Mancha', 'de cuyo nombre']
    """
    if not text:
        return []

    paragraphs: List[str] = []
    current: List[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if current:
                paragraphs.append(_join_lines(current))
                current = []
        else:
            current.append(line)

    if current:
        paragraphs.append(_join_lines(current))

    return paragraphs


def split_blocks(text: str) -> List[List[str]]:
    """
    Divide el texto en bloques separados por líneas en blanco.
    Cada bloque es una lista de líneas individuales (sin unir).
    A diferencia de reflow_paragraphs, preserva los saltos de línea
    dentro de cada bloque, lo que es esencial para conservar la
    estructura visual de poesía y verso.

    Ejemplo:
        >>> split_blocks("Verso uno\\nVerso dos\\n\\nOtro párrafo")
        [['Verso uno', 'Verso dos'], ['Otro párrafo']]
    """
    if not text:
        return []

    blocks: List[List[str]] = []
    current: List[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)

    if current:
        blocks.append(current)

    return blocks



    """Une líneas físicas en un párrafo, gestionando guiones de fin-de-línea."""
    if not lines:
        return ""
    out = lines[0]
    for line in lines[1:]:
        if out.endswith("-") and line and line[0].islower():
            # Guión blando: une sin guión ni espacio
            out = out[:-1] + line
        else:
            out = out + " " + line
    return out


# ── Microtipografía ─────────────────────────────────────────────────────

EM_DASH = "\u2014"   # —
EN_DASH = "\u2013"   # –
ELLIPSIS = "\u2026"  # …

LDQUO = "\u201c"     # “
RDQUO = "\u201d"     # ”
LSQUO = "\u2018"     # ‘
RSQUO = "\u2019"     # ’ (también apóstrofe)

NBSP = "\u00a0"      # espacio inseparable


def smart_typography(text: str, lang: str = "es") -> str:
    """
    Aplica microtipografía: comillas curvas, em-dash, elipsis, colapso de
    espacios múltiples, espacios inseparables tras abreviaturas comunes.

    Se usan comillas inglesas curvas (“ ”) que funcionan bien en español
    moderno y son universalmente soportadas por las fuentes. Quien
    quiera guillemets «» puede teclearlos directamente; los respetamos.
    """
    if not text:
        return text

    # Em-dash: '--' o '---' → —
    text = re.sub(r"-{2,3}", EM_DASH, text)

    # Elipsis con tres puntos
    text = text.replace("...", ELLIPSIS)

    # Comillas curvas
    text = _smart_double_quotes(text)
    text = _smart_single_quotes(text)

    # Colapsar espacios múltiples (sólo espacios literales, no tabs/newlines)
    text = re.sub(r"  +", " ", text)

    # Espacios inseparables tras abreviaturas frecuentes (evita huérfanos
    # como "Sr.\nGarcía"). Sólo aplicamos para abreviaturas seguidas de
    # capitalización (probable nombre propio).
    text = re.sub(
        r"\b(Sr|Sra|Srta|Dr|Dra|D|Dña|p|pp|cap|art|n|núm|vol|tomo|fig|tab|cf)\.\s+(?=[A-ZÁÉÍÓÚÑ])",
        lambda m: m.group(1) + "." + NBSP,
        text,
    )

    return text


def _smart_double_quotes(text: str) -> str:
    """Convierte '"' en comillas curvas alternando open/close."""
    out = []
    open_quote = True
    for ch in text:
        if ch == '"':
            out.append(LDQUO if open_quote else RDQUO)
            open_quote = not open_quote
        else:
            out.append(ch)
    return "".join(out)


def _smart_single_quotes(text: str) -> str:
    """
    Convierte "'" en comillas curvas o apóstrofes.

    Heurística:
      - Entre dos letras → apóstrofe (’) — interior de palabra: l'amour, don't
      - Inicio de palabra (precedido por espacio o inicio) → apertura (‘)
      - Fin de palabra → cierre/apóstrofe (’)
    """
    out = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch == "'":
            prev = text[i - 1] if i > 0 else " "
            nxt = text[i + 1] if i + 1 < n else " "
            if prev.isalpha() and nxt.isalpha():
                out.append(RSQUO)            # apóstrofe interior
            elif prev.isspace() or i == 0:
                out.append(LSQUO)            # apertura
            else:
                out.append(RSQUO)
        else:
            out.append(ch)
    return "".join(out)


# ── Separación silábica (opcional, requiere pyphen) ────────────────────

def add_soft_hyphens(text: str, lang: str = "es") -> str:
    """
    Inserta U+00AD (soft hyphen) en posiciones válidas de separación
    silábica. ReportLab y los lectores EPUB respetan el soft hyphen para
    romper líneas, dando justificado mucho mejor.

    Si pyphen no está instalado o no soporta el idioma, devolvemos el
    texto tal cual (degradación elegante).
    """
    if not text:
        return text
    try:
        import pyphen  # noqa: PLC0415
    except ImportError:
        return text

    try:
        dic = pyphen.Pyphen(lang=lang)
    except KeyError:
        return text

    SHY = "\u00ad"

    def _hyphenate_word(match):
        word = match.group(0)
        if len(word) < 5:
            return word
        try:
            return dic.inserted(word, hyphen=SHY)
        except Exception:
            return word

    # Sólo sobre palabras de 5+ letras (incluyendo acentos)
    return re.sub(r"\w{5,}", _hyphenate_word, text, flags=re.UNICODE)
