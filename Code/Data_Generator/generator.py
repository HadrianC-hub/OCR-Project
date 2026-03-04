import re
import random
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image, ImageDraw, ImageFont, ImageFilter


#  CONFIGURACIÓN

CORPUS_DIR  = "Corpus"
FONTS_DIR   = "Fonts"
OUTPUT_DIR  = "Generated"

TOTAL_IMAGES = 20000
FONT_SIZE    = 80

MIN_TOKENS = 5
MAX_TOKENS = 15

MAX_CHARS_PER_LINE = 75

MARGIN_TOP    = 22
MARGIN_BOTTOM = 22
MARGIN_LEFT   = 30
MARGIN_RIGHT  = 30

ENABLE_TYPEWRITER_FX = True

# FRECUENCIAS OBJETIVO DE LA GENERACIÓN POR CARACTER

FRECUENCIAS_OBJETIVO = {

    # ── Vocales minúsculas (Grupo A) ──────────────────────────────────────────
    "a": 10.2, "e": 11.5, "i":  5.3, "o":  7.5, "u":  3.1,

    # ── Vocales acentuadas minúsculas (Grupo B) ───────────────────────────────
    "á":  2.5, "é":  2.5, "í":  2.5, "ó":  2.5, "ú":  2.5, "ü":  1.5,

    # ── Consonantes comunes minúsculas (Grupo A) ──────────────────────────────
    "s":  6.8, "r":  5.9, "n":  5.7, "l":  5.0, "d":  4.2,
    "t":  3.9, "c":  2.9, "m":  2.5, "p":  2.3,

    # ── Consonantes medias minúsculas (Grupo A/B) ─────────────────────────────
    "b":  2.5, "h":  2.5, "q":  2.5, "v":  2.5,
    "f":  2.5, "g":  2.5, "y":  2.5, "z":  2.5,

    # ── Consonantes raras minúsculas (Grupo B/C) ──────────────────────────────
    "j":  2.0,   # frecuencia real ~0.44 % -> sobremuestreo ×4.5
    "ñ":  2.0,   # frecuencia real ~0.17 % -> sobremuestreo ×11.8
    "k":  2.0,   # frecuencia real ~0.01 % -> sobremuestreo ×200
    "w":  2.0,   # frecuencia real ~0.004% -> sobremuestreo ×500
    "x":  2.0,   # frecuencia real ~0.22 % -> sobremuestreo ×9

    # ── Vocales mayúsculas (Grupo A reducido) ─────────────────────────────────
    "A":  4.5, "E":  5.0, "I":  2.5, "O":  3.5, "U":  1.5,

    # ── Vocales mayúsculas acentuadas (Grupo B) ───────────────────────────────
    "Á":  1.5, "É":  1.5, "Í":  1.5, "Ó":  1.5, "Ú":  1.5, "Ü":  1.0,

    # ── Consonantes mayúsculas comunes (Grupo A reducido) ─────────────────────
    "S":  3.0, "R":  2.5, "N":  2.5, "L":  2.5, "D":  2.0,
    "T":  2.0, "C":  2.0, "M":  2.0, "P":  2.0,

    # ── Consonantes mayúsculas medias (Grupo B) ───────────────────────────────
    "B":  1.5, "H":  1.5, "Q":  1.5, "V":  1.5,
    "F":  1.5, "G":  1.5, "Y":  1.5, "Z":  1.5,

    # ── Consonantes mayúsculas raras (Grupo B/C) ──────────────────────────────
    "J":  1.5,
    "Ñ":  1.5,
    "K":  1.0, "W":  1.0, "X":  1.0,

    # ── Dígitos (Grupo B) ─────────────────────────────────────────────────────
    # Todos con el mismo peso base; en texto literario los dígitos son poco
    # frecuentes pero conviene que todos estén representados por igual.
    "0":  1.5, "1":  1.5, "2":  1.5, "3":  1.5, "4":  1.5,
    "5":  1.5, "6":  1.5, "7":  1.5, "8":  1.5, "9":  1.5,

    # ── Puntuación básica (Grupo A/B) ─────────────────────────────────────────
    ".":  3.0, ",":  2.5, ";":  1.5, ":":  1.5,
    "!":  1.5, "?":  1.5, "¡":  2.0, "¿":  2.0,
    "(":  1.0, ")":  1.0, '"':  1.0, "'":  1.0,
    "-":  2.0,   # guión corto

    # ── Símbolos tipográficos españoles (Grupo B) ────────────────────────────
    # Pesos elevados respecto al valor real para garantizar ≥500 apariciones
    # en el dataset (umbral mínimo para entrenamiento confiable).
    # Requieren corpus de augmentación (Augmented_corpus.txt en Corpus/).
    "—":  3.0,   # raya española — diálogos literarios     (real ~0.02 % → ×150)
    "«":  2.0,   # comilla latina apertura                  (real ~0.01 % → ×200)
    "»":  2.0,   # comilla latina cierre                    (real ~0.01 % → ×200)

    # ── Símbolos especiales (Grupo C) ─────────────────────────────────────────
    # Pesos ajustados para superar umbral de 500 apariciones.
    # & $ # estaban bajo 500 con pesos originales (0.1, 0.1, 0.05).
    "%":  1.5,   # porcentaje  — textos financieros/estadísticos
    "&":  1.5,   # ampersand   — razones sociales (García & López)
    "$":  1.5,   # símbolo moneda — precios y valores monetarios
    "#":  0.8,   # almohadilla
    "*":  1.0,   # asterisco   — notas a pie
    "[":  0.8,   # corchete apertura
    "]":  0.8,   # corchete cierre
}

#  CORPUS

def leer_corpus(corpus_dir: str):
    tokens_totales = []
    corpus_path = Path(corpus_dir)

    if not corpus_path.exists():
        print(f"No se encontró la carpeta '{corpus_dir}'")
        return [], []

    regex = r"[a-zA-ZáéíóúÁÉÍÓÚüÜñÑ]+|\d|[—«»]|[^\w\s]"

    for archivo in sorted(corpus_path.glob("*.txt")):
        try:
            texto  = archivo.read_text(encoding="utf-8")
            tokens = re.findall(regex, texto)
            tokens_totales.extend(tokens)
            print(f"    {archivo.name}: {len(tokens):,} tokens")
        except Exception as e:
            print(f"    {archivo.name}: {e}")

    tokens_unicos = list(set(tokens_totales))
    print(f"\nTokens únicos: {len(tokens_unicos):,}")
    return tokens_totales, tokens_unicos

def indexar_por_caracter(tokens_unicos: list) -> dict:
    indice = defaultdict(list)
    for token in tokens_unicos:
        for char in set(token):
            indice[char].append(token)
    print(f"Índice creado: {len(indice)} claves de carácter")
    return indice

def obtener_fuentes(fonts_dir: str) -> list:
    fonts_path = Path(fonts_dir)

    if not fonts_path.exists():
        print(f"No se encontró la carpeta '{fonts_dir}'")
        return []

    fuentes = sorted(fonts_path.glob("*.ttf")) + sorted(fonts_path.glob("*.otf"))
    print(f"\n{len(fuentes)} fuentes encontradas:")
    for f in fuentes:
        print(f"    {f.name}")
    return [str(f) for f in fuentes]

#  STRINGS BALANCEADOS

def _calcular_deficit(contador: Counter, obj: dict, total: int) -> dict:
    total_peso = sum(obj.values())
    return {
        char: (peso / total_peso) * total - contador.get(char, 0)
        for char, peso in obj.items()
    }

def _elegir_token(indice: dict, deficit: dict, tokens_todos: list) -> str:
    prioritarios = sorted(deficit.items(), key=lambda x: x[1], reverse=True)[:15]
    random.shuffle(prioritarios)
    for char, def_val in prioritarios:
        if def_val > 0 and char in indice:
            return random.choice(indice[char])
    return random.choice(tokens_todos)

def generar_strings_balanceados(
    tokens_todos: list,
    indice: dict,
    frecuencias_obj: dict,
    num_samples: int,
    min_tokens: int,
    max_tokens: int,
    max_chars: int,
) -> list:
    strings         = []
    contador_global = Counter()

    for _ in range(num_samples):
        n_tokens     = random.randint(min_tokens, max_tokens)
        tokens_frase = []
        chars_usados = 0

        # Tope de chars aleatorio por frase para distribuir longitudes
        # uniformemente en [max_chars//3, max_chars] en lugar de concentrarlas
        # cerca del límite superior (efecto rampa observado en el EDA).
        max_chars_frase = random.randint(max_chars // 3, max_chars)

        total_actual = sum(contador_global.values())
        deficit = (
            _calcular_deficit(contador_global, frecuencias_obj, total_actual)
            if total_actual > 0
            else {k: 1.0 for k in frecuencias_obj}
        )

        for _ in range(n_tokens):
            token = _elegir_token(indice, deficit, tokens_todos)
            if chars_usados + len(token) + 1 > max_chars_frase:
                break
            tokens_frase.append(token)
            chars_usados += len(token) + 1

        if not tokens_frase:
            tokens_frase = [random.choice(tokens_todos)[:max_chars]]

        frase = " ".join(tokens_frase)
        strings.append(frase)
        contador_global.update(c for c in frase if c != " ")

    return strings

#  RENDERIZADO DE TEXTO

_FONT_CACHE: dict = {}

def _cargar_fuente(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (font_path, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(font_path, size)
    return _FONT_CACHE[key]

def renderizar_texto(
    texto: str,
    font_path: str,
    font_size: int,
    margin_top: int    = MARGIN_TOP,
    margin_bottom: int = MARGIN_BOTTOM,
    margin_left: int   = MARGIN_LEFT,
    margin_right: int  = MARGIN_RIGHT,
) -> Image.Image:
    font = _cargar_fuente(font_path, font_size)

    tmp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bx0, by0, bx1, by1 = tmp_draw.textbbox((0, 0), texto, font=font, anchor="lt")

    bbox_w = bx1 - bx0
    bbox_h = by1 - by0

    canvas_w = max(1, bbox_w + margin_left + margin_right)
    canvas_h = max(1, bbox_h + margin_top  + margin_bottom)

    img  = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.text((margin_left - bx0, margin_top - by0), texto,
              font=font, fill=(0, 0, 0), anchor="lt")

    return img

# Parámetros ajustados para pipeline con binarización posterior.
# Los rangos son ligeramente más amplios que en un pipeline sin binarización
# porque el preprocesamiento actúa como regulador: el efecto que llega al
# modelo es siempre la binarización del ruido, no el ruido en sí.
GAUSS_SIGMA_RANGE = (2.0, 8.0)      # sigma más alto -> bordes más difusos al binarizar
SP_DENSITY_RANGE  = (0.001, 0.007)  # hasta 0.7 % de píxeles afectados
BLUR_PROB         = 0.50            # 50 % de imágenes con ink bleed simulado
BLUR_RADIUS_RANGE = (0.3, 0.9)      # radio mayor -> trazos más engrosados al binarizar

def aplicar_efecto_maquina(img: Image.Image) -> Image.Image:
    arr  = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # 1. Ruido gaussiano — modifica cómo binariza Sauvola en los bordes
    sigma = random.uniform(*GAUSS_SIGMA_RANGE)
    arr  += np.random.normal(0, sigma, arr.shape)

    # 2. Sal y pimienta — artefactos puntuales que persisten tras binarizar
    densidad  = random.uniform(*SP_DENSITY_RANGE)
    n_pixeles = int(h * w * densidad)

    ys = np.random.randint(0, h, n_pixeles)
    xs = np.random.randint(0, w, n_pixeles)
    arr[ys, xs] = 255.0   # sal (píxeles blancos aislados)

    ys = np.random.randint(0, h, n_pixeles)
    xs = np.random.randint(0, w, n_pixeles)
    arr[ys, xs] = 0.0     # pimienta (píxeles negros aislados)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # 3. Suavizado leve — simula ink bleed y resolución finita del escáner
    if random.random() < BLUR_PROB:
        img = img.filter(ImageFilter.GaussianBlur(
            radius=random.uniform(*BLUR_RADIUS_RANGE)
        ))

    return img

#  VERIFICACIÓN | GENERACIÓN DEL DATASET Y REPORTE

def verificar_imagen(img: Image.Image) -> bool:
    """Devuelve True si no hay píxeles oscuros en los bordes (= sin recorte)."""
    arr    = np.array(img.convert("L"))
    umbral = 180
    return (
        not np.any(arr[:2,  :] < umbral) and
        not np.any(arr[-2:, :] < umbral) and
        not np.any(arr[:,  :2] < umbral) and
        not np.any(arr[:, -2:] < umbral)
    )

def generar_dataset(
    strings_por_fuente: list,
    fuentes: list,
    output_dir: str,
    font_size: int,
    enable_typewriter_fx: bool,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total      = sum(len(s) for s in strings_por_fuente)
    re_renders = 0

    print(f"\nGENERANDO DATASET")
    print(f"    Fuentes: {len(fuentes)}")
    print(f"    Imágenes total: {total}")
    print(f"    Márgenes (px): top={MARGIN_TOP} bot={MARGIN_BOTTOM} left={MARGIN_LEFT} right={MARGIN_RIGHT}")
    print(f"    Destino: {output_path.resolve()}\n")

    for idx, (fuente, strings) in enumerate(zip(fuentes, strings_por_fuente)):
        font_name = Path(fuente).stem
        print(f"\n{'-'*80}")
        print(f"Fuente {idx+1}/{len(fuentes)}: {font_name}")

        try:
            _cargar_fuente(fuente, font_size)
        except Exception as e:
            print(f"No se puede cargar la fuente: {e}")
            continue

        contador = 0
        for texto in strings:
            try:
                img = renderizar_texto(texto, fuente, font_size)

                if not verificar_imagen(img):
                    re_renders += 1
                    img = renderizar_texto(
                        texto, fuente, font_size,
                        margin_top    = MARGIN_TOP    + 20,
                        margin_bottom = MARGIN_BOTTOM + 20,
                        margin_left   = MARGIN_LEFT   + 20,
                        margin_right  = MARGIN_RIGHT  + 20,
                    )

                if enable_typewriter_fx:
                    img = aplicar_efecto_maquina(img)

                base = f"{font_name}_{contador:06d}"
                img.save(output_path / f"{base}.png")
                (output_path / f"{base}.txt").write_text(texto, encoding="utf-8")

                contador += 1
                if contador % 200 == 0:
                    print(f"  {contador}/{len(strings)}")

            except Exception as e:
                print(f"Error [{contador}]: {e}")
                continue

        print(f"{contador} imágenes")

    print(f"\n{'='*80}")
    print(f"TOTAL: {total} imágenes  |  Re-renders por borde: {re_renders}")
    print(f"{'='*80}")

#  MAIN
if __name__ == "__main__":
    print("  GENERADOR OCR - ESPAÑOL")

    print("\nLeyendo corpus...")
    tokens_todos, tokens_unicos = leer_corpus(CORPUS_DIR)
    if not tokens_todos:
        print("Corpus vacío. Revisa 'Corpus/'"); exit(1)

    indice  = indexar_por_caracter(tokens_unicos)
    fuentes = obtener_fuentes(FONTS_DIR)
    if not fuentes:
        print("Sin fuentes. Añade .ttf/.otf a 'Fonts/'"); exit(1)

    imgs_por_fuente = max(1, TOTAL_IMAGES // len(fuentes))
    print(f"\n{TOTAL_IMAGES} imágenes / {len(fuentes)} fuentes = {imgs_por_fuente} por fuente")

    print(f"\nGenerando strings...")
    strings_por_fuente = []
    for i, fuente in enumerate(fuentes):
        print(f"  [{i+1}/{len(fuentes)}] {Path(fuente).stem}")
        strings_por_fuente.append(
            generar_strings_balanceados(
                tokens_todos, indice, FRECUENCIAS_OBJETIVO,
                imgs_por_fuente, MIN_TOKENS, MAX_TOKENS, MAX_CHARS_PER_LINE,
            )
        )

    generar_dataset(strings_por_fuente, fuentes, OUTPUT_DIR, FONT_SIZE, ENABLE_TYPEWRITER_FX)

    print("\n" + "="*80)
    print("DATASET LISTO")
    print("="*80)