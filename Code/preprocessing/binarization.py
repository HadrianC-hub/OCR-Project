"""
preprocessing/binarization.py
──────────────────────────────
Binarización adaptativa de Sauvola para documentos impresos y de
máquina de escribir.

Por qué Sauvola y no un umbral global (Otsu):
  Los documentos reales tienen iluminación no uniforme, manchas, bordes
  más oscuros y degradación local del papel. Un umbral global convierte
  esas variaciones en ruido de texto; Sauvola calcula un umbral diferente
  para cada píxel basándose en la media y la desviación estándar de su
  vecindario, lo que elimina el efecto de la iluminación no uniforme y
  preserva los trazos finos de las máquinas de escribir.

La fórmula (Sauvola & Pietikäinen, 2000):
  T(x,y) = μ(x,y) · [1 + k · (σ(x,y)/R − 1)]

  donde:
    μ(x,y)  = media local en la ventana w×w centrada en (x,y)
    σ(x,y)  = desviación estándar local
    R       = rango dinámico de σ (típicamente 128 para imágenes 8-bit)
    k       = parámetro de sensibilidad (0.2–0.5; valores altos
              preservan más fondo, valores bajos más texto)

Responsabilidades de este módulo:
  - Recibe siempre un np.ndarray ya en escala de grises (la selección de
    canal BGR es responsabilidad exclusiva de pipeline.to_gray()).
  - Aplica opcionalmente CLAHE y/o filtro bilateral antes de Sauvola.
  - No realiza I/O: quien necesite cargar una imagen desde disco debe
    usar pipeline.load_image() primero.

Coherencia con el generador:
  generator.py aplica ruido gaussiano, sal-pimienta y suavizado antes
  de guardar las imágenes sintéticas. Esta binarización es la misma
  que se usará en producción, cerrando el domain gap entre datos
  sintéticos y documentos reales.
"""

import numpy as np
import cv2
from scipy.ndimage import uniform_filter


# Parámetros por defecto

DEFAULT_WINDOW = 51     # ventana local en píxeles (debe ser impar)
DEFAULT_K      = 0.18   # sensibilidad; 0.2–0.3 es estándar para documentos
DEFAULT_R      = 128.0  # rango dinámico de σ (fijo para imágenes 8-bit)


# Implementación principal

def sauvola(
    img_gray:         np.ndarray,
    window:           int   = DEFAULT_WINDOW,
    k:                float = DEFAULT_K,
    r:                float = DEFAULT_R,
    global_floor_pct: float = 0.0,
) -> np.ndarray:
    """
    Aplica binarización de Sauvola a una imagen en escala de grises.

    Parámetros
    ----------
    img_gray         : np.ndarray  shape (H, W), dtype uint8
    window           : tamaño de ventana local (impar)
    k                : sensibilidad (0.1–0.5)
    r                : rango dinámico de σ
    global_floor_pct : percentil [0–100] por encima del cual un píxel NUNCA
                       puede clasificarse como texto, independientemente del
                       umbral local. 0 = desactivado.

                       Propósito: Sauvola puede clasificar como texto píxeles
                       de fondo (manchas, grano de papel) que localmente son
                       los más oscuros de su vecindad pero que globalmente son
                       claramente más claros que la tinta real. El piso global
                       impone que ningún píxel más brillante que este percentil
                       del fondo estimado pueda ser texto.

                       Valor recomendado: 85–92 para documentos con suciedad
                       moderada. Valores más altos son más conservadores.

    Retorna
    -------
    np.ndarray  shape (H, W), dtype uint8
        0 = texto (negro), 255 = fondo (blanco).
    """
    if img_gray.ndim != 2:
        raise ValueError(
            f"Se esperaba imagen 2D (escala de grises), "
            f"recibido shape {img_gray.shape}"
        )
    if window % 2 == 0:
        window += 1

    img      = img_gray.astype(np.float64)
    mean     = uniform_filter(img, size=window, mode="reflect")
    mean_sq  = uniform_filter(img ** 2, size=window, mode="reflect")
    variance = np.maximum(mean_sq - mean ** 2, 0.0)
    std      = np.sqrt(variance)

    threshold = mean * (1.0 + k * (std / r - 1.0))
    binary    = np.where(img < threshold, 0, 255).astype(np.uint8)

    # Piso global: forzar a fondo todos los píxeles por encima del percentil
    if global_floor_pct > 0.0:
        floor_val = float(np.percentile(img_gray, global_floor_pct))
        binary[img_gray >= floor_val] = 255

    return binary


# Mejora de contraste

def enhance_contrast(
    gray:       np.ndarray,
    clip_limit: float = 2.5,
    tile_size:  int   = 16,
) -> np.ndarray:
    """
    Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Imprescindible en documentos fotográficos (no escaneados), papel
    envejecido o imágenes con iluminación desigual donde la diferencia
    entre tinta y fondo es menor de ~30 niveles de gris.

    Parámetros
    ----------
    gray       : imagen en escala de grises uint8
    clip_limit : límite de amplificación (2–4 recomendado)
    tile_size  : tamaño de las teselas en píxeles (8–32)
    """
    clahe = cv2.createCLAHE(
        clipLimit    = clip_limit,
        tileGridSize = (tile_size, tile_size),
    )
    return clahe.apply(gray)


# Reducción de ruido con preservación de bordes

def bilateral_denoise(
    gray:        np.ndarray,
    diameter:    int   = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """
    Filtro bilateral: reduce ruido preservando los bordes de los trazos.

    A diferencia del filtro Gaussiano, el bilateral pondera cada píxel
    vecino tanto por su distancia espacial como por su similitud de
    intensidad, lo que evita el borroneo de los bordes del texto.

    Recomendado para documentos con ruido granular (papel envejecido,
    fotocopias) antes de Sauvola. No reemplaza a CLAHE: pueden combinarse.

    Parámetros
    ----------
    gray        : imagen en escala de grises uint8
    diameter    : diámetro del vecindario de cada píxel (5–15)
    sigma_color : varianza de color (25–150; valores altos = más suavizado)
    sigma_space : varianza espacial (25–150; debe coincidir con diameter)
    """
    return cv2.bilateralFilter(gray, diameter, sigma_color, sigma_space)


# Normalización de iluminación por estimación morfológica del fondo

def normalize_illumination(
    gray:        np.ndarray,
    kernel_size: int = 0,
) -> np.ndarray:
    """
    Normaliza la iluminación dividiendo la imagen por la estimación del fondo.

    Usa closing morfológico con un kernel grande para estimar el fondo
    (papel, manchas, variaciones de iluminación). Al dividir la imagen
    original por esa estimación, los píxeles de fondo se aproximan a 255
    y los trazos de tinta mantienen su contraste relativo, independientemente
    de si el papel a su alrededor era claro u oscuro.

    Efecto sobre las manchas del papel
    ────────────────────────────────────
    Una mancha de envejecimiento es un área donde el fondo es localmente más
    oscuro que el resto del papel. El closing la incluye en la estimación del
    fondo. Al dividir, esa zona oscura "sube" a 255, eliminando la mancha de
    la imagen normalizada. La tinta en esa zona también sube, pero sigue siendo
    mucho más oscura que 255 por lo que Sauvola la clasifica correctamente.

    Por qué closing y no opening
    ──────────────────────────────
    Opening erosiona primero (elimina picos finos) y luego dilata. Para texto
    sobre papel, los picos finos SON la tinta → opening los elimina del fondo
    estimado → el fondo subestima la zona de tinta → artifacts post-división.
    Closing dilata primero (llena valles = ignora la tinta) y luego erosiona.
    El resultado es una estimación del fondo que "salta por encima" de los
    trazos de tinta sin ser distorsionada por ellos.

    Parámetros
    ──────────
    gray        : imagen en escala de grises uint8
    kernel_size : tamaño del kernel de closing (0 = automático: min(H,W)//15,
                  mínimo 25). Debe ser mayor que el grosor máximo de los trazos.

    Retorna
    ───────
    np.ndarray  shape (H, W), dtype uint8  — misma forma que la entrada.
    """
    H, W = gray.shape
    if kernel_size <= 0:
        kernel_size = max(25, min(H, W) // 15)
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Dividir: normalizar tono de papel a ~255 sin saturar los trazos
    bg_f       = np.maximum(background.astype(np.float32), 1.0)
    gray_f     = gray.astype(np.float32)
    normalized = np.clip(gray_f / bg_f * 255.0, 0, 255).astype(np.uint8)
    return normalized


# Punto de entrada principal

def binarize(
    img:              np.ndarray,
    window:           int   = DEFAULT_WINDOW,
    k:                float = DEFAULT_K,
    r:                float = DEFAULT_R,
    invert:           bool  = False,
    use_clahe:        bool  = False,
    clahe_clip:       float = 3.0,
    clahe_tile:       int   = 16,
    use_bilateral:    bool  = False,
    bilateral_d:      int   = 9,
    bilateral_sc:     float = 75.0,
    bilateral_ss:     float = 75.0,
    global_floor_pct: float = 0.0,
    use_remove_bg:    bool  = False,
    remove_bg_kernel: int   = 0,
) -> np.ndarray:
    """
    Binariza una imagen de documento.

    La imagen debe llegar ya en escala de grises (shape (H, W)) o BGR
    (shape (H, W, 3)). La selección del canal óptimo (gris estándar vs
    canal B para tinta azul) es responsabilidad de pipeline.to_gray(),
    que debe aplicarse antes de llamar a esta función.

    Parámetros
    ----------
    img : np.ndarray
        Imagen en escala de grises o BGR. Si es BGR se convierte a gris
        con la fórmula estándar.
    window, k, r : parámetros de Sauvola.
    invert : bool
        Invierte la salida (para fondos oscuros).
    use_remove_bg : bool
        Normaliza iluminación antes del pipeline bilateral+CLAHE+Sauvola.
        Usar para documentos con manchas de papel o iluminación desigual.
        Ver normalize_illumination() para detalles.
    remove_bg_kernel : int
        Tamaño del kernel morfológico para la normalización (0 = automático).
    use_clahe : bool
        Amplifica contraste local antes de Sauvola.
    clahe_clip, clahe_tile : parámetros de CLAHE.
    use_bilateral : bool
        Aplica filtro bilateral antes de Sauvola (y antes de CLAHE si ambos).
    bilateral_d, bilateral_sc, bilateral_ss : parámetros del bilateral.
    global_floor_pct : float [0–100]
        Percentil por encima del cual ningún píxel puede ser texto.
        Ver docstring de sauvola() para detalles.
        Valor típico para documentos ruidosos: 85–93. 0 = desactivado.

    Retorna
    -------
    np.ndarray  shape (H, W), dtype uint8
        0 = texto (negro), 255 = fondo (blanco).
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Paso 0: normalización de iluminación (elimina manchas y gradientes)
    # Debe aplicarse ANTES del bilateral para que el bilateral trabaje sobre
    # una imagen ya sin variaciones lentas de iluminación.
    if use_remove_bg:
        gray = normalize_illumination(gray, kernel_size=remove_bg_kernel)

    if use_bilateral:
        gray = bilateral_denoise(
            gray,
            diameter=bilateral_d,
            sigma_color=bilateral_sc,
            sigma_space=bilateral_ss,
        )

    if use_clahe:
        gray = enhance_contrast(gray, clip_limit=clahe_clip, tile_size=clahe_tile)

    binary = sauvola(gray, window=window, k=k, r=r,
                     global_floor_pct=global_floor_pct)

    if invert:
        binary = 255 - binary

    return binary


# Post-procesado morfológico

def clean_binary(
    binary:      np.ndarray,
    morph_open:  int = 0,
    morph_close: int = 0,
) -> np.ndarray:
    """
    Post-procesado morfológico ligero sobre la imagen binaria.

    morph_open  > 0 : elimina puntitos de ruido (sal y pimienta residual)
    morph_close > 0 : rellena pequeños huecos en los trazos de texto

    Valores recomendados: 2–3. Nunca usar valores grandes o se degradan
    los trazos finos de las máquinas de escribir.
    """
    result = binary.copy()

    if morph_open > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_open, morph_open)
        )
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)

    if morph_close > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_close, morph_close)
        )
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    return result


def filter_small_components(
    binary:        np.ndarray,
    min_area_px:   int,
    max_area_frac: float = 0.10,
) -> np.ndarray:
    """
    Elimina componentes conexos con área fuera de [min_area_px, max_area].

    ⚠ Umbral fijo — frágil ante variaciones de DPI y tamaño de fuente.
    Usa adaptive_filter_components() siempre que sea posible.

    Parámetros
    ----------
    binary        : imagen binaria uint8 (0=texto, 255=fondo)
    min_area_px   : área mínima en píxeles para conservar un componente
    max_area_frac : fracción máxima del área total de la imagen (default 0.10)
    """
    H, W     = binary.shape
    max_area = int(H * W * max_area_frac)

    fg = (binary < 128).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg, connectivity=8
    )

    result = np.full_like(binary, 255)
    for label_id in range(1, n_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if min_area_px <= area <= max_area:
            result[labels == label_id] = 0

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Filtrado adaptativo — sin parámetros externos
# ─────────────────────────────────────────────────────────────────────────────

def _noise_gap_threshold(areas: np.ndarray) -> int:
    """
    Detecta el umbral natural que separa blobs de ruido de componentes reales
    buscando la primera brecha significativa en la distribución de áreas.

    Principio
    ─────────
    Los blobs de ruido de Sauvola forman un cluster denso en la zona baja
    del histograma: 1, 1, 2, 2, 3, 4, 5... (muchos, pequeños, consecutivos).
    El primer componente de texto real (punto, coma, punto de i) es
    notablemente mayor: hay un salto relativo ≥ GAP_RATIO.

    Tres guardas conservadoras para no borrar texto legítimo
    ─────────────────────────────────────────────────────────
    1. MAX_NOISE_AREA = 50 px²
       Límite absoluto de búsqueda. A cualquier DPI donde el OCR sea viable
       (≥ 150 dpi), un punto real tiene área > 50 px². Lo que cae por debajo
       son speckles de 1–3 px de lado impossibles de ser tinta.
       (A 72 dpi el documento es borroso e ilegible independientemente.)

    2. GAP_RATIO = 2.5
       El salto debe ser brusco. Evita confundir dos puntos de tamaño
       ligeramente distinto con un salto ruido→texto.

    3. MIN_NOISE_COUNT = 5
       Se necesitan ≥ 5 componentes debajo del umbral para activar el filtro.
       Si hay pocos componentes pequeños son probablemente puntuación real
       (documento limpio sin ruido apreciable) y no se tocan.

    Retorna
    -------
    int — threshold de área; componentes ≤ este valor son ruido.
          0 si no se encontró brecha clara → no filtrar.
    """
    GAP_RATIO       = 2.5
    MAX_NOISE_AREA  = 50    # px² — cap absoluto
    MIN_NOISE_COUNT = 5     # mínimo de blobs en el cluster

    sorted_areas = np.sort(areas)

    for i in range(len(sorted_areas) - 1):
        a0 = int(sorted_areas[i])
        a1 = int(sorted_areas[i + 1])

        if a0 > MAX_NOISE_AREA:
            break

        if a0 >= 1 and a1 >= a0 * GAP_RATIO:
            noise_count = i + 1          # todos los anteriores son ≤ a0
            if noise_count >= MIN_NOISE_COUNT:
                return a0                # umbral: eliminar ≤ a0

    return 0                             # sin brecha → no filtrar


def adaptive_filter_components(
    binary:        np.ndarray,
    max_area_frac: float = 0.10,
) -> np.ndarray:
    """
    Elimina blobs de ruido de Sauvola sin ningún parámetro externo.

    Por qué es mejor que filter_small_components()
    ───────────────────────────────────────────────
    filter_small_components() necesita min_area_px, que depende del DPI y
    del tamaño de fuente — ambos desconocidos en tiempo de configuración.
    Un umbral fijo calculado antes de ver la imagen binarizada produce:
      • Umbrales demasiado altos → borra puntos, comas, puntos de i, tildes.
      • Umbrales demasiado bajos → deja ruido de papel.

    Este método deriva el umbral directamente de la distribución real de
    componentes del binario, sin suponer nada sobre el documento.

    Comportamiento esperado según tipo de documento
    ─────────────────────────────────────────────────
    • Documento limpio / bilateral activo:
      Pocos o ningún componente < 50 px² → threshold=0 → sin filtrado.

    • Ruido moderado de papel (bilateral activo):
      Cluster de ≥ 5 blobs en 1–10 px², salto a punto/coma en 40–80 px²
      → threshold ≈ 10 → blobs borrados, puntuación conservada.

    • Documento muy degradado (bajo contraste, sin bilateral):
      Muchos blobs 1–30 px², salto a puntuación en 50–150 px²
      → threshold ≈ 30 → blobs borrados, puntuación conservada.

    • Impreso de baja resolución (< 150 dpi):
      Ruido y puntos solapan en área, no hay salto ≥ 2.5× bajo 50 px²
      → threshold=0 → sin filtrado (conserva todo, incluso posible ruido,
      pero nunca borra puntuación real).

    Parámetros
    ----------
    binary        : imagen binaria uint8 (0=texto, 255=fondo)
    max_area_frac : fracción máxima del área total → protege contra manchas
                    enormes de iluminación. Default 0.10.

    Retorna
    -------
    np.ndarray  misma shape y dtype que la entrada.
    """
    H, W     = binary.shape
    max_area = int(H * W * max_area_frac)

    fg = (binary < 128).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg, connectivity=8
    )

    if n_labels < 3:
        return binary                    # prácticamente vacío

    areas = np.array(
        [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)],
        dtype=np.int64,
    )

    noise_thresh = _noise_gap_threshold(areas)

    result = np.full_like(binary, 255)
    for label_id in range(1, n_labels):
        area = int(areas[label_id - 1])
        if area <= noise_thresh:
            continue                     # blob de ruido → descartar
        if area > max_area:
            continue                     # artefacto enorme → descartar
        result[labels == label_id] = 0

    return result

def gaussian_denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(gray, (5, 5), 0)