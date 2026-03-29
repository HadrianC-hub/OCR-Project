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

def auto_tune_sauvola_k(
    gray: np.ndarray,
    window: int,
    k_init: float = 0.15,
    target_ink: tuple = (0.08, 0.18),
    max_iter: int = 5,
) -> float:

    k = k_init

    # 🔹 1. Crear máscara de “zonas relevantes”
    # evita fondo blanco dominante
    thr = np.percentile(gray, 90)
    mask = gray < thr

    # fallback por si la máscara queda vacía
    if mask.sum() < gray.size * 0.05:
        mask = np.ones_like(gray, dtype=bool)

    for _ in range(max_iter):
        binary = sauvola(gray, window=window, k=k)

        # 🔹 2. Calcular tinta SOLO en zonas relevantes
        ink_ratio = float((binary[mask] < 128).mean())

        if ink_ratio > target_ink[1]:
            k *= 0.85
        elif ink_ratio < target_ink[0]:
            k *= 1.15
        else:
            break

    return float(np.clip(k, 0.05, 0.35))

# Mejora de contraste

def enhance_contrast(
    gray:       np.ndarray,
    clip_limit: float = 2.5,
    tile_size:  int   = 16,
) -> np.ndarray:

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

    return cv2.bilateralFilter(gray, diameter, sigma_color, sigma_space)


# Normalización de iluminación por estimación morfológica del fondo

def normalize_illumination(
    gray:        np.ndarray,
    kernel_size: int = 0,
) -> np.ndarray:

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

    k = auto_tune_sauvola_k(gray, window=window, k_init=k)

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