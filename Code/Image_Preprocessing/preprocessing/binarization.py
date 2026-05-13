import cv2
import numpy as np
from scipy.ndimage import uniform_filter


DEFAULT_WINDOW = 51
DEFAULT_K      = 0.18
DEFAULT_R      = 128.0   # rango dinamico fijo para imagenes 8-bit


def sauvola(
    img_gray:         np.ndarray,
    window:           int   = DEFAULT_WINDOW,
    k:                float = DEFAULT_K,
    r:                float = DEFAULT_R,
    global_floor_pct: float = 0.0,
    pre_blur:         float = 0.6,
) -> np.ndarray:
    # Binarizacion Sauvola con pre-blur opcional y piso global anti-ruido.
    if img_gray.ndim != 2:
        raise ValueError(f"Se esperaba imagen 2D, recibido shape {img_gray.shape}")
    if window % 2 == 0:
        window += 1

    if pre_blur and pre_blur > 0.0:
        img_for_stats = cv2.GaussianBlur(img_gray, (0, 0), sigmaX=pre_blur, sigmaY=pre_blur)
    else:
        img_for_stats = img_gray

    img      = img_for_stats.astype(np.float64)
    mean     = uniform_filter(img, size=window, mode="reflect")
    mean_sq  = uniform_filter(img ** 2, size=window, mode="reflect")
    variance = np.maximum(mean_sq - mean ** 2, 0.0)
    std      = np.sqrt(variance)

    threshold = mean * (1.0 + k * (std / r - 1.0))
    binary    = np.where(img < threshold, 0, 255).astype(np.uint8)

    if global_floor_pct > 0.0:
        floor_val   = float(np.percentile(img_gray, global_floor_pct))
        otsu_thr, _ = cv2.threshold(
            img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if floor_val > float(otsu_thr) + 10:
            binary[img_gray >= floor_val] = 255

    return binary


def auto_tune_sauvola_k(
    gray: np.ndarray,
    window: int,
    k_init: float = 0.15,
    target_ink: tuple = (0.07, 0.16),
    max_iter: int = 5,
) -> float:
    # Ajusta k iterativamente para llevar la fraccion de tinta al rango objetivo.
    k = k_init
    thr = np.percentile(gray, 90)
    mask = gray < thr
    if mask.sum() < gray.size * 0.05:
        mask = np.ones_like(gray, dtype=bool)

    for _ in range(max_iter):
        binary = sauvola(gray, window=window, k=k)
        ink_ratio = float((binary[mask] < 128).mean())

        if ink_ratio > target_ink[1]:
            k *= 1.05
        elif ink_ratio < target_ink[0]:
            k *= 0.95
        else:
            break

    return float(np.clip(k, 0.12, 0.30))


def enhance_contrast(
    gray:       np.ndarray,
    clip_limit: float = 2.5,
    tile_size:  int   = 16,
) -> np.ndarray:
    # CLAHE para realzar contraste local en zonas oscuras.
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def bilateral_denoise(
    gray:        np.ndarray,
    diameter:    int   = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    # Denoising bilateral: suaviza fondos preservando bordes de letras.
    return cv2.bilateralFilter(gray, diameter, sigma_color, sigma_space)


def normalize_illumination(
    gray:        np.ndarray,
    kernel_size: int = 0,
) -> np.ndarray:
    # Compensa iluminacion no uniforme dividiendo por el fondo estimado.
    H, W = gray.shape
    if kernel_size <= 0:
        kernel_size = max(25, min(H, W) // 15)
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    bg_f       = np.maximum(background.astype(np.float32), 1.0)
    normalized = np.clip(gray.astype(np.float32) / bg_f * 255.0, 0, 255).astype(np.uint8)
    return normalized


def _background_uniformity(gray: np.ndarray) -> float:
    # Mide cuanto varia el percentil 90 entre 16 sub-tiles (proxy de uniformidad de fondo).
    H, W = gray.shape
    if H < 32 or W < 32:
        return float(gray.std())
    grid = 4
    bh, bw = H // grid, W // grid
    bg_samples = []
    for gy in range(grid):
        for gx in range(grid):
            tile = gray[gy*bh:(gy+1)*bh, gx*bw:(gx+1)*bw]
            if tile.size:
                bg_samples.append(float(np.percentile(tile, 90)))
    return float(np.std(bg_samples))


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
    method:           str   = "auto",
) -> np.ndarray:
    # Orquestador: pre-proceso opcional, eleccion otsu/sauvola y binarizacion.
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if use_remove_bg:
        gray = normalize_illumination(gray, kernel_size=remove_bg_kernel)

    if use_bilateral:
        gray = bilateral_denoise(gray, diameter=bilateral_d,
                                 sigma_color=bilateral_sc, sigma_space=bilateral_ss)

    if use_clahe:
        gray = enhance_contrast(gray, clip_limit=clahe_clip, tile_size=clahe_tile)

    chosen = method
    if chosen == "auto":
        bg_std = _background_uniformity(gray)
        chosen = "otsu" if bg_std < 12.0 else "sauvola"

    if chosen == "otsu":
        gray_for_otsu = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.5, sigmaY=0.5)
        _, binary = cv2.threshold(gray_for_otsu, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # global_floor_pct sigue funcionando si el caller lo pidio
        if global_floor_pct > 0.0:
            floor_val = float(np.percentile(gray, global_floor_pct))
            otsu_thr  = (float(np.unique(gray_for_otsu[binary == 255]).min())
                         if (binary == 255).any() else 128.0)
            if floor_val > otsu_thr + 10:
                binary[gray >= floor_val] = 255
    else:
        k      = auto_tune_sauvola_k(gray, window=window, k_init=k)
        binary = sauvola(gray, window=window, k=k, r=r,
                         global_floor_pct=global_floor_pct)

    if invert:
        binary = 255 - binary

    return binary


def clean_binary(
    binary:      np.ndarray,
    morph_open:  int = 0,
    morph_close: int = 0,
) -> np.ndarray:
    # Apertura/clausura morfologica para eliminar ruido puntual o cerrar trazos.
    result = binary.copy()
    if morph_open > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_open, morph_open))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    if morph_close > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_close, morph_close))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    return result


def filter_small_components(
    binary:        np.ndarray,
    min_area_px:   int,
    max_area_frac: float = 0.10,
) -> np.ndarray:
    # Conserva componentes con area entre min_area_px y max_area_frac*HW.
    H, W     = binary.shape
    max_area = int(H * W * max_area_frac)
    fg = (binary < 128).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    result = np.full_like(binary, 255)
    for label_id in range(1, n_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if min_area_px <= area <= max_area:
            result[labels == label_id] = 0
    return result


def _noise_gap_threshold(areas: np.ndarray) -> int:
    # Detecta el salto entre ruido y trazos reales en la distribucion de areas.
    GAP_RATIO       = 2.5
    MAX_NOISE_AREA  = 50
    MIN_NOISE_COUNT = 5

    sorted_areas = np.sort(areas)
    for i in range(len(sorted_areas) - 1):
        a0 = int(sorted_areas[i])
        a1 = int(sorted_areas[i + 1])
        if a0 > MAX_NOISE_AREA:
            break
        if a0 >= 1 and a1 >= a0 * GAP_RATIO:
            noise_count = i + 1
            if noise_count >= MIN_NOISE_COUNT:
                return a0
    return 0


def adaptive_filter_components(
    binary:        np.ndarray,
    max_area_frac: float = 0.10,
) -> np.ndarray:
    # Filtra componentes pequenos usando umbral adaptativo (gap natural).
    H, W     = binary.shape
    max_area = int(H * W * max_area_frac)
    fg = (binary < 128).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)

    if n_labels < 3:
        return binary

    areas = np.array(
        [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)], dtype=np.int64
    )
    noise_thresh = _noise_gap_threshold(areas)

    result = np.full_like(binary, 255)
    for label_id in range(1, n_labels):
        area = int(areas[label_id - 1])
        if area <= noise_thresh or area > max_area:
            continue
        result[labels == label_id] = 0

    return result


def mask_binding_strips(
    binary:        np.ndarray,
    max_frac:      float = 0.15,
    density_thr:   float = 0.30,
) -> tuple[np.ndarray, int, int]:
    # Enmascara franjas oscuras laterales (encuadernacion, borde de libro).
    H, W = binary.shape
    if W < 20:
        return binary, 0, W

    ink_col = (binary < 128).mean(axis=0)
    max_x   = max(1, int(W * max_frac))
    sec_thr = density_thr * 0.7   # umbral secundario para extension

    # Lado izquierdo
    left_band = ink_col[:max_x] >= density_thr
    if left_band.any():
        rightmost = int(np.where(left_band)[0].max())
        # Extension hacia el interior con umbral secundario
        gap = 0
        i   = rightmost + 1
        while i < max_x and gap <= 2:
            if ink_col[i] >= sec_thr:
                rightmost = i
                gap = 0
            else:
                gap += 1
            i += 1
        x_left_safe = rightmost + 1
    else:
        x_left_safe = 0

    # Lado derecho (simetrico)
    right_band = ink_col[W - max_x:] >= density_thr
    if right_band.any():
        leftmost_global = (W - max_x) + int(np.where(right_band)[0].min())
        gap = 0
        i   = leftmost_global - 1
        while i >= W - max_x and gap <= 2:
            if ink_col[i] >= sec_thr:
                leftmost_global = i
                gap = 0
            else:
                gap += 1
            i -= 1
        x_right_safe = leftmost_global
    else:
        x_right_safe = W

    if x_left_safe == 0 and x_right_safe == W:
        return binary, 0, W

    out = binary.copy()
    if x_left_safe > 0:
        out[:, :x_left_safe] = 255
    if x_right_safe < W:
        out[:, x_right_safe:] = 255

    return out, x_left_safe, x_right_safe


def trim_orphan_components(
    strip_binary: np.ndarray,
    band_thr:     float = 0.15,
    y_tol_frac:   float = 0.30,
) -> np.ndarray:
    # Elimina componentes huerfanos: tinta de lineas vecinas en el padding.
    H, W = strip_binary.shape
    if H < 4 or W < 4:
        return strip_binary

    fg = (strip_binary < 128).astype(np.uint8)
    if fg.sum() == 0:
        return strip_binary

    row_ink = fg.sum(axis=1).astype(np.float64)
    pk      = float(row_ink.max())
    if pk <= 0.0:
        return strip_binary

    band_rows = np.where(row_ink >= pk * band_thr)[0]
    if len(band_rows) < 2:
        return strip_binary

    runs: list[tuple[int, int]] = []
    cur_start = int(band_rows[0])
    prev      = int(band_rows[0])
    for r in band_rows[1:]:
        ri = int(r)
        if ri - prev <= 1:
            prev = ri
            continue
        runs.append((cur_start, prev + 1))
        cur_start = ri
        prev      = ri
    runs.append((cur_start, prev + 1))
    runs.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
    main_top, main_bot = runs[0]
    if (main_bot - main_top) < 2:
        return strip_binary
    band_h = main_bot - main_top
    tol    = max(2, int(band_h * y_tol_frac))
    y_min  = main_top - tol
    y_max  = main_bot + tol

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n_labels < 2:
        return strip_binary

    edge_band = max(2, int(W * 0.05))

    main_band_cxs: list[float] = []
    for label_id in range(1, n_labels):
        cy = float(centroids[label_id, 1])
        if y_min <= cy <= y_max:
            x, _, w, _, _ = stats[label_id]
            if w > 4:   # solo letras anchas definen el cuerpo del texto
                main_band_cxs.append(float(centroids[label_id, 0]))

    keep_mask = np.zeros(n_labels, dtype=bool)
    keep_mask[0] = True   # fondo
    for label_id in range(1, n_labels):
        cy = float(centroids[label_id, 1])
        if not (y_min <= cy <= y_max):
            continue
        x, y, w, h, _ = stats[label_id]
        x_right    = x + w
        is_thin    = (w <= 4)
        hugs_edge  = (x < edge_band) or (x_right > W - edge_band)
        if is_thin and hugs_edge and main_band_cxs:
            cx       = float(centroids[label_id, 0])
            text_min = min(main_band_cxs)
            text_max = max(main_band_cxs)
            isolated = (cx < text_min - W * 0.04) or (cx > text_max + W * 0.04)
            if isolated:
                continue
        keep_mask[label_id] = True

    if keep_mask.sum() <= 1:
        # Nada que conservar (muy raro): no tocar para evitar borrar todo
        return strip_binary

    keep_pixels = keep_mask[labels]
    result = np.full_like(strip_binary, 255)
    result[keep_pixels & (fg > 0)] = 0
    return result
