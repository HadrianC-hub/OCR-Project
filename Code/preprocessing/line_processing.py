import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import savgol_filter, find_peaks

from preprocessing.config import (
    PipelineConfig, TARGET_HEIGHT, PAD_VALUE, MIN_WIDTH
)


# 1. NORMALIZACIÓN

def trim_vertical(binary: np.ndarray, margin: int = 2) -> np.ndarray:
    row_min   = binary.min(axis=1)
    text_rows = np.where(row_min < 128)[0]
    if len(text_rows) == 0:
        return binary

    text_h         = text_rows[-1] - text_rows[0]
    dynamic_margin = max(margin, int(text_h * 0.10))
    top            = max(0, text_rows[0] - dynamic_margin)
    bottom         = min(binary.shape[0], text_rows[-1] + dynamic_margin + 1)
    return binary[top:bottom, :]


def resize_to_height(img: np.ndarray, target_height: int = 64, min_width: int = 16) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_height, min_width), 255, dtype=img.dtype)
    scale = target_height / h
    new_w = max(min_width, int(w * scale))
    return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)


def to_float(img: np.ndarray, invert: bool = False) -> np.ndarray:
    """uint8 [0,255] → float32 [0,1]. Convención: 0=tinta, 1=fondo."""
    f = img.astype(np.float32) / 255.0
    return 1.0 - f if invert else f


def normalize_line(
    binary:        np.ndarray,
    target_height: int = TARGET_HEIGHT,
    trim_margin:   int = 2,
    min_width:     int = MIN_WIDTH,
) -> np.ndarray:
    """trim vertical → resize a altura fija → float32 [0,1]."""
    return to_float(resize_to_height(
        trim_vertical(binary, margin=trim_margin),
        target_height=target_height,
        min_width=min_width,
    ))


# 2. DETECCIÓN DE LÍNEAS

def _split_oversized_boxes(
    boxes:           list[tuple[int, int]],
    projection:      np.ndarray,
    median_h:        float,
    min_line_height: int,
) -> list[tuple[int, int]]:
    """Parte cajas >1.6× la mediana buscando el mínimo de proyección en el 50% central."""
    if len(boxes) < 2 or median_h <= 0:
        return boxes
    result: list[tuple[int, int]] = []
    for (t, b) in boxes:
        box_h = b - t
        if box_h > median_h * 1.6 and box_h >= min_line_height * 2:
            quarter = max(1, box_h // 4)
            st, sb  = t + quarter, b - quarter
            if sb - st >= min_line_height:
                split_y = st + int(np.argmin(projection[st:sb]))
                if split_y - t >= min_line_height and b - split_y >= min_line_height:
                    result.append((t, split_y))
                    result.append((split_y, b))
                    continue
        result.append((t, b))
    return result


def detect_lines(
    binary: np.ndarray,
    cfg:    PipelineConfig,
) -> list[tuple[int, int]]:
    """
    Detecta líneas por proyección horizontal con pre-dilatación morfológica.
    Devuelve lista de (y_top, y_bot) ordenada de arriba a abajo, sin solapamientos.
    """
    H_img, W_img = binary.shape

    if cfg.use_clahe or cfg.invert_binary:
        kernel         = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        binary_for_seg = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    else:
        binary_for_seg = binary

    text_mask = (binary_for_seg < 128).astype(np.uint8)

    # Dilatación horizontal: une palabras de la misma línea.
    # Dilatación vertical: conecta el acento de una mayúscula (Á, É…) con el
    # cuerpo de la letra en la proyección, evitando que Otsu lo clasifique como
    # fila inactiva. El hueco acento↔cuerpo es ~8-15% de la altura de línea;
    # usar el 10% del alto del crop como mínimo dinámico garantiza la conexión
    # sin fundir líneas adyacentes (cuyo espacio es ~25-40%).
    h_dil = cfg.line_h_dilation if cfg.line_h_dilation > 0 else max(5, min(150, W_img // 18))
    v_dil = cfg.line_v_dilation if cfg.line_v_dilation > 0 else max(3, min(15, H_img // 20))
    if h_dil > 1 or v_dil > 1:
        kernel_pre     = cv2.getStructuringElement(cv2.MORPH_RECT, (h_dil, v_dil))
        text_mask_proj = cv2.dilate(text_mask, kernel_pre).astype(np.float32)
    else:
        text_mask_proj = text_mask.astype(np.float32)

    # Proyección horizontal (excluye márgenes laterales ~3%)
    margin_px  = max(15, W_img // 35)
    projection = text_mask_proj[:, margin_px: W_img - margin_px].sum(axis=1)

    smooth = max(1, cfg.projection_smooth)
    if getattr(cfg, 'use_savgol', False) and smooth > 1:
        wl = cfg.savgol_window if cfg.savgol_window % 2 == 1 else cfg.savgol_window + 1
        wl = max(3, min(wl, len(projection) - (0 if len(projection) % 2 == 1 else 1)))
        if wl >= 5:
            projection = np.clip(
                savgol_filter(projection.astype(np.float64), wl, cfg.savgol_polyorder), 0, None
            )
    elif smooth > 1:
        projection = uniform_filter(projection.astype(np.float64), size=smooth)

    proj_nonzero = projection[projection > 0]
    if len(proj_nonzero) == 0:
        return []

    # Umbral Otsu adaptado a la proyección 1-D, con cap al 40% del pico
    p_max       = float(proj_nonzero.max())
    proj_u8     = np.clip(projection / p_max * 255, 0, 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(proj_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_thresh = min(float(otsu_val) / 255.0 * p_max, p_max * 0.40)
    if otsu_thresh < 1.0:
        otsu_thresh = max(1.0, W_img * 0.008)

    active  = projection > otsu_thresh
    boxes: list[tuple[int, int]] = []
    in_line = False
    y_start = 0
    for y, is_active in enumerate(active):
        if is_active and not in_line:
            y_start = y
            in_line = True
        elif not is_active and in_line:
            boxes.append((y_start, y))
            in_line = False
    if in_line:
        boxes.append((y_start, len(active)))

    # Fusión de segmentos próximos
    merge_gap = cfg.line_merge_gap
    if merge_gap > 0 and len(boxes) > 1:
        merged = [boxes[0]]
        for (t, b) in boxes[1:]:
            gap = t - merged[-1][1]
            if gap <= merge_gap:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((t, b))
        boxes = merged

    # Fallback por bisección de picos si la segmentación primaria es pobre
    if H_img > 250 and (
        len(boxes) <= 1 or
        (H_img / max(1, len(boxes)) > H_img * 0.25 and len(boxes) < 4)
    ):
        smooth_s = uniform_filter(projection.astype(np.float64), size=max(5, smooth * 3))
        peaks, _ = find_peaks(
            smooth_s,
            distance=max(5, H_img // 50),
            prominence=float(smooth_s.max()) * 0.10,
        )
        if len(peaks) >= 2:
            boundaries = (
                [0]
                + [int((peaks[i] + peaks[i + 1]) // 2) for i in range(len(peaks) - 1)]
                + [H_img]
            )
            peak_boxes = list(zip(boundaries[:-1], boundaries[1:]))
            if len(peak_boxes) > len(boxes):
                boxes = peak_boxes

    valid = [
        (t, b) for (t, b) in boxes
        if (b - t) >= cfg.min_line_height
        and int((binary[t:b, :] < 128).any(axis=0).sum()) >= cfg.min_line_width
    ]

    if len(valid) >= 2:
        heights = [b - t for t, b in valid]
        valid   = _split_oversized_boxes(valid, projection, float(np.median(heights)), cfg.min_line_height)

    # Corrección de solapamientos: separa en el mínimo de proyección de la zona solapada
    no_overlap_boxes: list[tuple[int, int]] = []
    for i, (yt, yb) in enumerate(valid):
        if i == 0:
            no_overlap_boxes.append((yt, yb))
        else:
            prev_yt, prev_yb = no_overlap_boxes[-1]
            if yt < prev_yb and yt > prev_yt:
                overlap_region = projection[yt:prev_yb]
                if len(overlap_region) > 0:
                    split_y = int(np.argmin(overlap_region) + prev_yb)
                    if (split_y - prev_yt >= cfg.min_line_height and
                            yb - split_y >= cfg.min_line_height):
                        no_overlap_boxes[-1] = (prev_yt, split_y)
                        no_overlap_boxes.append((split_y, yb))
                    else:
                        no_overlap_boxes.append((max(prev_yb, yt), yb))
                else:
                    no_overlap_boxes.append((max(prev_yb, yt), yb))
            else:
                no_overlap_boxes.append((yt, yb))

    # Padding superior para capturar acentos de mayúsculas (Á, É, Í…).
    # El hueco entre acento y cuerpo es típicamente 10-20% de la altura de línea.
    if no_overlap_boxes:
        median_line_h = float(np.median([b - t for t, b in no_overlap_boxes]))
    else:
        median_line_h = float(cfg.expand_no_ink_gap * 2)
    top_pad = max(3, int(median_line_h * 0.20))
    padded: list[tuple[int, int]] = []
    for (t, b) in no_overlap_boxes:
        prev_bot = padded[-1][1] if padded else 0
        padded.append((max(prev_bot, t - top_pad), b))

    return padded


# 3. EXPANSIÓN Y ENDEREZADO DE LÍNEAS

def expand_all_boxes(
    binary:          np.ndarray,
    boxes:           list[tuple[int, int, int, int]],
    max_expand_frac: float = 0.80,
    no_ink_gap:      int   = 4,
    min_ink_frac:    float = 0.003,
    block_boxes:     list[tuple[int, int, int, int]] | None = None,
) -> list[tuple[int, int, int, int]]:
    """
    Expande cada caja fila a fila hasta encontrar la primera fila sin tinta
    o colisionar con la línea adyacente de la misma columna.

    `no_ink_gap` tolera N filas vacías consecutivas para saltar el hueco
    entre el cuerpo de una letra y sus diacríticos.
    Si se proporciona `block_boxes`, la expansión respeta los límites Y de bloque.
    """
    from collections import defaultdict

    H_bin  = binary.shape[0]
    result = list(boxes)

    col_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, (_, _, xl, xr) in enumerate(boxes):
        col_groups[(xl, xr)].append(idx)

    block_limits: dict[tuple[int, int], tuple[int, int]] = {}
    if block_boxes:
        for block_yt, block_yb, block_xl, block_xr in block_boxes:
            key = (block_xl, block_xr)
            if key not in block_limits:
                block_limits[key] = (block_yt, block_yb)
            else:
                prev_yt, prev_yb = block_limits[key]
                block_limits[key] = (min(prev_yt, block_yt), max(prev_yb, block_yb))

    for (xl, xr), indices in col_groups.items():
        indices.sort(key=lambda i: boxes[i][0])
        local        = [boxes[i] for i in indices]
        n            = len(local)
        min_ink_cols = max(2, int((xr - xl) * min_ink_frac))
        block_yt, block_yb = block_limits.get((xl, xr), (0, H_bin))

        def longest_run_bool(arr: np.ndarray) -> int:
            if arr.size == 0:
                return 0
            max_run = cur = 0
            for v in arr:
                if v:
                    cur += 1
                    if cur > max_run:
                        max_run = cur
                else:
                    cur = 0
            return max_run

        def row_has_ink(y: int) -> bool:
            if not (0 <= y < H_bin):
                return False
            row = (binary[y, xl:xr] < 128)
            cnt = int(row.sum())
            if cnt < min_ink_cols:
                return False
            return longest_run_bool(row) >= max(1, min(min_ink_cols // 3, 2))

        for li, orig_idx in enumerate(indices):
            yt, yb, _, _ = local[li]
            limit_top = (local[li - 1][1] + yt) // 2 if li > 0     else block_yt
            limit_bot = (yb + local[li + 1][0]) // 2 if li < n - 1 else block_yb

            # Expansión hacia arriba: busca la fila con tinta más alta en [limit_top, yt)
            new_top = yt
            for y in range(yt - 1, limit_top - 1, -1):
                if row_has_ink(y):
                    new_top = y

            # Expansión hacia abajo con tolerancia de gap
            new_bot      = yb
            gap_count    = 0
            last_ink_row = yb - 1
            for y in range(yb, limit_bot):
                if row_has_ink(y):
                    last_ink_row = y
                    gap_count    = 0
                else:
                    gap_count += 1
                    if gap_count > no_ink_gap:
                        break
            new_bot = last_ink_row + 1

            result[orig_idx] = (new_top, new_bot, xl, xr)

    # Resolución de solapamientos post-expansión
    for (xl, xr), indices in col_groups.items():
        indices.sort(key=lambda i: result[i][0])
        for li in range(len(indices) - 1):
            idx_curr = indices[li]
            idx_next = indices[li + 1]
            yt_curr, yb_curr, _, _ = result[idx_curr]
            yt_next, yb_next, _, _ = result[idx_next]
            if yb_curr > yt_next:
                gap_orig_curr = boxes[idx_curr][1]
                gap_orig_next = boxes[idx_next][0]
                gap_between   = gap_orig_next - gap_orig_curr
                if gap_between > 0:
                    midpoint = (gap_orig_curr + gap_orig_next) // 2
                    if yb_curr > midpoint:
                        result[idx_curr] = (yt_curr, midpoint, xl, xr)
                    if yt_next < midpoint:
                        result[idx_next] = (midpoint, yb_next, xl, xr)
                else:
                    mid_overlap = (yt_curr + yb_curr + yt_next + yb_next) // 4
                    result[idx_curr] = (yt_curr, mid_overlap, xl, xr)
                    result[idx_next] = (mid_overlap, yb_next, xl, xr)

    return result


def straighten_line(
    binary:         np.ndarray,
    poly_degree:    int   = 2,
    n_slices:       int   = 0,
    min_ink_frac:   float = 0.030,
    max_shift_frac: float = 0.20,
) -> np.ndarray:
    """
    Corrige curvatura de una franja de línea por ajuste polinomial del centroide Y
    de la tinta en rebanadas verticales.

    Devuelve la franja original si el ajuste no supera los controles de calidad
    (R² < 0.72, shift excesivo o pocos puntos válidos).
    """
    H, W = binary.shape
    if H < 4 or W < 20:
        return binary

    if n_slices <= 0:
        n_slices = max(5, min(40, W // 80))

    slice_w = max(1, W // n_slices)
    row_idx = np.arange(H, dtype=np.float64)
    xs: list[float] = []
    ys: list[float] = []

    for i in range(n_slices):
        x0, x1   = i * slice_w, min((i + 1) * slice_w, W)
        ink       = (binary[:, x0:x1] < 128).astype(np.float32)
        total_ink = float(ink.sum())
        if total_ink < (x1 - x0) * H * min_ink_frac:
            continue
        ink_per_row = ink.sum(axis=1).astype(np.float64)
        xs.append((x0 + x1) / 2.0)
        ys.append(float((row_idx * ink_per_row).sum() / ink_per_row.sum()))

    if len(xs) < poly_degree + 2:
        return binary

    xs_arr = np.array(xs, dtype=np.float64)
    ys_arr = np.array(ys, dtype=np.float64)

    if len(ys_arr) >= 4:
        q1, q3 = np.percentile(ys_arr, [25, 75])
        fence   = max((q3 - q1) * 1.5, 4.0)
        med     = float(np.median(ys_arr))
        mask    = np.abs(ys_arr - med) <= fence
        if mask.sum() >= poly_degree + 2:
            xs_arr, ys_arr = xs_arr[mask], ys_arr[mask]
        else:
            return binary

    if float(ys_arr.var()) < 1.0:
        return binary

    eff_deg = poly_degree if len(xs_arr) >= poly_degree + 3 else 1
    try:
        coeffs  = np.polyfit(xs_arr, ys_arr, eff_deg)
        poly_fn = np.poly1d(coeffs)
    except np.linalg.LinAlgError:
        return binary

    ss_res = float(np.sum((ys_arr - poly_fn(xs_arr)) ** 2))
    ss_tot = float(np.sum((ys_arr - float(ys_arr.mean())) ** 2))
    if ss_tot > 1e-9 and (1.0 - ss_res / ss_tot) < 0.72:
        return binary

    col_arr      = np.arange(W, dtype=np.float64)
    col_baseline = poly_fn(col_arr)

    x_min, x_max = float(xs_arr.min()), float(xs_arr.max())
    col_baseline = np.where(
        col_arr < x_min, float(poly_fn(x_min)),
        np.where(col_arr > x_max, float(poly_fn(x_max)), col_baseline),
    )

    shifts = col_baseline - float(H) / 2.0
    if float(np.abs(shifts).max()) > float(H) * max_shift_frac:
        return binary

    pad_top = max(0, int(np.ceil(-float(shifts.min()))))
    pad_bot = max(0, int(np.ceil( float(shifts.max()))))
    src     = np.pad(binary, ((pad_top, pad_bot), (0, 0)),
                     mode='constant', constant_values=255) if (pad_top or pad_bot) else binary

    map_x    = np.tile(np.arange(W, dtype=np.float32), (H, 1))
    row_grid = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))
    map_y    = (row_grid + shifts[np.newaxis, :] + pad_top).astype(np.float32)
    result   = cv2.remap(src, map_x, map_y,
                         interpolation=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    rows_before = int((binary < 128).any(axis=1).sum())
    rows_after  = int((result < 128).any(axis=1).sum())
    if rows_before > 0 and rows_after < rows_before * 0.40:
        return binary

    rows_with_ink = np.where((result < 128).any(axis=1))[0]
    if len(rows_with_ink) > 0:
        if int(rows_with_ink[-1]) - int(rows_with_ink[0]) + 1 < H * 0.30:
            return binary

    def _norm_width(img: np.ndarray) -> int:
        r = np.where((img < 128).any(axis=1))[0]
        if len(r) == 0:
            return 1
        return max(16, int(round(img.shape[1] * TARGET_HEIGHT / max(1, int(r[-1]) - int(r[0]) + 1))))

    if _norm_width(result) > _norm_width(binary) * 1.8:
        return binary

    return result


def rotate_strip_by_baseline(
    strip:         np.ndarray,
    n_slices:      int   = 0,
    min_ink_frac:  float = 0.003,
    max_angle_deg: float = 20.0,
) -> tuple[np.ndarray, float]:
    """
    Estima el ángulo medio de la línea por centroides en rebanadas verticales
    y devuelve una versión rotada del strip alineada horizontalmente.

    Retorna (rotated_strip, angle_deg). Si no se aplica rotación, angle_deg=0.0.
    """
    H, W = strip.shape[:2]
    if H < 4 or W < 8:
        return strip, 0.0

    if n_slices <= 0:
        n_slices = max(5, min(40, W // 80))
    slice_w = max(1, W // n_slices)

    xs, ys  = [], []
    row_idx = np.arange(H, dtype=np.float64)
    for i in range(n_slices):
        x0, x1 = i * slice_w, min((i + 1) * slice_w, W)
        ink = (strip[:, x0:x1] < 128).astype(np.float32)
        if ink.sum() < (x1 - x0) * H * min_ink_frac:
            continue
        ink_per_row = ink.sum(axis=1).astype(np.float64)
        if ink_per_row.sum() <= 0:
            continue
        xs.append((x0 + x1) / 2.0)
        ys.append(float((row_idx * ink_per_row).sum() / ink_per_row.sum()))

    if len(xs) < 2:
        return strip, 0.0

    xs_arr = np.array(xs, dtype=np.float64)
    ys_arr = np.array(ys, dtype=np.float64)

    try:
        coeffs = np.polyfit(xs_arr, ys_arr, 1)
    except np.linalg.LinAlgError:
        return strip, 0.0

    # Rechazo de outliers IQR
    if len(ys_arr) >= 4:
        q1, q3 = np.percentile(ys_arr, [25, 75])
        fence   = max((q3 - q1) * 1.5, 4.0)
        med     = float(np.median(ys_arr))
        mask    = np.abs(ys_arr - med) <= fence
        if mask.sum() >= 2:
            xs_arr, ys_arr = xs_arr[mask], ys_arr[mask]
            try:
                coeffs = np.polyfit(xs_arr, ys_arr, 1)
            except np.linalg.LinAlgError:
                return strip, 0.0
        else:
            return strip, 0.0

    # Guardia R²: umbral 0.60 (ajuste lineal grado 1; más permisivo que el 0.72
    # de straighten_line que usa polinomio de mayor grado)
    predicted = coeffs[0] * xs_arr + coeffs[1]
    ss_res    = float(np.sum((ys_arr - predicted) ** 2))
    ss_tot    = float(np.sum((ys_arr - float(ys_arr.mean())) ** 2))
    if ss_tot > 1e-9 and (1.0 - ss_res / ss_tot) < 0.60:
        return strip, 0.0

    slope     = float(coeffs[0])
    angle_deg = float(np.degrees(np.arctan(slope)))

    if abs(angle_deg) < 0.3 or abs(angle_deg) > max_angle_deg:
        return strip, 0.0

    cx, cy = W / 2.0, H / 2.0
    M      = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)

    cos_a = np.abs(M[0, 0])
    sin_a = np.abs(M[0, 1])
    new_W = int((H * sin_a) + (W * cos_a))
    new_H = int((H * cos_a) + (W * sin_a))

    M[0, 2] += (new_W / 2.0) - cx
    M[1, 2] += (new_H / 2.0) - cy

    rotated = cv2.warpAffine(strip, M, (new_W, new_H), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return rotated, angle_deg