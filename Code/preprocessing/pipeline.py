from pathlib import Path
from typing  import Optional

import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal  import find_peaks

from preprocessing.binarization    import (
    binarize, clean_binary, adaptive_filter_components, filter_small_components
)
from preprocessing.config          import (
    ImageMetrics, PipelineConfig, PipelineResult
)
from preprocessing.line_processing import (
    detect_lines, expand_all_boxes, normalize_line, straighten_line
)


# 1. ANÁLISIS DE IMAGEN Y AUTO-CONFIGURACIÓN

def analyze(img: np.ndarray) -> ImageMetrics:
    """Analiza la imagen y devuelve métricas para auto-configuración."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    H, W = gray.shape

    p5, p95  = float(np.percentile(gray, 5)), float(np.percentile(gray, 95))
    contrast = p95 - p5
    mean_lum = float(gray.mean())

    margin  = max(10, min(H, W) // 12)
    corners = np.concatenate([
        gray[:margin, :margin].ravel(),   gray[:margin, W - margin:].ravel(),
        gray[H - margin:, :margin].ravel(), gray[H - margin:, W - margin:].ravel(),
    ])
    dark_background = float(corners.mean()) < 127

    best_channel = "gray"
    if img.ndim == 3:
        stds = {
            "blue":  float(img[:, :, 0].std()),
            "green": float(img[:, :, 1].std()),
            "red":   float(img[:, :, 2].std()),
            "gray":  float(gray.std()),
        }
        best = max(stds, key=stds.get)
        best_channel = best if stds[best] >= stds["gray"] * 1.20 else "gray"

    src = gray if not dark_background else (255 - gray)
    _, rough     = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj_smooth  = uniform_filter((rough.sum(axis=1) / 255.0).astype(np.float64),
                                  size=max(9, H // 60))
    p_max_smooth = float(proj_smooth.max())

    if p_max_smooth > 0:
        peaks, _    = find_peaks(proj_smooth, height=p_max_smooth * 0.08,
                                 distance=max(8, H // 55))
        n_lines_est = max(1, len(peaks))
        estimated_text_h = int((proj_smooth > p_max_smooth * 0.10).sum()) / n_lines_est
    else:
        n_lines_est      = 1
        estimated_text_h = H / 5.0

    needs_clahe = contrast < 80

    bg_mask = rough == 0
    if bg_mask.sum() > 100:
        sx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        bg_noise = float(np.sqrt(sx ** 2 + sy ** 2)[bg_mask].mean())
    else:
        bg_noise = 0.0
    needs_bilateral = (bg_noise / max(1.0, contrast)) > 0.08

    return ImageMetrics(
        contrast=contrast, mean_luminance=mean_lum,
        dark_background=dark_background,
        estimated_text_h=estimated_text_h, estimated_n_lines=n_lines_est,
        needs_clahe=needs_clahe, needs_bilateral=needs_bilateral,
        best_channel=best_channel, H=H, W=W,
    )


def auto_config(img: np.ndarray) -> PipelineConfig:
    """Devuelve PipelineConfig optimizado para la imagen dada."""
    m = analyze(img)

    use_clahe    = m.needs_clahe
    clahe_clip   = 2.0 if m.contrast < 50 else 2.5
    clahe_tile   = max(8, min(32, m.W // 80))
    use_bilateral = (m.needs_bilateral or use_clahe) and m.contrast < 200
    bilateral_sc  = 90.0 if m.contrast < 50 else 75.0

    max_win        = max(15, min(m.H, m.W) // 2)
    raw_win        = max(15, min(201, int(m.estimated_text_h * 1.5), max_win))
    sauvola_window = raw_win + (0 if raw_win % 2 == 1 else 1)
    sauvola_k      = 0.10 if m.contrast < 50 else 0.17 if m.contrast < 80 else 0.25
    global_floor_pct = 93.0 if m.contrast < 50 else 92.0 if m.contrast < 80 else 0.0

    line_merge_gap    = max(2, min(8, int(m.estimated_text_h * 0.04)))  # reduce merge gap for inclined text
    projection_smooth = max(5, min(15, int(m.estimated_text_h * 0.12)))  # increase smoothing to reduce noise
    expand_no_ink_gap = max(8, int(m.estimated_text_h * 0.50))

    return PipelineConfig(
        sauvola_window=sauvola_window, sauvola_k=sauvola_k,
        morph_open=2 if m.contrast < 50 else 0,
        use_clahe=use_clahe, clahe_clip=clahe_clip, clahe_tile=clahe_tile,
        use_blue_channel=(m.best_channel == "blue"),
        use_bilateral=use_bilateral, bilateral_sc=bilateral_sc, bilateral_ss=bilateral_sc,
        invert_binary=m.dark_background,
        global_floor_pct=global_floor_pct, min_component_area=0,
        use_adaptive_component_filter=True,
        use_remove_bg=(m.contrast < 70),
        line_merge_gap=line_merge_gap, projection_smooth=projection_smooth,
        expand_no_ink_gap=expand_no_ink_gap,
        line_h_dilation=0, line_v_dilation=2,
        detect_text_blocks=True, expand_to_ink=True, straighten_lines=True,
        block_h_thr_frac=0.02, block_min_h_gap=80,
        deskew_blocks=True, deskew_block_max_angle=10.0,
    )


# 2. DETECCIÓN DE BLOQUES DE TEXTO (XY-Cut de 3 niveles)

def _find_h_separators(binary: np.ndarray, min_gap_h: int = 0,
                        threshold_frac: float = 0.04) -> list[int]:
    """Detecta bandas horizontales vacías entre secciones."""
    H, W    = binary.shape
    if min_gap_h <= 0:
        min_gap_h = max(100, H // 30)
    h_proj  = (binary < 128).astype(np.float64).sum(axis=1)
    h_sm    = uniform_filter(h_proj, size=max(3, H // 200))
    h_max   = float(h_sm.max())
    if h_max == 0:
        return []
    blank   = h_sm < h_max * threshold_frac
    seps    = []
    in_b    = False
    b_start = 0
    for y in range(H):
        if blank[y] and not in_b:
            b_start, in_b = y, True
        elif not blank[y] and in_b:
            if y - b_start >= min_gap_h:
                seps.append((b_start + y) // 2)
            in_b = False
    if in_b and H - b_start >= min_gap_h:
        seps.append((b_start + H) // 2)
    return seps


def _find_column_separators(
    binary: np.ndarray, min_gap_width: int = 0,
    max_n_cols: int = 4, smooth_size: int = 0,
    min_depth: float = 0.28,
) -> list[int]:
    """Detecta separadores verticales de columna por análisis de proyección."""
    H, W          = binary.shape
    smooth_size   = smooth_size   or max(20, W // 20)
    min_gap_width = min_gap_width or max(10, W // 50)
    min_col_dist  = max(W // 4, 80)

    v_proj = (binary < 128).astype(np.float64).sum(axis=0)
    v_sm   = uniform_filter(v_proj, size=smooth_size)
    v_max  = float(v_sm.max())
    if v_max == 0:
        return []

    peaks, _ = find_peaks(v_sm, height=v_max * 0.10, distance=min_col_dist)
    if len(peaks) < 2:
        return []

    total_ink  = float(v_sm.sum())
    separators = []
    for i in range(len(peaks) - 1):
        vi  = int(np.argmin(v_sm[peaks[i]: peaks[i + 1] + 1])) + peaks[i]
        vv  = float(v_sm[vi])
        lp  = min(float(v_sm[peaks[i]]), float(v_sm[peaks[i + 1]]))
        depth = (lp - vv) / lp if lp > 0 else 0.0
        if depth < min_depth or not (W * 0.15 < vi < W * 0.85):
            continue
        # Verificar anchura del valle
        gL, gR   = vi, vi
        half_thr = lp * (1 - min_depth * 0.5)
        while gL > 0     and v_sm[gL - 1] <= half_thr: gL -= 1
        while gR < W - 1 and v_sm[gR + 1] <= half_thr: gR += 1
        if (gR - gL) < max(20, W // 35):
            continue
        if total_ink > 0:
            if v_sm[:vi].sum() / total_ink < 0.20 or v_sm[vi:].sum() / total_ink < 0.20:
                continue
        separators.append(vi)

    if len(separators) > max_n_cols - 1:
        def _depth(vi):
            near = sorted(peaks, key=lambda p: abs(p - vi))[:2]
            lp   = min(float(v_sm[near[0]]), float(v_sm[near[1]])) if len(near) >= 2 else 1.0
            return (lp - float(v_sm[vi])) / lp if lp > 0 else 0.0
        separators = sorted(
            [vi for _, vi in sorted([(_depth(vi), vi) for vi in separators], reverse=True)[: max_n_cols - 1]]
        )
    return separators


def _col_x_boundaries(W: int, separators: list[int], margin: int = 0) -> list[tuple[int, int]]:
    """Convierte separadores en rangos (x_left, x_right) por columna."""
    margin = margin or max(3, W // 150)
    boundaries = [0] + separators + [W]
    return [
        (max(0, boundaries[i] - (margin if i > 0 else 0)),
         min(W, boundaries[i + 1] + (margin if i < len(boundaries) - 2 else 0)))
        for i in range(len(boundaries) - 1)
    ]


def _group_into_paragraphs(
    line_ys: list[tuple[int, int]], threshold_factor: float = 2.5
) -> list[list[tuple[int, int]]]:
    """Agrupa líneas en párrafos por análisis de huecos relativos."""
    if not line_ys: return []
    if len(line_ys) == 1: return [line_ys]
    gaps      = [line_ys[i + 1][0] - line_ys[i][1] for i in range(len(line_ys) - 1)]
    threshold = threshold_factor * max(float(np.median(gaps)), 1.0)
    groups    = [[line_ys[0]]]
    for i, gap in enumerate(gaps):
        if gap > threshold:
            groups.append([])
        groups[-1].append(line_ys[i + 1])
    return groups


def _merge_close_blocks(
    binary:    np.ndarray,
    blocks:    list[tuple[tuple, list]],
    merge_gap: int,
    margin:    int,
) -> list[tuple[tuple, list]]:
    """
    Fusiona pares de bloques de la misma columna cuyo hueco vertical
    es menor que merge_gap.

    Cada bloque es (block_box, [line_boxes]).
    """
    if len(blocks) < 2:
        return blocks

    H = binary.shape[0]
    changed = True
    while changed:
        changed = False
        merged  = []
        used    = [False] * len(blocks)
        for i in range(len(blocks)):
            if used[i]:
                continue
            bb_i, lines_i = blocks[i]
            best_j, best_gap = -1, merge_gap + 1
            for j in range(i + 1, len(blocks)):
                if used[j]:
                    continue
                bb_j, lines_j = blocks[j]
                # Mismo rango x aproximado
                if abs(bb_i[2] - bb_j[2]) > max(50, (bb_i[3] - bb_i[2]) // 4):
                    continue
                gap = bb_j[0] - bb_i[1] if bb_j[0] >= bb_i[1] else bb_i[0] - bb_j[1]
                if 0 <= gap < best_gap:
                    best_j, best_gap = j, gap
            if best_j >= 0:
                bb_j, lines_j = blocks[best_j]
                new_box = (
                    min(bb_i[0], bb_j[0]), max(bb_i[1], bb_j[1]),
                    min(bb_i[2], bb_j[2]), max(bb_i[3], bb_j[3]),
                )
                new_lines = sorted(lines_i + lines_j, key=lambda b: b[0])
                merged.append((new_box, new_lines))
                used[i] = used[best_j] = True
                changed = True
            else:
                merged.append(blocks[i])
                used[i] = True
        blocks = merged
    return blocks


def segment_all(
    binary: np.ndarray,
    cfg:    PipelineConfig,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """
    Segmentación XY-Cut de 3 niveles:
      1. Separadores horizontales globales → secciones
      2. Separadores verticales por sección → columnas
      3. Sub-separadores dentro de cada columna
      4. detect_lines dentro de cada sub-sección
      5. Agrupación en párrafos, medición x-extent por tinta real

    Retorna (line_boxes, block_boxes) — ambas como (y_top, y_bot, x_left, x_right).
    """
    H, W       = binary.shape
    min_gap_h  = getattr(cfg, 'block_min_h_gap',    0)
    h_thr_frac = getattr(cfg, 'block_h_thr_frac',   0.04)
    max_cols   = getattr(cfg, 'block_max_cols',      4)
    min_depth  = getattr(cfg, 'block_col_min_depth', 0.28)
    min_line_h = getattr(cfg, 'min_line_height',     8)
    para_split = getattr(cfg, 'para_split_factor',   2.5)
    merge_gap  = getattr(cfg, 'block_merge_gap',     0) or max(100, H // 30)

    h_seps     = _find_h_separators(binary, min_gap_h=min_gap_h, threshold_frac=h_thr_frac)
    h_bounds   = [0] + h_seps + [H]
    h_sections = [(h_bounds[i], h_bounds[i + 1])
                  for i in range(len(h_bounds) - 1)
                  if h_bounds[i + 1] - h_bounds[i] >= min_line_h]

    all_blocks: list[tuple[tuple, list]] = []
    min_h_vsplit  = max(500, H // 7)
    min_h_section = max(50,  H // 50)
    margin        = max(4, W // 200)

    for (y0, y1) in h_sections:
        section_h = y1 - y0
        if section_h < min_h_section:
            continue
        section = binary[y0:y1, :]
        if int((section < 128).sum()) < max(50, int(section.size * 0.003)):
            continue

        v_seps     = (_find_column_separators(section, max_n_cols=max_cols, min_depth=min_depth)
                      if section_h >= min_h_vsplit else [])
        col_ranges = _col_x_boundaries(W, v_seps) if v_seps else [(0, W)]

        for (x0, x1) in col_ranges:
            if (x1 - x0) < W // (max(1, len(col_ranges)) * 2):
                continue
            col_crop = binary[y0:y1, x0:x1]
            if col_crop.size == 0 or int((col_crop < 128).sum()) < max(5, int(col_crop.size * 0.001)):
                continue

            col_min_gap = max(150, min(300, section_h // 8))
            col_h_seps   = _find_h_separators(col_crop, min_gap_h=col_min_gap, threshold_frac=h_thr_frac)
            col_h_bounds = [0] + col_h_seps + [section_h]
            sub_sections = [(col_h_bounds[i], col_h_bounds[i + 1])
                            for i in range(len(col_h_bounds) - 1)
                            if col_h_bounds[i + 1] - col_h_bounds[i] >= min_line_h]

            for (sc_y0, sc_y1) in sub_sections:
                abs_y_base = y0 + sc_y0
                sub_cell   = binary[abs_y_base: y0 + sc_y1, x0:x1]
                if sub_cell.size == 0 or int((sub_cell < 128).sum()) < max(5, int(sub_cell.size * 0.001)):
                    continue

                line_ys     = detect_lines(sub_cell, cfg)
                para_groups = _group_into_paragraphs(line_ys, threshold_factor=para_split)

                for para_lines in para_groups:
                    if not para_lines:
                        continue
                    abs_y0 = abs_y_base + para_lines[0][0]
                    abs_y1 = abs_y_base + para_lines[-1][1]

                    y_pad       = max(4, (abs_y1 - abs_y0) // 12)
                    scan_y0     = max(0, abs_y0 - y_pad)
                    scan_y1     = min(H, abs_y1 + y_pad)
                    overflow    = max(margin, W // 40)
                    sx0         = max(0, x0 - margin)
                    sx1         = min(W, x1 + overflow)
                    ink_cols    = np.where((binary[scan_y0:scan_y1, sx0:sx1] < 128).any(axis=0))[0]
                    if len(ink_cols) == 0:
                        continue

                    para_x0   = max(0, sx0 + int(ink_cols[0])  - margin)
                    para_x1   = min(W, sx0 + int(ink_cols[-1]) + 1 + margin)
                    para_cell = binary[abs_y0:abs_y1, para_x0:para_x1]
                    if para_cell.size == 0 or not (para_cell < 128).any():
                        continue

                    block_lines = [(abs_y_base + yt, abs_y_base + yb, para_x0, para_x1)
                                   for (yt, yb) in para_lines]
                    all_blocks.append(((abs_y0, abs_y1, para_x0, para_x1), block_lines))

    all_blocks = _merge_close_blocks(binary, all_blocks, merge_gap, margin)

    block_boxes = [bb for bb, _ in all_blocks]
    all_lines   = [line for (_, lines) in all_blocks for line in lines]
    filtered    = [(yt, yb, xl, xr) for (yt, yb, xl, xr) in all_lines
                   if binary[yt:yb, xl:xr].size > 0 and (binary[yt:yb, xl:xr] < 128).any()]

    hzf      = getattr(cfg, 'header_zone_frac', 1.0 / 6.0)
    HEADER_Y = int(H * hzf) if hzf and hzf > 0 else 0

    h_lines = sorted([(yt, yb, xl, xr) for (yt, yb, xl, xr) in filtered if yt < HEADER_Y], key=lambda b: b[0])
    b_lines = sorted([(yt, yb, xl, xr) for (yt, yb, xl, xr) in filtered if yt >= HEADER_Y], key=lambda b: (b[2], b[0]))
    h_blks  = sorted([(yt, yb, xl, xr) for (yt, yb, xl, xr) in block_boxes if yt < HEADER_Y], key=lambda b: b[0])
    b_blks  = sorted([(yt, yb, xl, xr) for (yt, yb, xl, xr) in block_boxes if yt >= HEADER_Y], key=lambda b: (b[2], b[0]))

    return h_lines + b_lines, h_blks + b_blks


# 3. CONVERSIÓN Y DESKEW

def load_image(path: str | Path) -> np.ndarray:
    """Lee imagen con soporte de rutas no-ASCII en Windows."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return img


def to_gray(img: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """BGR → escala de grises con selección de canal óptimo."""
    if img.ndim == 2:
        return img
    return img[:, :, 0] if cfg.use_blue_channel else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def deskew_image(
    gray:      np.ndarray,
    max_angle: float = 15.0,
) -> tuple[np.ndarray, float]:
    """
    Corrige la inclinación global del documento.

    Usa HoughLinesP con peso longitud² + mediana ponderada + guardia IQR.
    En manuscritos cursivos los trazos oblicuos de letras individuales
    generan segmentos cortos a 5–12°; ponderando por longitud² los baselines
    largos dominan la estimación y se evitan rotaciones espurias.
    """
    h, w = gray.shape
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=100, minLineLength=max(40, w // 5), maxLineGap=20,
    )
    if lines is None or len(lines) == 0:
        return gray, 0.0

    angles, weights = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        if abs(angle) >= max_angle:
            continue
        length = float(np.hypot(dx, dy))
        angles.append(angle)
        weights.append(length * length)

    if not angles:
        return gray, 0.0

    angles_arr  = np.array(angles,  dtype=np.float64)
    weights_arr = np.array(weights, dtype=np.float64)
    weights_arr /= weights_arr.sum()

    sort_idx       = np.argsort(angles_arr)
    sorted_angles  = angles_arr[sort_idx]
    sorted_weights = weights_arr[sort_idx]
    cumulative     = np.cumsum(sorted_weights)

    median_angle = float(sorted_angles[np.searchsorted(cumulative, 0.5)])
    q1_angle     = float(sorted_angles[np.searchsorted(cumulative, 0.25)])
    q3_angle     = float(sorted_angles[np.searchsorted(cumulative, 0.75)])
    if q3_angle - q1_angle > 4.0:
        return gray, 0.0

    if abs(median_angle) < 0.1:
        return gray, median_angle

    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    # Estimar el color de fondo a partir de los píxeles del borde de la imagen.
    # BORDER_REPLICATE puede introducir contenido oscuro del borde del scan
    # (oscurecimiento del papel, artefactos de escáner) como una banda en las
    # esquinas rotadas, que tras la binarización aparece como "tinta" falsa en
    # la primera y última línea de cada bloque. BORDER_CONSTANT con el valor
    # mediano del borde (≈255 para papel blanco) evita completamente ese efecto.
    border_pixels = np.concatenate([
        gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
    ])
    bg_val = int(np.median(border_pixels))
    corrected = cv2.warpAffine(gray, M, (w, h),
                               flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=bg_val)
    return corrected, median_angle


def estimate_block_skew(
    binary:    np.ndarray,
    max_angle: float = 15.0,
    n_coarse:  int   = 61,
    n_fine:    int   = 21,
) -> float:
    """
    Estima el ángulo de un bloque maximizando la varianza de la proyección H.
    Barrido coarse-to-fine: grueso sobre imagen reducida, fino sobre la original.
    """
    H, W = binary.shape
    if H < 20 or W < 40:
        return 0.0

    max_dim = max(H, W)
    small   = (cv2.resize(binary, (max(20, int(W * 600 / max_dim)),
                                   max(10, int(H * 600 / max_dim))),
                          interpolation=cv2.INTER_NEAREST)
               if max_dim > 600 else binary)

    def _var(img: np.ndarray, angle: float) -> float:
        if abs(angle) < 0.05:
            rot = img
        else:
            h, w = img.shape
            M   = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            rot = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        return float((rot < 128).sum(axis=1).astype(np.float64).var())

    coarse_angles = np.linspace(-max_angle, max_angle, n_coarse)
    best_coarse   = float(coarse_angles[int(np.argmax([_var(small, a) for a in coarse_angles]))])
    step          = (coarse_angles[1] - coarse_angles[0]) if len(coarse_angles) > 1 else 1.0
    fine_angles   = np.linspace(best_coarse - step, best_coarse + step, n_fine)
    best_angle    = float(fine_angles[int(np.argmax([_var(binary, a) for a in fine_angles]))])

    return best_angle if abs(best_angle) >= 0.3 else 0.0


def deskew_block(binary: np.ndarray, max_angle: float = 15.0) -> tuple[np.ndarray, float]:
    """Estima y corrige el ángulo local de un bloque binario."""
    angle = estimate_block_skew(binary, max_angle=max_angle)
    if abs(angle) < 0.3:
        return binary, 0.0
    H, W = binary.shape
    M    = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, 1.0)
    return cv2.warpAffine(binary, M, (W, H),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255), angle


# 4. PROCESADO INTERNO: CAMINO CON DESKEW POR BLOQUE

def _process_with_block_deskew(
    binary:      np.ndarray,
    block_boxes: list[tuple[int, int, int, int]],
    cfg:         PipelineConfig,
    warns:       list[str],
)-> tuple[list[np.ndarray], list[tuple[int, int, int, int]], list[tuple[float, float, float, float, float]]]:
    """
    Para cada bloque: deskew local → re-detección de líneas → expand → straighten → normalize.
    Las coordenadas de retorno son aproximadas (no aplica rotación inversa).
    """
    if cfg.debug:
        print(f"[_process_with_block_deskew] Processing {len(block_boxes)} blocks")
    lines:       list[np.ndarray]               = []
    valid_boxes: list[tuple[int, int, int, int]] = []
    oriented_boxes: list[tuple[float, float, float, float, float]] = []
    H_bin = binary.shape[0]

    for (by0, by1, bx0, bx1) in block_boxes:
        if (by1 - by0) < cfg.min_line_height or (bx1 - bx0) < cfg.min_line_width:
            continue

        # Padding vertical para capturar diacríticos (tildes, acentos…) que
        # sobresalen fuera del bounding-box del bloque detectado por segment_all.
        # Reducimos la fracción de altura utilizada para padding para evitar
        # incluir líneas adyacentes en documentos densos.
        block_h = by1 - by0
        pad_v = min(max(cfg.expand_no_ink_gap, block_h // 8), 30)
        crop_y0 = max(0,     by0 - pad_v)
        crop_y1 = min(H_bin, by1 + pad_v)

        crop = binary[crop_y0:crop_y1, bx0:bx1]
        if crop.size == 0 or not (crop < 128).any():
            continue

        rotated, angle = deskew_block(crop, max_angle=cfg.deskew_block_max_angle)
        if cfg.debug and abs(angle) > 0.3:
            print(f"  [deskew_blocks] y=[{by0},{by1}] x=[{bx0},{bx1}] -> {angle:.2f} deg")

        line_ys = detect_lines(rotated, cfg)
        if not line_ys:
            continue

        # Filtrar líneas detectadas en el crop: conservar solo aquellas cuyo
        # centro vertical cae dentro de la región ORIGINAL del bloque (sin el
        # padding). Esto evita capturar líneas de bloques adyacentes que queden
        # dentro del área de padding, lo que causaría cajas solapadas.
        block_top_in_crop = by0 - crop_y0
        block_bot_in_crop = by1 - crop_y0
        line_ys = [
            (yt, yb) for (yt, yb) in line_ys
            if block_top_in_crop <= (yt + yb) / 2 <= block_bot_in_crop
        ]
        if not line_ys:
            continue

        bW           = rotated.shape[1]
        boxes_local  = [(yt, yb, 0, bW) for (yt, yb) in line_ys]
        if cfg.expand_to_ink:
            # Safe-crop asimétrico:
            #   · Hacia arriba  → safe_y0 limitado a accent_margin por encima
            #     del borde del bloque. Permite capturar acentos sobre
            #     mayúsculas (Ú, Í, Á…), pero sin llegar hasta y=0 del crop
            #     (que incluye pad_v de contenido ajeno al bloque). Con
            #     safe_y0=0, expand_all_boxes podía expandir la primera línea
            #     hasta el comienzo del crop y absorber tinta del bloque
            #     anterior, produciendo manchas en las imágenes individuales.
            #   · Hacia abajo   → safe_y1 ajustado al bloque + accent_margin.
            #     Evita que la última línea sangre hasta el bloque siguiente.
            # Usar un margen seguro dentro del crop para evitar recortar
            # trazos desplazados por la rotación/enderezado.
            accent_margin = min(pad_v, rotated.shape[0] // 2)
            safe_y0 = max(0, block_top_in_crop - accent_margin)
            safe_y1 = min(rotated.shape[0], block_bot_in_crop + accent_margin)
            safe_crop     = rotated[safe_y0:safe_y1, :]
            boxes_in_safe = [(yt - safe_y0, yb - safe_y0, xl, xr)
                             for (yt, yb, xl, xr) in boxes_local]
            expanded = expand_all_boxes(
                safe_crop, boxes_in_safe,
                max_expand_frac=cfg.expand_max_frac,
                no_ink_gap=cfg.expand_no_ink_gap,
                min_ink_frac=cfg.expand_min_ink_frac,
            )
            boxes_local = [(yt + safe_y0, yb + safe_y0, xl, xr)
                           for (yt, yb, xl, xr) in expanded]

        for (yt, yb, xl, xr) in boxes_local:
            orig_strip = rotated[yt:yb, xl:xr]
            strip = orig_strip
            if strip.size == 0:
                continue
            # Primero intentar recortar orientado según la baseline local
            ang = 0.0
            if getattr(cfg, 'use_oriented_crop', False):
                try:
                    from preprocessing.line_processing import rotate_strip_by_baseline
                    rstrip, ang = rotate_strip_by_baseline(strip, n_slices=cfg.straighten_slices)
                    if rstrip is not None and rstrip.size > 0:
                        # reemplazar strip por la versión rotada para normalizar
                        strip_was_rotated = (rstrip is not strip)  # True if rotation occurred
                        strip = rstrip
                        if cfg.debug and abs(ang) > 0.05:  # lowered threshold from 0.1 to catch more
                            try:
                                dbg_dir = Path(cfg.debug_dir)
                                dbg_dir.mkdir(parents=True, exist_ok=True)
                                name = f"rotated_block_by_{by0}_{bx0}_yt_{yt}.png"
                                ok, buf = cv2.imencode('.png', strip)
                                if ok:
                                    (dbg_dir / name).write_bytes(buf.tobytes())
                                    if cfg.debug:
                                        print(f"  [rotate] angle={ang:.2f} deg -> saved")
                            except Exception as e:
                                if cfg.debug:
                                    print(f"  [rotate] angle={ang:.2f}° but save failed: {e}")
                        elif cfg.debug:
                            print(f"  [rotate] angle={ang:.2f}° (not saved, threshold < 0.05)")
                except Exception as e:
                    ang = 0.0
                    if cfg.debug:
                        print(f"  [rotate] exception: {e}")
            if cfg.straighten_lines:
                strip = straighten_line(strip, poly_degree=cfg.straighten_poly,
                                        n_slices=cfg.straighten_slices)
            try:
                norm = normalize_line(strip, target_height=cfg.target_height,
                                      trim_margin=cfg.trim_margin)
            except Exception as e:
                warns.append(f"Error normalizando línea y=[{by0},{by1}] x=[{bx0},{bx1}]: {e}")
                continue
            if float((norm < 0.5).mean()) < 0.02:
                continue
            lines.append(norm)
            # Ajustar las coordenadas de la caja según la tinta real en el
            # `strip` resultante (puede haberse desplazado verticalmente al
            # enderezar). Las coordenadas se expresan respecto al origen de
            # la imagen original.
            rows = np.where((strip < 128).any(axis=1))[0]
            if rows.size == 0:
                continue
            new_top = int(rows[0])
            new_bot = int(rows[-1]) + 1
            # Conservativamente expandir la caja original verticalmente para
            # cubrir posibles desplazamientos por rotación aplicada.
            try:
                vert_pad = int(np.ceil(abs(np.sin(np.radians(float(ang)))) * strip.shape[1] / 2.0))
            except Exception:
                vert_pad = 0
            g_top = max(0, crop_y0 + yt + new_top - vert_pad)
            g_bot = min(H_bin, crop_y0 + yt + new_bot + vert_pad)
            g_left = bx0 + xl
            g_right = bx0 + xr
            valid_boxes.append((g_top, g_bot, g_left, g_right))
            # Calcular bounding-rotado (minAreaRect) sobre la tinta de la
            # franja ORIGINAL (antes de rotar). Esto asegura que las
            # coordenadas del rectángulo se mapeen correctamente al sistema
            # de coordenadas global de la imagen.
            try:
                pts = np.column_stack(np.where(orig_strip < 128))
                if pts.shape[0] >= 3:
                    pts_xy = pts[:, ::-1].astype(np.float32)
                    rect = cv2.minAreaRect(pts_xy)
                    (cx, cy), (w, h), ang_rect = rect
                    # convertir centro a coordenadas globales
                    g_cx = g_left + float(cx)
                    g_cy = g_top + float(cy)
                    oriented_boxes.append((g_cx, g_cy, float(w), float(h), float(ang_rect)))
            except Exception:
                pass

    return lines, valid_boxes, oriented_boxes


# 5. ORQUESTADOR PRINCIPAL

def run(
    img_or_path: np.ndarray | str | Path,
    cfg:         Optional[PipelineConfig] = None,
) -> PipelineResult:
    """
    Ejecuta el pipeline completo sobre una imagen o ruta de archivo.

    Si cfg es None se llama a auto_config() automáticamente.
    Con cfg.deskew_blocks=True usa _process_with_block_deskew (camino B);
    de lo contrario usa el procesado estándar (camino A).
    """
    warns: list[str] = []

    img_bgr = load_image(img_or_path) if isinstance(img_or_path, (str, Path)) else img_or_path
    if cfg is None:
        cfg = auto_config(img_bgr)

    gray = to_gray(img_bgr, cfg)
    if cfg.deskew:
        gray, angle = deskew_image(gray, max_angle=cfg.deskew_max_angle)
        if cfg.debug and abs(angle) > 0.1:
            print(f"  [pipeline] Deskew global: {angle:.2f}°")

    binary = binarize(
        img=gray,
        window=cfg.sauvola_window, k=cfg.sauvola_k,
        use_clahe=cfg.use_clahe, clahe_clip=cfg.clahe_clip, clahe_tile=cfg.clahe_tile,
        invert=cfg.invert_binary,
        use_bilateral=cfg.use_bilateral, bilateral_d=cfg.bilateral_d,
        bilateral_sc=cfg.bilateral_sc, bilateral_ss=cfg.bilateral_ss,
        global_floor_pct=cfg.global_floor_pct,
        use_remove_bg=getattr(cfg, 'use_remove_bg', False),
    )
    if cfg.morph_open > 0 or cfg.morph_close > 0:
        binary = clean_binary(binary, cfg.morph_open, cfg.morph_close)
    if getattr(cfg, 'use_adaptive_component_filter', False):
        binary = adaptive_filter_components(binary)
    if cfg.min_component_area > 0:
        binary = filter_small_components(binary, cfg.min_component_area)

    if cfg.detect_text_blocks:
        boxes_4d, block_boxes = segment_all(binary, cfg)
    else:
        raw         = detect_lines(binary, cfg)
        H_i, W_i   = binary.shape
        boxes_4d    = [(t, b, 0, W_i) for t, b in raw]
        block_boxes = [(0, H_i, 0, W_i)]

    if len(boxes_4d) == 0 and not cfg.deskew_blocks:
        warns.append("No se detectaron líneas de texto.")
        return PipelineResult(lines=[], line_boxes=[], block_boxes=block_boxes,
                              binary=binary, warnings=warns, config_used=cfg)

    oriented_boxes: list[tuple[float, float, float, float, float]] = []
    if cfg.deskew_blocks and block_boxes:
        lines, valid_boxes, oriented_boxes = _process_with_block_deskew(binary, block_boxes, cfg, warns)
        if not lines:
            warns.append("deskew_blocks activo pero no se extrajeron líneas.")
    else:
        if cfg.expand_to_ink and boxes_4d:
            boxes_4d = expand_all_boxes(
                binary, boxes_4d,
                max_expand_frac=cfg.expand_max_frac,
                no_ink_gap=cfg.expand_no_ink_gap,
                min_ink_frac=cfg.expand_min_ink_frac,
                block_boxes=block_boxes,
            )
        lines, valid_boxes = [], []
        oriented_boxes = []
        for (y_top, y_bot, x_left, x_right) in boxes_4d:
            strip = binary[y_top:y_bot, x_left:x_right]
            if strip.size == 0:
                continue
            # Intentar recortar orientado por baseline antes de enderezar
            ang = 0.0
            if getattr(cfg, 'use_oriented_crop', False):
                try:
                    from preprocessing.line_processing import rotate_strip_by_baseline
                    rstrip, ang = rotate_strip_by_baseline(strip, n_slices=cfg.straighten_slices)
                    if rstrip is not None and rstrip.size > 0:
                        strip = rstrip
                        if cfg.debug and abs(ang) > 0.1:
                            try:
                                dbg_dir = Path(cfg.debug_dir)
                                dbg_dir.mkdir(parents=True, exist_ok=True)
                                name = f"rotated_line_{y_top}_{x_left}.png"
                                ok, buf = cv2.imencode('.png', strip)
                                if ok:
                                    (dbg_dir / name).write_bytes(buf.tobytes())
                                print(f"  [debug] rotated line strip saved: {dbg_dir / name} angle={ang:.2f}")
                            except Exception:
                                pass
                except Exception:
                    ang = 0.0
            if cfg.straighten_lines:
                strip = straighten_line(strip, poly_degree=cfg.straighten_poly,
                                        n_slices=cfg.straighten_slices)
            try:
                norm = normalize_line(strip, target_height=cfg.target_height,
                                      trim_margin=cfg.trim_margin)
            except Exception as e:
                warns.append(f"Error normalizando línea y=[{y_top},{y_bot}]: {e}")
                continue
            if float((norm < 0.5).mean()) < 0.02:
                continue
            lines.append(norm)
            # Ajustar caja según tinta real en `strip` tras enderezar
            rows = np.where((strip < 128).any(axis=1))[0]
            if rows.size == 0:
                continue
            new_top = int(rows[0])
            new_bot = int(rows[-1]) + 1
            # Conservatively expand vertical by margin based on rotation angle
            try:
                vert_pad = int(np.ceil(abs(np.sin(np.radians(float(ang)))) * strip.shape[1] / 2.0))
            except Exception:
                vert_pad = 0
            g_top = max(0, y_top + new_top - vert_pad)
            g_bot = min(binary.shape[0], y_top + new_bot + vert_pad)
            g_left = x_left
            g_right = x_left + strip.shape[1]
            valid_boxes.append((g_top, g_bot, g_left, g_right))
            # compute oriented minAreaRect for this strip
            try:
                pts = np.column_stack(np.where(strip < 128))
                if pts.shape[0] >= 3:
                    pts_xy = pts[:, ::-1].astype(np.float32)
                    rect = cv2.minAreaRect(pts_xy)
                    (cx, cy), (w, h), ang = rect
                    g_cx = g_left + float(cx)
                    g_cy = g_top + float(cy)
                    oriented_boxes.append((g_cx, g_cy, float(w), float(h), float(ang)))
            except Exception:
                pass

    if cfg.debug:
        _save_debug(gray, binary, valid_boxes, block_boxes, cfg.debug_dir)

    return PipelineResult(lines=lines, line_boxes=valid_boxes, block_boxes=block_boxes,
                          oriented_boxes=oriented_boxes, binary=binary, warnings=warns, config_used=cfg)


# 6. DEBUG

def _save_debug(gray, binary, boxes, block_boxes, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, img in [("gray.png", gray), ("binary.png", binary)]:
        ok, buf = cv2.imencode(".png", img)
        if ok:
            (out / name).write_bytes(buf.tobytes())
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for (bt, bb, bl, br) in block_boxes:
        cv2.rectangle(vis, (bl, bt), (br, bb), (255, 180, 0), 2)
    for i, (yt, yb, xl, xr) in enumerate(boxes):
        c = (0, 180, 0) if i % 2 == 0 else (180, 80, 0)
        cv2.rectangle(vis, (xl, yt), (xr, yb), c, 2)
        cv2.putText(vis, str(i + 1), (xl + 4, yt + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    ok, buf = cv2.imencode(".png", vis)
    if ok:
        (out / "lines_detected.png").write_bytes(buf.tobytes())