from pathlib import Path
from typing  import Optional

import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal  import find_peaks

from preprocessing.binarization    import (
    binarize, clean_binary, adaptive_filter_components, filter_small_components,
    mask_binding_strips, trim_orphan_components,
)
from preprocessing.config          import (
    ImageMetrics, PipelineConfig, PipelineResult
)
from preprocessing.line_processing import (
    detect_lines, expand_all_boxes, normalize_line, straighten_line
)


# 1. ANÁLISIS DE IMAGEN Y AUTO-CONFIGURACIÓN

def analyze(img: np.ndarray) -> ImageMetrics:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    H, W = gray.shape

    p5, p95  = float(np.percentile(gray, 5)), float(np.percentile(gray, 95))
    contrast = p95 - p5
    mean_lum = float(gray.mean())

    margin  = max(10, min(H, W) // 12)
    # Excluir franjas laterales antes de muestrear las esquinas: artefactos de
    # encuadernación (bandas oscuras) en el borde izquierdo/derecho del escáner
    # empujarían la media por debajo de 127 y harían dark_background=True
    # aunque el fondo real de la página sea blanco.
    border_frac = max(0.06, min(0.15, 100.0 / max(W, 1)))
    cx0 = int(W * border_frac)
    cx1 = W - cx0
    gray_c = gray[:, cx0:cx1]
    Wc     = gray_c.shape[1]
    cm     = max(1, min(margin, Wc // 2, H // 2))
    corners = np.concatenate([
        gray_c[:cm,  :cm].ravel(),        gray_c[:cm,  Wc - cm:].ravel(),
        gray_c[H-cm:, :cm].ravel(),       gray_c[H-cm:, Wc - cm:].ravel(),
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

    src          = gray if not dark_background else (255 - gray)
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

    needs_clahe = contrast < 60

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
    m = analyze(img)

    use_clahe  = m.contrast < 60
    clahe_clip = 1.0
    clahe_tile = 16
    use_bilateral = use_clahe

    max_win        = max(15, min(m.H, m.W) // 2)
    raw_win        = max(15, min(201, int(m.estimated_text_h * 1.5), max_win))
    sauvola_window = raw_win + (0 if raw_win % 2 == 1 else 1)
    sauvola_k      = 0.10 if m.contrast < 50 else 0.17 if m.contrast < 80 else 0.25

    return PipelineConfig(
        sauvola_window=sauvola_window,
        sauvola_k=sauvola_k,
        morph_open=2 if m.contrast < 45 else 0,
        use_clahe=use_clahe,
        clahe_clip=clahe_clip,
        clahe_tile=clahe_tile,
        use_bilateral=use_bilateral,
        bilateral_sc=50.0,
        bilateral_ss=50.0,
        invert_binary=m.dark_background,
        global_floor_pct=92.0 if m.contrast < 80 else 0.0,
        min_component_area=4,
        use_adaptive_component_filter=False,
        use_remove_bg=False,
        line_v_dilation=max(2, int(m.estimated_text_h * 0.12)),
        # min_line_height: valor adaptativo basado en la altura estimada del texto.
        # El defecto (24px) es demasiado alto para texto comprimido o baja resolución.
        min_line_height=max(8, int(m.estimated_text_h * 0.35)),
        expand_to_ink=True,
        straighten_lines=True,
        deskew=True,
        deskew_blocks=True,
    )


# 2. DETECCIÓN DE BLOQUES DE TEXTO (XY-Cut de 3 niveles)

def _find_h_separators(binary: np.ndarray, min_gap_h: int = 0,
                        threshold_frac: float = 0.04) -> list[int]:
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
        # Detección bilateral del hueco (gutter): se camina desde cada pico
        # hacia el otro hasta que la densidad cae por debajo de un umbral
        # situado a media profundidad del valle. `gL` marca dónde termina la
        # caída desde el pico izquierdo (≈ derecha del texto izquierdo); `gR`
        # dónde empieza la subida hacia el pico derecho (≈ izquierda del
        # texto derecho). El separador se coloca en el centro de [gL, gR].
        #
        # Esto reemplaza la elección anterior basada en argmin, que en libros
        # con encuadernación central daba resultados sesgados: la zona del lomo
        # introduce moteado de tinta (densidad ~0.3 × pico) que no rompe la
        # condición de "valle" pero desplaza el mínimo absoluto hacia uno de
        # los dos lados, dejando el separador pegado al borde de la página
        # vecina y permitiendo que sus letras iniciales se filtren al bbox de
        # la columna actual.
        #
        # El umbral a media profundidad (vv + (lp-vv)*0.5 ≈ punto medio entre
        # fondo del valle y la cresta más baja) es el compromiso correcto: si
        # se pone demasiado cerca del fondo del valle (≤ 0.15) los gutters
        # estrechos —típicos cuando la encuadernación no genera moteado— dan
        # un ancho de hueco insuficiente y se rechazan; si se pone cerca del
        # pico (≥ 0.80) cualquier rebaje de densidad se cuenta como gutter y
        # el separador se desplaza dentro del cuerpo de texto.
        gutter_thr = vv + (lp - vv) * 0.5
        gL = peaks[i]
        while gL < vi and v_sm[gL] >= gutter_thr:
            gL += 1
        gR = peaks[i + 1]
        while gR > vi and v_sm[gR] >= gutter_thr:
            gR -= 1
        if gR <= gL or (gR - gL) < max(40, W // 20):
            continue
        if total_ink > 0:
            if v_sm[:vi].sum() / total_ink < 0.20 or v_sm[vi:].sum() / total_ink < 0.20:
                continue
        sep_x = (gL + gR) // 2
        separators.append(sep_x)

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
    """Fusiona bloques de la misma columna cuyo hueco vertical es menor que merge_gap."""
    if len(blocks) < 2:
        return blocks

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
                merged.append((new_box, sorted(lines_i + lines_j, key=lambda b: b[0])))
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
      3. Sub-separadores dentro de cada columna → sub-secciones
      4. detect_lines dentro de cada sub-sección
      5. Agrupación en párrafos y medición x-extent por tinta real

    Retorna (line_boxes, block_boxes), ambas como (y_top, y_bot, x_left, x_right).
    """
    H, W       = binary.shape
    min_gap_h  = getattr(cfg, 'block_min_h_gap',    0)
    h_thr_frac = getattr(cfg, 'block_h_thr_frac',   0.04)
    max_cols   = getattr(cfg, 'block_max_cols',      4)
    min_depth  = getattr(cfg, 'block_col_min_depth', 0.28)
    min_line_h = getattr(cfg, 'min_line_height',     8)
    para_split = getattr(cfg, 'para_split_factor',   2.5)
    merge_gap  = getattr(cfg, 'block_merge_gap', 0) or max(40, H // 60)

    h_seps     = _find_h_separators(binary, min_gap_h=min_gap_h, threshold_frac=h_thr_frac)
    h_bounds   = [0] + h_seps + [H]
    h_sections = [(h_bounds[i], h_bounds[i + 1])
                  for i in range(len(h_bounds) - 1)
                  if h_bounds[i + 1] - h_bounds[i] >= min_line_h]

    all_blocks: list[tuple[tuple, list]] = []
    min_h_vsplit  = max(500, H // 7)
    min_h_section = max(min_line_h, H // 50)  # fix: 50 hardcoded causaba que imágenes < 50px se saltaran por completo
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

            col_min_gap  = max(150, min(300, section_h // 8))
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

                # Calcular el x-extent UNA VEZ por sub-sección, sobre TODAS las
                # líneas detectadas. Calcularlo por párrafo (como antes) hace
                # que un párrafo encabezado por una línea sangrada (p. ej.
                # "Apretó el instrumento.") fije `para_x0` al borde de la
                # sangría; las líneas siguientes del párrafo, que llegan al
                # margen real de la columna, quedan cortadas por la izquierda.
                sx0_sub  = max(0, x0 - margin)
                sx1_sub  = min(W, x1 + margin)
                if line_ys:
                    sub_y0 = abs_y_base + line_ys[0][0]
                    sub_y1 = abs_y_base + line_ys[-1][1]
                else:
                    sub_y0 = abs_y_base
                    sub_y1 = y0 + sc_y1
                ink_cols_sub = np.where(
                    (binary[max(0, sub_y0): min(H, sub_y1), sx0_sub:sx1_sub] < 128).any(axis=0)
                )[0]
                if len(ink_cols_sub) == 0:
                    continue
                sub_x0 = max(0, sx0_sub + int(ink_cols_sub[0])  - margin)
                sub_x1 = min(W, sx0_sub + int(ink_cols_sub[-1]) + 1 + margin)

                for para_lines in para_groups:
                    if not para_lines:
                        continue
                    abs_y0 = abs_y_base + para_lines[0][0]
                    abs_y1 = abs_y_base + para_lines[-1][1]

                    para_cell = binary[abs_y0:abs_y1, sub_x0:sub_x1]
                    if para_cell.size == 0 or not (para_cell < 128).any():
                        continue

                    block_lines = [(abs_y_base + yt, abs_y_base + yb, sub_x0, sub_x1)
                                   for (yt, yb) in para_lines]

                    # accent_guard: margen superior extra para que expand_all_boxes
                    # pueda alcanzar acentos de mayúsculas que quedan por encima
                    # del y_top detectado en la primera línea del párrafo.
                    para_h       = abs_y1 - abs_y0
                    accent_guard = max(5, int(para_h * 0.08))
                    all_blocks.append(((max(0, abs_y0 - accent_guard), abs_y1, sub_x0, sub_x1), block_lines))

    all_blocks = _merge_close_blocks(binary, all_blocks, merge_gap, margin)

    block_boxes = [bb for bb, _ in all_blocks]
    all_lines   = [line for (_, lines) in all_blocks for line in lines]
    filtered    = [(yt, yb, xl, xr) for (yt, yb, xl, xr) in all_lines
                   if binary[yt:yb, xl:xr].size > 0 and (binary[yt:yb, xl:xr] < 128).any()]

    hzf      = getattr(cfg, 'header_zone_frac', 1.0 / 6.0)
    HEADER_Y = int(H * hzf) if hzf and hzf > 0 else 0

    h_lines = sorted([(yt, yb, xl, xr) for (yt, yb, xl, xr) in filtered    if yt < HEADER_Y], key=lambda b: b[0])
    b_lines = sorted([(yt, yb, xl, xr) for (yt, yb, xl, xr) in filtered    if yt >= HEADER_Y], key=lambda b: (b[2], b[0]))
    h_blks  = sorted([(yt, yb, xl, xr) for (yt, yb, xl, xr) in block_boxes if yt < HEADER_Y], key=lambda b: b[0])
    b_blks  = sorted([(yt, yb, xl, xr) for (yt, yb, xl, xr) in block_boxes if yt >= HEADER_Y], key=lambda b: (b[2], b[0]))

    return h_lines + b_lines, h_blks + b_blks


# 3. CONVERSIÓN Y DESKEW

def load_image(path: str | Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return img


def to_gray(img: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    if img.ndim == 2:
        return img
    return img[:, :, 0] if cfg.use_blue_channel else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def deskew_image(
    gray:      np.ndarray,
    max_angle: float = 15.0,
) -> tuple[np.ndarray, float]:
    """
    Corrige la inclinación global del documento.

    Usa HoughLinesP ponderando por longitud² para que los baselines largos dominen
    la estimación sobre los trazos oblicuos de letras individuales (~5-12°).
    La mediana ponderada con guardia IQR descarta distribuciones multimodales
    (mezcla de letras y líneas rectas) antes de rotar.
    El color de borde se estima como mediana de los píxeles de las 4 aristas para
    evitar bandas negras de escáner en las esquinas rotadas.
    """
    h, w  = gray.shape
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
    border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg_val        = int(np.median(border_pixels))
    # Interpolación CÚBICA en grises. Antes se usaba INTER_NEAREST aquí mismo y
    # eso producía aliasing fuerte: cada píxel rotado tomaba el valor del más
    # cercano de la cuadrícula original sin promediar, así que en ángulos no
    # múltiplos de 90° las letras quedaban "movidas" — los bordes oblicuos se
    # convertían en una escalera de saltos discretos en vez de un degradado.
    # Tras la binarización Sauvola (que reacciona localmente a esos saltos)
    # esa escalera se transforma en bordes irregulares y caracteres torcidos.
    # INTER_CUBIC reconstruye cada píxel con una vecindad 4×4 ponderada, así
    # los bordes en gris quedan suavizados y Sauvola produce binarios mucho
    # más limpios. Se usa NEAREST en cambio para imágenes ya binarizadas.
    corrected     = cv2.warpAffine(gray, M, (w, h),
                                   flags=cv2.INTER_CUBIC,
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
    Barrido coarse-to-fine: grueso sobre imagen reducida (≤600px), fino sobre la original.
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
    angle = estimate_block_skew(binary, max_angle=max_angle)
    if abs(angle) < 0.3:
        return binary, 0.0
    H, W = binary.shape
    M    = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, 1.0)
    return cv2.warpAffine(binary, M, (W, H),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255), angle


# 4. PROCESADO INTERNO: CAMINO CON DESKEW POR BLOQUE

def _process_strip(
    strip: np.ndarray,
    cfg:   "PipelineConfig",
    warns: list,
    label: str = "",
) -> "tuple | None":
    """
    Núcleo de procesado de una franja binaria de línea:
      oriented-crop -> straighten_line -> trim_orphan_components -> trim vertical
      -> normalize_line -> filtro de tinta mínima.

    Devuelve (norm, ang, new_top, new_bot, clean_strip) o None si debe descartarse.
      - norm: array float32 [0,1] de altura fija (para OCR).
      - ang: ángulo de rotación oriented (informativo).
      - new_top, new_bot: rango Y dentro del strip ORIGINAL que cubre la línea
        principal una vez ajustada a la tinta real (sin huérfanos).
      - clean_strip: strip uint8 binarizado y limpio (rotado/straightened y sin
        tinta de líneas vecinas), recortado al rango Y de la línea principal.
        Listo para guardar como JPG.
    """
    if strip.size == 0:
        return None

    orig_strip_h = strip.shape[0]
    rows_orig = np.where((strip < 128).any(axis=1))[0]
    if rows_orig.size == 0:
        return None

    ang = 0.0
    if getattr(cfg, "use_oriented_crop", False):
        try:
            from preprocessing.line_processing import rotate_strip_by_baseline
            rstrip, ang = rotate_strip_by_baseline(strip, n_slices=cfg.straighten_slices)
            if rstrip is not None and rstrip.size > 0:
                strip = rstrip
        except Exception:
            ang = 0.0

    if cfg.straighten_lines:
        strip = straighten_line(strip, poly_degree=cfg.straighten_poly,
                                n_slices=cfg.straighten_slices)

    # Limpieza de huérfanos: elimina descenders/ascenders/manchas de líneas
    # adyacentes que se cuelan en el padding superior o inferior del strip.
    if getattr(cfg, "trim_orphans_per_line", True):
        strip = trim_orphan_components(strip)

    # Recorte vertical apretado a la banda principal (con margen mínimo) sobre
    # el strip ya limpio. Esto define new_top/new_bot que se usarán para
    # mapear el bbox al sistema de coordenadas del binary global.
    rows_clean = np.where((strip < 128).any(axis=1))[0]
    if rows_clean.size == 0:
        return None
    ink_top = int(rows_clean[0])
    ink_bot = int(rows_clean[-1]) + 1
    pad     = max(cfg.trim_margin, 2)
    new_top = max(0, ink_top - pad)
    new_bot = min(strip.shape[0], ink_bot + pad)

    clean_strip = strip[new_top:new_bot, :]

    # Recorte HORIZONTAL apretado al rango con tinta. Sin esto, una línea
    # corta de fin de párrafo (p. ej. "necía.") queda como una mancha de
    # ~100 px de tinta en un lienzo de ~572 px (el ancho del bloque); tras
    # normalize_line + redimensionado a altura fija, la fracción de tinta
    # cae por debajo del 0.02 que exige el filtro final, y la línea se
    # descarta erróneamente. Recortando primero por columnas con tinta, la
    # densidad se recupera y la línea pasa el filtro.
    cols_clean = np.where((clean_strip < 128).any(axis=0))[0]
    if cols_clean.size == 0:
        return None
    ink_left  = int(cols_clean[0])
    ink_right = int(cols_clean[-1]) + 1
    pad_h     = max(cfg.trim_margin, 2)
    cl_left   = max(0,                       ink_left  - pad_h)
    cl_right  = min(clean_strip.shape[1],    ink_right + pad_h)
    clean_strip = clean_strip[:, cl_left:cl_right]

    try:
        norm = normalize_line(clean_strip, target_height=cfg.target_height,
                              trim_margin=cfg.trim_margin)
    except Exception as e:
        warns.append(f"Error normalizando {label}: {e}")
        return None

    if float((norm < 0.5).mean()) < 0.02:
        return None

    # Las coordenadas new_top/new_bot están en el sistema del strip POSIBLEMENTE
    # rotado. Se devuelven tal cual; pipeline.run las usa relativas al strip
    # original al construir el bbox global, asumiendo que el rotado preserva
    # aproximadamente las coordenadas (estamos hablando de ángulos < 5° en
    # general). Esa aproximación es suficiente para visualización del bbox.
    return norm, ang, new_top, new_bot, clean_strip


def _process_with_block_deskew(
    binary:       np.ndarray,
    block_boxes:  list[tuple[int, int, int, int]],
    cfg:          PipelineConfig,
    warns:        list[str],
    global_angle: float = 0.0,
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[int, int, int, int]]]:
    """
    Para cada bloque: deskew local residual -> re-detección de líneas -> expand -> straighten -> normalize.

    El deskew global ya fue aplicado sobre toda la imagen antes de llamar aquí.
    La corrección local solo se aplica cuando el crop es suficientemente alto
    (≥ block_min_h_for_skew) y el ángulo residual supera block_residual_threshold.

    Retorna (lines_normalized, line_crops_uint8, line_boxes).
    """
    if cfg.debug:
        print(f"[_process_with_block_deskew] {len(block_boxes)} bloques  "
              f"global_angle={global_angle:.2f}°")

    lines:       list[np.ndarray]               = []
    line_crops:  list[np.ndarray]               = []
    valid_boxes: list[tuple[int, int, int, int]] = []
    H_bin        = binary.shape[0]
    min_h_local  = getattr(cfg, "block_min_h_for_skew",     60)
    residual_thr = getattr(cfg, "block_residual_threshold", 0.5)

    for (by0, by1, bx0, bx1) in block_boxes:
        if (by1 - by0) < cfg.min_line_height or (bx1 - bx0) < cfg.min_line_width:
            continue

        block_h = by1 - by0
        pad_v   = max(cfg.expand_no_ink_gap, int(block_h * 0.15))
        crop_y0 = max(0,     by0 - pad_v)
        crop_y1 = min(H_bin, by1 + pad_v)

        crop = binary[crop_y0:crop_y1, bx0:bx1]
        if crop.size == 0 or not (crop < 128).any():
            continue

        crop_h  = crop.shape[0]
        rotated = crop
        if crop_h >= min_h_local:
            residual = estimate_block_skew(crop, max_angle=cfg.deskew_block_max_angle)
            if abs(residual) >= residual_thr:
                H_c, W_c = crop.shape
                M = cv2.getRotationMatrix2D((W_c / 2.0, H_c / 2.0), residual, 1.0)
                rotated  = cv2.warpAffine(crop, M, (W_c, H_c),
                                          flags=cv2.INTER_NEAREST,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)
                if cfg.debug:
                    print(f"  [deskew_blocks] y=[{by0},{by1}] x=[{bx0},{bx1}] "
                          f"h={crop_h}px residual={residual:.2f}° → corregido")

        line_ys = detect_lines(rotated, cfg)
        if not line_ys:
            continue

        # Solo líneas cuyo centro cae dentro del bloque original (sin el padding)
        block_top_in_crop = by0 - crop_y0
        block_bot_in_crop = by1 - crop_y0
        line_ys = [
            (yt, yb) for (yt, yb) in line_ys
            if block_top_in_crop <= (yt + yb) / 2 <= block_bot_in_crop
        ]
        if not line_ys:
            continue

        bW          = rotated.shape[1]
        boxes_local = [(yt, yb, 0, bW) for (yt, yb) in line_ys]
        if cfg.expand_to_ink:
            # accent_margin: margen pequeño para capturar acentos/ascendentes que
            # sobresalen levemente del y_top detectado. NO debe ser tan grande como
            # pad_v (que sirve para cargar contexto de líneas adyacentes), porque
            # si safe_crop abarca múltiples secciones, expand_all_boxes puede cruzar
            # la frontera del bloque y generar cajas enormes que mezclan párrafos.
            accent_margin = min(20, max(5, int(block_h * 0.02)))
            safe_y0       = max(0,               block_top_in_crop - accent_margin)
            safe_y1       = min(rotated.shape[0], block_bot_in_crop + accent_margin)
            safe_crop     = rotated[safe_y0:safe_y1, :]
            boxes_in_safe = [(yt - safe_y0, yb - safe_y0, xl, xr)
                             for (yt, yb, xl, xr) in boxes_local
                             if safe_y0 <= yt and yb <= safe_y1]
            if boxes_in_safe:
                expanded    = expand_all_boxes(
                    safe_crop, boxes_in_safe,
                    max_expand_frac=cfg.expand_max_frac,
                    no_ink_gap=cfg.expand_no_ink_gap,
                    min_ink_frac=cfg.expand_min_ink_frac,
                )
                boxes_local = [(yt + safe_y0, yb + safe_y0, xl, xr)
                               for (yt, yb, xl, xr) in expanded]

        for (yt, yb, xl, xr) in boxes_local:
            orig_strip = rotated[yt:yb, xl:xr]
            label      = f"block_y{by0}_{by1}_x{bx0}_{bx1}_yt{yt}"
            result     = _process_strip(orig_strip, cfg, warns, label=label)
            if result is None:
                continue
            norm, _ang, new_top, new_bot, clean_strip = result
            lines.append(norm)
            line_crops.append(clean_strip)
            # bbox global apretado: new_top/new_bot ya están ajustados a la
            # tinta limpia del strip, sin padding extra. NO añadimos safe_pad
            # ni vert_pad: eso era lo que causaba que los bbox absorbieran
            # descenders/ascenders de las líneas vecinas.
            g_top = max(0,     crop_y0 + yt + new_top)
            g_bot = min(H_bin, crop_y0 + yt + new_bot)
            valid_boxes.append((g_top, g_bot, bx0 + xl, bx0 + xr))

    # Deduplicación entre bloques solapados. `segment_all` añade un margen
    # superior `accent_guard` a cada bloque para que `expand_to_ink` pueda
    # alcanzar acentos; cuando dos bloques contiguos quedan separados por una
    # distancia menor que ese margen (típico cuando el detector partió la
    # página en tres tramos: sup, medio, inf), las primeras líneas del tramo
    # inferior caen también dentro del padding del tramo superior. detect_lines
    # las captura dos veces — una por bloque — generando crops idénticos.
    # Aquí descartamos las repetidas comparando por bbox: si la última línea
    # añadida cubre prácticamente la misma zona Y/X que la actual, conservamos
    # la primera y omitimos la segunda.
    if len(valid_boxes) >= 2:
        dedup_lines:  list[np.ndarray] = []
        dedup_crops:  list[np.ndarray] = []
        dedup_boxes:  list[tuple[int, int, int, int]] = []
        order = sorted(range(len(valid_boxes)),
                       key=lambda i: (valid_boxes[i][2], valid_boxes[i][0]))
        seen: list[tuple[int, int, int, int]] = []
        for i in order:
            yt, yb, xl, xr = valid_boxes[i]
            is_dup = False
            for (sy0, sy1, sx0, sx1) in seen:
                if abs(xl - sx0) > 30 or abs(xr - sx1) > 30:
                    continue  # columnas distintas
                inter = max(0, min(yb, sy1) - max(yt, sy0))
                union = max(yb, sy1) - min(yt, sy0)
                if union > 0 and inter / union >= 0.60:
                    is_dup = True
                    break
            if is_dup:
                continue
            dedup_lines.append(lines[i])
            dedup_crops.append(line_crops[i])
            dedup_boxes.append(valid_boxes[i])
            seen.append(valid_boxes[i])
        lines       = dedup_lines
        line_crops  = dedup_crops
        valid_boxes = dedup_boxes

    return lines, line_crops, valid_boxes


# 5. ORQUESTADOR PRINCIPAL

def run(
    img_or_path: np.ndarray | str | Path,
    cfg:         Optional[PipelineConfig] = None,
) -> PipelineResult:
    """
    Ejecuta el pipeline completo sobre una imagen o ruta de archivo.
    Si cfg es None se llama a auto_config() automáticamente.
    Camino A (estándar): deskew global → binarización → mask binding →
                         segmentación → straighten → trim huérfanos → normalize.
    Camino B (deskew_blocks): igual que A pero con corrección local de inclinación por bloque.
    """
    warns: list[str] = []

    img_bgr = load_image(img_or_path) if isinstance(img_or_path, (str, Path)) else img_or_path
    if cfg is None:
        cfg = auto_config(img_bgr)

    gray         = to_gray(img_bgr, cfg)
    global_angle = 0.0
    if cfg.deskew:
        gray, global_angle = deskew_image(gray, max_angle=cfg.deskew_max_angle)
        if cfg.debug and abs(global_angle) > 0.1:
            print(f"  [pipeline] Deskew global: {global_angle:.2f}°")

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

    # Enmascarar las franjas oscuras laterales (encuadernación / borde del libro)
    # ANTES de cualquier detección de líneas o bloques. Esto pinta esas columnas
    # a blanco en `binary`, lo que elimina el artefacto en TODA la cadena
    # (proyecciones, bbox, crops finales) sin necesidad de parches posteriores.
    # Guardia: en imágenes muy bajas (recortes de línea única, H < 3×min_line_height)
    # la densidad de tinta por columna (ink_px / H) es artificialmente alta y los
    # trazos normales de letras superan el umbral del 30 %, lo que hace que
    # mask_binding_strips borre palabras enteras en los extremos izquierdo/derecho.
    _mask_binding_min_h = 3 * cfg.min_line_height
    if getattr(cfg, 'mask_binding', True) and binary.shape[0] >= _mask_binding_min_h:
        binary, x_left_safe, x_right_safe = mask_binding_strips(
            binary,
            max_frac=cfg.binding_max_frac,
            density_thr=cfg.binding_density_thr,
        )
        if cfg.debug and (x_left_safe > 0 or x_right_safe < binary.shape[1]):
            print(f"  [pipeline] Binding enmascarado: "
                  f"izq=[0,{x_left_safe}) der=[{x_right_safe},{binary.shape[1]})")

    if cfg.detect_text_blocks:
        boxes_4d, block_boxes = segment_all(binary, cfg)
    else:
        raw         = detect_lines(binary, cfg)
        H_i, W_i   = binary.shape
        boxes_4d    = [(t, b, 0, W_i) for t, b in raw]
        block_boxes = [(0, H_i, 0, W_i)]

    if len(boxes_4d) == 0 and not cfg.deskew_blocks:
        warns.append("No se detectaron líneas de texto.")
        return PipelineResult(lines=[], line_boxes=[], line_crops=[],
                              block_boxes=block_boxes,
                              binary=binary, deskew_angle=global_angle,
                              warnings=warns, config_used=cfg)

    lines:      list[np.ndarray] = []
    line_crops: list[np.ndarray] = []
    valid_boxes: list[tuple[int, int, int, int]] = []

    if cfg.deskew_blocks and block_boxes:
        lines, line_crops, valid_boxes = _process_with_block_deskew(
            binary, block_boxes, cfg, warns, global_angle=global_angle,
        )

    # Camino estándar (también fallback si deskew_blocks no extrajo líneas)
    if not lines:
        if cfg.deskew_blocks:
            warns.append(
                "deskew_blocks activo pero no se extrajeron líneas. "
                "Reintentando con camino estándar (sin deskew por bloque)."
            )
        if cfg.expand_to_ink and boxes_4d:
            boxes_4d = expand_all_boxes(
                binary, boxes_4d,
                max_expand_frac=cfg.expand_max_frac,
                no_ink_gap=cfg.expand_no_ink_gap,
                min_ink_frac=cfg.expand_min_ink_frac,
                block_boxes=block_boxes,
            )
        for (y_top, y_bot, x_left, x_right) in boxes_4d:
            strip  = binary[y_top:y_bot, x_left:x_right]
            label  = f"line_y{y_top}_{y_bot}_x{x_left}"
            result = _process_strip(strip, cfg, warns, label=label)
            if result is None:
                continue
            norm, _ang, new_top, new_bot, clean_strip = result
            lines.append(norm)
            line_crops.append(clean_strip)
            g_top = max(0, y_top + new_top)
            g_bot = min(binary.shape[0], y_top + new_bot)
            valid_boxes.append((g_top, g_bot, x_left, x_right))

    if cfg.debug:
        _save_debug(gray, binary, valid_boxes, block_boxes, cfg.debug_dir)

    return PipelineResult(lines=lines, line_boxes=valid_boxes, line_crops=line_crops,
                          block_boxes=block_boxes,
                          binary=binary, deskew_angle=global_angle,
                          warnings=warns, config_used=cfg)


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