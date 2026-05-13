"""
preprocessing/manuscript_lines.py — Detector de líneas para manuscritos cursivos.

¿Por qué un detector específico para manuscritos?
=================================================

`pipeline.run` (en `pipeline.py`) está diseñado para texto impreso. Su
detección de líneas falla sistemáticamente en cursiva manuscrita por tres
razones documentadas durante el debug del proyecto:

  1. **Descendentes largos partidos como dos líneas.** En una cursiva
     normal "Si para recobrar lo recobrado" la proyección horizontal
     de tinta tiene un pico para el cuerpo de las letras y otro pico
     menor para los descendentes ("p", "g", "y" debajo del baseline).
     Si `min_line_height = 0.35 × text_h` (valor por defecto en
     `auto_config`), ambos picos superan el umbral y se devuelven como
     dos "líneas". El modelo recibe la mitad superior y la mitad
     inferior de la misma línea como dos crops distintos, cortados a
     mitad de letra. La predicción sale corrompida.

  2. **Columnas inexistentes inducidas por espacios entre palabras.**
     `_find_column_separators` busca valles profundos en la proyección
     vertical. En manuscrito con interletraje grande (típico de
     escritura inglesa), las separaciones entre palabras producen
     valles del 30–50% de profundidad. El detector las interpreta como
     gutters y parte el documento por la mitad: las primeras 4 líneas
     se devuelven como 8 medio-líneas izquierda/derecha en orden
     incorrecto.

  3. **Líneas adyacentes fusionadas.** Cuando los ascendentes de la
     línea N+1 invaden el espacio interlineal de la línea N, la
     proyección no toca cero y el detector trata las dos líneas como
     una sola caja. El modelo recibe entonces una imagen multi-línea
     que no sabe leer.

Diseño de este detector
========================

  - **No segmenta bloques ni columnas.** Asume que el documento
    completo es un único bloque vertical de líneas. Si en el futuro
    hay manuscritos multi-columna, se maneja explícitamente desde la
    UI (regiones definidas por el usuario en `ocr_regions`).

  - **`min_line_height` adaptado a manuscrito** (`0.55 × text_h` en
    lugar de `0.35`). Las divisiones cuerpo/descendentes desaparecen.

  - **Clausura morfológica vertical PRE-detección.** Antes de
    proyectar, aplicamos `cv2.MORPH_CLOSE` con un kernel vertical
    ~`text_h * 0.5`. Esto une cuerpo + descendentes de la misma línea
    en una sola masa conexa, eliminando el doble pico.

  - **Suavizado del histograma con ventana proporcional**, no fija.
    Una ventana de 5 píxeles en una imagen de 3000 px de alto deja
    demasiado ruido; una de 50 emborrona separaciones reales.

  - **Devuelve bboxes en coordenadas del binario deskewed**, idénticas
    a las que devolvía `pipeline.run.line_boxes`, así el código
    consumidor (visualización de segmentación, OCR por regiones,
    etc.) sigue funcionando sin cambios.

Salida
------
    `detect_manuscript_lines(img_bgr) -> ManuscriptDetection`
        .binary:       np.ndarray (uint8) — binario deskewed
        .deskew_angle: float — ángulo aplicado
        .line_boxes:   list[tuple[int,int,int,int]] — (y_top, y_bot,
                       x_left, x_right) de cada línea detectada
        .block_boxes:  list[tuple[...]] — siempre 1 elemento que
                       envuelve todas las líneas (compat con frontend
                       que muestra "B1 — N bloques")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing      import Optional

import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal  import find_peaks

from preprocessing.binarization    import sauvola, normalize_illumination
from preprocessing.line_processing import rotate_strip_by_baseline


# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------

@dataclass
class ManuscriptLineConfig:
    """
    Parámetros del detector de líneas para manuscritos. Los defaults están
    calibrados para escritura cursiva moderna sobre papel blanco a unos
    1200–2400 px de altura. Si el documento es mucho más pequeño/grande,
    el código adapta varios valores en runtime usando la altura estimada
    del texto, así que normalmente NO hace falta tocar nada.
    """
    # --- Preprocesamiento (igual filosofía que line_preprocess.LineConfig) ---
    pick_best_channel:  bool  = True
    use_bilateral:      bool  = True
    bilateral_d:        int   = 5
    bilateral_sigma:    float = 40.0
    normalize_bg:       bool  = True

    # --- Deskew global ---
    deskew:             bool  = True
    max_skew_angle:     float = 6.0

    # --- Binarización Sauvola ---
    sauvola_window:     int   = 0       # 0 = auto a partir de text_h
    sauvola_k:          float = 0.12
    sauvola_pre_blur:   float = 0.6

    # --- Detector de líneas ---
    # Fracción de la altura estimada de texto que se considera "altura
    # mínima de línea". 0.55 evita que los descendentes pasen como su
    # propia línea (cf. doc del módulo).
    min_line_height_frac: float = 0.55

    # Clausura morfológica vertical PRE-detección: fusiona cuerpo y
    # descendentes en una sola masa. Tamaño = `kernel_vfrac × text_h`.
    # Si lo subes, evitas el doble pico cuerpo/descendentes pero
    # arriesgas fundir líneas adyacentes con interlineado pequeño;
    # 0.3 es el compromiso correcto para cursiva moderna (mantiene el
    # gap interlineal mientras une cuerpo + descendientes).
    morph_close_vfrac:    float = 0.30

    # Suavizado de la proyección horizontal (en unidades de text_h).
    proj_smooth_vfrac:    float = 0.15

    # Para considerar una fila como "vacía", la proyección debe caer
    # por debajo de `gap_frac × max_proj`. Sube el valor para fusionar
    # líneas casi pegadas; bájalo para separar líneas con descendentes
    # invadiendo la siguiente.
    gap_frac:             float = 0.08

    # Si una banda detectada supera `split_band_factor × text_h` de
    # altura, asumimos que son dos (o más) líneas pegadas que el closing
    # no separó. Se intenta partir buscando el valle más profundo
    # dentro de la banda. Repetimos hasta que ninguna sub-banda lo
    # supere o no haya un valle válido.
    split_band_factor:    float = 1.7

    # Padding vertical aplicado a cada línea detectada antes de recortar
    # (en píxeles). Pequeño porque line_preprocess hará su propio trim.
    pad_vertical:         int   = 6


# ----------------------------------------------------------------------------
# Estructuras de salida
# ----------------------------------------------------------------------------

@dataclass
class ManuscriptDetection:
    """Análoga a `PipelineResult` pero específica para esta detección."""
    binary:           np.ndarray
    gray_deskewed:    np.ndarray
    deskew_angle:     float
    line_boxes:       list[tuple[int, int, int, int]]
    block_boxes:      list[tuple[int, int, int, int]]
    text_height_est:  int
    warnings:         list[str] = field(default_factory=list)

    @property
    def n_lines(self) -> int:
        return len(self.line_boxes)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _to_gray_best(img: np.ndarray, pick_best: bool) -> np.ndarray:
    """Como en line_preprocess._to_gray_best: elige el canal con más contraste."""
    if img.ndim == 2:
        return img.copy()
    if not pick_best:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    candidates = {
        "blue":  img[:, :, 0],
        "green": img[:, :, 1],
        "red":   img[:, :, 2],
        "gray":  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
    }
    def _contrast(ch):
        p5, p95 = np.percentile(ch, 5), np.percentile(ch, 95)
        return float(p95 - p5)
    contrasts = {k: _contrast(v) for k, v in candidates.items()}
    best = max(contrasts, key=contrasts.get)
    if contrasts[best] >= contrasts["gray"] * 1.15:
        return candidates[best]
    return candidates["gray"]


def _estimate_text_height(gray: np.ndarray) -> int:
    """
    Altura típica de UNA línea de texto. Estrategia en dos pasos:

      Paso 1 — Estimación gruesa por componentes conectados:
        Otsu binariza, sacamos las alturas de los componentes filtrando
        ruido y unicornios (marcos enteros). El p70 de las alturas da
        una primera aproximación robusta a `text_h`.

      Paso 2 — Refinado con proyección horizontal suavizada:
        Usamos el `text_h` grueso para calibrar el suavizado de la
        proyección. La separación mediana entre picos da la "altura
        de línea" (centro a centro = texto + interlineado). Tomamos
        el 65% de eso como `text_h` final.

      Si el paso 2 falla (una sola línea o picos inconsistentes),
      caemos a la estimación gruesa del paso 1.

    Esta estrategia evita los dos modos de fallo conocidos:
      (a) Smoothing fijo demasiado bajo → muchos picos falsos por
          descendentes → text_h artificialmente bajo.
      (b) Smoothing fijo demasiado alto → picos fusionados → text_h
          artificialmente alto.
    """
    H, W = gray.shape
    _, rough = cv2.threshold(gray, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Paso 1: componentes conectados
    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
        rough, connectivity=8,
    )
    rough_text_h = 0
    if n_labels >= 2:
        heights = stats[1:, cv2.CC_STAT_HEIGHT]
        areas   = stats[1:, cv2.CC_STAT_AREA]
        area_min = max(20, H * W // 100000)
        area_max = H * W // 10
        mask = (areas >= area_min) & (areas <= area_max)
        h_filt = heights[mask]
        if len(h_filt) > 0:
            rough_text_h = int(np.percentile(h_filt, 70))

    if rough_text_h < 15:
        rough_text_h = max(20, H // 30)

    # Paso 2: refinado por proyección. El smoothing se basa en el
    # rough_text_h: queremos suavizar a una escala de ~30% del text_h
    # para fundir descendentes con cuerpo sin emborronar separaciones
    # entre líneas reales (que suelen ser de 30-50% del text_h).
    row_ink  = rough.sum(axis=1).astype(np.float64)
    smooth_w = max(5, int(rough_text_h * 0.5))
    proj     = uniform_filter(row_ink, size=smooth_w)
    peak_h   = proj.max() * 0.15
    # min_distance también escala con rough_text_h: dos líneas no pueden
    # estar más cerca de 0.7 × text_h
    min_dist = max(10, int(rough_text_h * 0.7))

    if proj.max() > 0:
        peaks, _ = find_peaks(proj, height=peak_h, distance=min_dist)
        if len(peaks) >= 2:
            gaps = np.diff(peaks)
            line_spacing = float(np.median(gaps))
            text_h = int(line_spacing * 0.65)
            return max(15, text_h)
        # 1 pico solo: probablemente una única línea. Usamos su rango.
        rows = np.where(proj > proj.max() * 0.10)[0]
        if rows.size > 0:
            return max(20, int(rows[-1] - rows[0] + 1))

    # Sin información de proyección: paso 1
    return max(20, rough_text_h)


def _deskew_grayscale(
    gray: np.ndarray, max_angle: float
) -> tuple[np.ndarray, float]:
    """
    Estima el ángulo global usando Otsu + ajuste de baseline y rota el
    grayscale con interpolación cúbica. Mismo método que line_preprocess
    pero a escala de página (no solo línea).
    """
    if gray.size == 0:
        return gray, 0.0

    _, rough = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, angle_deg = rotate_strip_by_baseline(rough, max_angle_deg=max_angle)
    if abs(angle_deg) < 0.3:
        return gray, 0.0

    H, W = gray.shape
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg = int(np.median(border))
    cx, cy = W / 2.0, H / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    cos_a = abs(float(M[0, 0]));  sin_a = abs(float(M[0, 1]))
    new_W = int((H * sin_a) + (W * cos_a))
    new_H = int((H * cos_a) + (W * sin_a))
    M[0, 2] += (new_W / 2.0) - cx
    M[1, 2] += (new_H / 2.0) - cy

    rotated = cv2.warpAffine(
        gray, M, (new_W, new_H),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=bg,
    )
    return rotated, angle_deg


def _binarize(gray: np.ndarray, window: int, k: float,
              pre_blur: float) -> np.ndarray:
    """Sauvola directa, sin retry (la imagen es la página completa, el retry
    de line_preprocess pierde sentido aquí porque las distintas líneas
    pueden requerir distintos k)."""
    return sauvola(gray, window=window, k=k, pre_blur=pre_blur)


def _detect_line_bands(
    binary:        np.ndarray,
    text_h:        int,
    cfg:           ManuscriptLineConfig,
) -> list[tuple[int, int]]:
    """
    Devuelve la lista de bandas (y_top, y_bot) detectadas como líneas.

    Paso a paso:
      1. Clausura morfológica vertical para FUSIONAR cuerpo+descendentes
         de cada línea en una sola masa.
      2. Proyección horizontal: cuántos píxeles de tinta hay por fila.
      3. Suavizado.
      4. Detección de regiones contiguas por encima de `gap_frac × max`.
      5. Filtrado por altura mínima.
    """
    H, W = binary.shape
    if H < text_h * 2:
        # Documento demasiado bajo para tener varias líneas; tratamos
        # toda la imagen como una sola línea.
        return [(0, H)]

    # 1. Clausura vertical: une cuerpo y descendentes
    fg = (binary < 128).astype(np.uint8)
    k_h = max(3, int(round(text_h * cfg.morph_close_vfrac)))
    if k_h % 2 == 0:
        k_h += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_h))
    closed = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

    # 2 + 3. Proyección horizontal suavizada
    proj = closed.sum(axis=1).astype(np.float64)
    smooth = max(3, int(round(text_h * cfg.proj_smooth_vfrac)))
    proj = uniform_filter(proj, size=smooth)

    if proj.max() <= 0:
        return []

    # 4. Bandas: regiones consecutivas por encima del umbral
    thr = proj.max() * cfg.gap_frac
    above = proj > thr
    bands: list[tuple[int, int]] = []
    in_band = False
    start_y = 0
    for y in range(H):
        if above[y] and not in_band:
            in_band = True
            start_y = y
        elif not above[y] and in_band:
            in_band = False
            bands.append((start_y, y))
    if in_band:
        bands.append((start_y, H))

    # 5. Filtrar bandas demasiado bajas (probablemente ruido o un
    # descendiente suelto que el closing no atrapó)
    min_line_h = max(8, int(round(text_h * cfg.min_line_height_frac)))
    bands = [b for b in bands if (b[1] - b[0]) >= min_line_h]

    # Si una banda es enorme (> split_band_factor × text_h), probablemente
    # son N líneas fusionadas que el closing pegó. Buscamos el valle más
    # profundo dentro y partimos. Aplicado recursivamente hasta que
    # ninguna sub-banda lo supere o no haya un valle válido que respete
    # min_line_h en ambos lados. Sin tope explícito de profundidad porque
    # cada iteración o bien parte (decreciendo el tamaño) o se rinde, así
    # que la recursión converge en O(log N) iteraciones.
    split_threshold = cfg.split_band_factor * text_h

    def _split_recursive(yt: int, yb: int) -> list[tuple[int, int]]:
        height = yb - yt
        if height < split_threshold:
            return [(yt, yb)]
        # Buscar el mínimo del proj dentro de la parte central de la banda
        # (no muy cerca de los bordes, para no separar un descendiente
        # del cuerpo de su línea).
        margin = max(min_line_h // 2, int(height * 0.2))
        lo = yt + margin
        hi = yb - margin
        if hi - lo < 3:
            return [(yt, yb)]
        rel_valley = int(np.argmin(proj[lo:hi]))
        valley     = lo + rel_valley
        # Verificar que el valle es lo bastante "valle": su valor debe
        # ser claramente menor que la media de la banda
        band_mean  = float(np.mean(proj[yt:yb]))
        valley_v   = float(proj[valley])
        if valley_v >= band_mean * 0.85:
            return [(yt, yb)]
        # Y que ambos lados respetan min_line_h
        if (valley - yt) < min_line_h or (yb - valley) < min_line_h:
            return [(yt, yb)]
        # Recursión sobre las dos mitades
        return _split_recursive(yt, valley) + _split_recursive(valley, yb)

    refined: list[tuple[int, int]] = []
    for (yt, yb) in bands:
        refined.extend(_split_recursive(yt, yb))

    return refined


# ----------------------------------------------------------------------------
# Entrada pública
# ----------------------------------------------------------------------------

def detect_manuscript_lines(
    img_bgr: np.ndarray,
    cfg:     Optional[ManuscriptLineConfig] = None,
) -> ManuscriptDetection:
    """
    Detecta líneas en una página manuscrita.

    Pasos:
      1. Grayscale (canal con más contraste si es color).
      2. Denoise bilateral suave.
      3. Normalización de iluminación.
      4. Deskew global sobre grayscale.
      5. Estimación de altura del texto.
      6. Binarización Sauvola con `k` fijo (sin retry).
      7. Detección de bandas con closing vertical + proyección.

    Devuelve `ManuscriptDetection` con la binaria deskewed, los bboxes
    de líneas, un único bloque envolvente y metadatos.
    """
    if cfg is None:
        cfg = ManuscriptLineConfig()
    warns: list[str] = []

    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Imagen vacía o no cargable")

    # 1. Grayscale
    gray = _to_gray_best(img_bgr, pick_best=cfg.pick_best_channel)

    # 2. Denoise
    if cfg.use_bilateral:
        gray = cv2.bilateralFilter(
            gray, d=cfg.bilateral_d,
            sigmaColor=cfg.bilateral_sigma, sigmaSpace=cfg.bilateral_sigma,
        )

    # 3. Normalización de fondo
    if cfg.normalize_bg:
        gray = normalize_illumination(gray)

    # 4. Deskew
    angle = 0.0
    if cfg.deskew:
        gray, angle = _deskew_grayscale(gray, max_angle=cfg.max_skew_angle)

    # 5. Altura de texto
    text_h = _estimate_text_height(gray)

    # 6. Binarización
    win = cfg.sauvola_window if cfg.sauvola_window > 0 \
          else max(15, int(text_h * 1.2))
    if win % 2 == 0:
        win += 1
    binary = _binarize(gray, window=win, k=cfg.sauvola_k,
                       pre_blur=cfg.sauvola_pre_blur)

    # 7. Detección de bandas
    bands = _detect_line_bands(binary, text_h=text_h, cfg=cfg)

    # Generar bboxes (siempre cubren ancho completo: NO recortamos
    # lateralmente — eso preservaría el problema de mask_binding_strips
    # que line_preprocess fue diseñado a evitar)
    H, W = binary.shape
    line_boxes: list[tuple[int, int, int, int]] = []
    pad = cfg.pad_vertical
    for (yt, yb) in bands:
        y_top = max(0, yt - pad)
        y_bot = min(H, yb + pad)
        line_boxes.append((y_top, y_bot, 0, W))

    # Un único bloque envolviendo todas las líneas (compat con frontend)
    if line_boxes:
        block = (
            min(b[0] for b in line_boxes),
            max(b[1] for b in line_boxes),
            0, W,
        )
        block_boxes = [block]
    else:
        block_boxes = [(0, H, 0, W)]
        warns.append("No se detectaron líneas en el manuscrito.")

    return ManuscriptDetection(
        binary=binary,
        gray_deskewed=gray,
        deskew_angle=angle,
        line_boxes=line_boxes,
        block_boxes=block_boxes,
        text_height_est=text_h,
        warnings=warns,
    )
