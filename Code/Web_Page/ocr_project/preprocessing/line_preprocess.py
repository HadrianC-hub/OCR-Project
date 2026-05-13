"""
Pipeline ESPECIALIZADO para preprocesar imágenes que YA son una sola línea
manuscrita recortada — el caso típico de un dataset de OCR línea-a-línea.

Diferencia clave respecto a `pipeline.run`:

  · NUNCA recorta lateralmente. Conserva el ancho completo de la imagen de
    entrada en todas las etapas. El recorte solo se aplica en vertical.
  · No usa `mask_binding_strips` (esa función trata columnas densas como
    encuadernación de libro y se come letras altas como Y, S, J al principio
    o final de una línea recortada).
  · No segmenta bloques ni busca columnas. Asume que la entrada es una sola
    franja horizontal de texto.
  · La limpieza de ruido respeta cualquier componente conexo que toque el
    borde izquierdo o derecho (probable resto de letra parcialmente cortada
    por el recortado previo de la línea).

Pasos:

  1. Conversión a grayscale eligiendo el canal con más contraste tinta/papel
     (azul → canal R ofrece más separación que el promedio gris).
  2. Denoising bilateral suave para limpiar textura del papel sin difuminar
     trazos.
  3. Normalización de iluminación (compensar sombras, gradientes del escáner).
  4. Estimación de inclinación a partir de centroides Y por rebanada vertical
     y rotación de la imagen grayscale con interpolación bicúbica (bordes
     limpios después de la binarización).
  5. Binarización Sauvola con `k` bajo (0.12) para conservar trazos finos.
  6. (Opcional) Straightening polinomial: corrige curvatura suave del baseline.
  7. Limpieza de ruido: elimina componentes < `noise_max_area`, PERO conserva
     cualquier componente que toque un borde (probable letra cortada).
  8. Recorte vertical (solo top/bottom, NUNCA lateral) con margen para
     ascenders/descenders.
  9. Redimensionado a altura objetivo manteniendo la relación de aspecto.

Salida: imagen uint8 con fondo blanco (255) y tinta negra (0), lista para
alimentar a un encoder convolucional de OCR.
"""

from dataclasses import dataclass, field
from typing      import Optional, Union
from pathlib     import Path

import cv2
import numpy as np

from preprocessing.binarization    import sauvola, normalize_illumination
from preprocessing.line_processing import rotate_strip_by_baseline, straighten_line


# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------

@dataclass
class LineConfig:
    """
    Configuración del preprocesador de líneas. Cada bloque está agrupado por
    etapa del pipeline; los valores por defecto cubren handwriting típico
    (bolígrafo o lápiz sobre papel blanco, ~60-100 px de altura de texto).
    """
    # --- Salida ---
    target_height:     int   = 64     # altura final fija (divisible por 16, OCR-friendly)
    min_output_width:  int   = 32     # ancho mínimo para evitar tensores degenerados

    # --- Conversión inicial ---
    pick_best_channel: bool  = True   # elegir canal RGB con mayor contraste
    use_bilateral:     bool  = True   # denoising bilateral (preserva bordes)
    bilateral_d:       int   = 5
    bilateral_sigma:   float = 40.0
    normalize_bg:      bool  = True   # corregir iluminación no uniforme

    # --- Deskew ---
    deskew:            bool  = True
    max_skew_angle:    float = 8.0
    straighten:        bool  = True   # corrección polinomial post-binarización

    # --- Binarización (Sauvola) ---
    sauvola_window:    int   = 0      # 0 = auto basado en altura de texto
    sauvola_k:         float = 0.12   # bajo: conserva trazos finos
    sauvola_pre_blur:  float = 0.6
    target_ink_range:  tuple = (0.04, 0.22)   # rango aceptable; reintento si falla

    # --- Limpieza de ruido ---
    remove_noise:      bool  = True
    noise_max_area:    int   = 8       # componentes ≤ N px² son ruido
    preserve_edges:    bool  = True    # NO eliminar componentes que tocan el borde lateral
    remove_rule_lines: bool  = True    # eliminar rayas horizontales (líneas de cuaderno)
    rule_line_max_h:    int   = 3       # altura máxima en px de un trozo de pauta
    rule_min_coverage:  float = 0.20    # longitud mínima de raya = fracción × ancho imagen
    rule_max_angle_deg: float = 4.0     # tolerancia angular para considerar una recta como pauta

    # --- Recorte vertical (NO HORIZONTAL — pero ver `vertical_margin`) ---
    vertical_trim:     bool  = True
    vertical_margin:   int   = 6       # padding arriba/abajo tras recortar

    # --- Debug ---
    debug:             bool  = False


@dataclass
class LineResult:
    """
    Resultado del preprocesado.

    Atributos
    ---------
    image
        Imagen final uint8 (fondo blanco, tinta negra) ya redimensionada a
        `cfg.target_height`. Lista para entrenar OCR.
    binary_full
        Binaria a resolución completa post-rotación, antes del resize final.
        Útil para inspección visual o entrenamiento con resolución variable.
    angle_deg
        Ángulo de deskew aplicado (grados, positivo CCW).
    text_height_est
        Altura estimada del texto en píxeles (post-rotación).
    warnings
        Lista de avisos no fatales (binarización dudosa, etc).
    """
    image:           np.ndarray
    binary_full:     np.ndarray
    angle_deg:       float = 0.0
    text_height_est: int   = 0
    warnings:        list  = field(default_factory=list)


# ----------------------------------------------------------------------------
# Conversión de color
# ----------------------------------------------------------------------------

def _to_gray_best(img: np.ndarray, pick_best: bool) -> np.ndarray:
    """
    Convierte a grayscale.

    Si `pick_best` y la imagen es color, elige el canal con mayor rango
    dinámico (p95 − p5). Para tinta azul, el canal R suele ofrecer ~30 %
    más contraste que el promedio gris porque la tinta absorbe rojo más
    fuertemente. Para tinta negra, los tres canales dan resultados similares
    y la métrica se queda con gray.
    """
    if img.ndim == 2:
        return img.copy()

    if not pick_best:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OpenCV usa orden BGR
    candidates = {
        "blue":  img[:, :, 0],
        "green": img[:, :, 1],
        "red":   img[:, :, 2],
        "gray":  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
    }

    def _contrast(ch: np.ndarray) -> float:
        p5, p95 = np.percentile(ch, 5), np.percentile(ch, 95)
        return float(p95 - p5)

    contrasts = {k: _contrast(v) for k, v in candidates.items()}
    best      = max(contrasts, key=contrasts.get)

    # Cambiar de gray solo si la mejora es ≥ 15 % — evita oscilaciones por ruido
    if contrasts[best] >= contrasts["gray"] * 1.15:
        return candidates[best]
    return candidates["gray"]


# ----------------------------------------------------------------------------
# Estimación de altura del texto y deskew
# ----------------------------------------------------------------------------

def _estimate_text_height(gray: np.ndarray) -> int:
    """
    Altura del cuerpo del texto (rango Y de filas con tinta significativa).

    Una sola binarización Otsu y se cuentan filas con tinta ≥ 8 % del pico
    de proyección. Sirve para escalar la ventana de Sauvola y otros parámetros.
    """
    H, W = gray.shape
    _, rough  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_ink   = rough.sum(axis=1).astype(np.float64)
    peak      = float(row_ink.max())
    if peak <= 0.0:
        return max(20, H // 2)
    rows      = np.where(row_ink > peak * 0.08)[0]
    if rows.size == 0:
        return max(20, H // 2)
    return int(rows[-1] - rows[0] + 1)


def _deskew_grayscale(
    gray: np.ndarray, max_angle: float = 8.0
) -> tuple[np.ndarray, float]:
    """
    Estima la inclinación de la línea con una binarización rápida + ajuste
    de baseline (`rotate_strip_by_baseline`) y aplica la rotación inversa
    sobre la imagen GRAYSCALE con interpolación CÚBICA.

    Rotar el grayscale antes de Sauvola produce bordes mucho más limpios que
    rotar la binaria ya umbralizada (donde cada saltito de cuadrícula sería
    visible). El borde del lienzo rotado se rellena con la mediana de los
    píxeles de las 4 aristas para no introducir bandas oscuras.
    """
    if gray.size == 0:
        return gray, 0.0

    _, rough     = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, angle_deg = rotate_strip_by_baseline(rough, max_angle_deg=max_angle)
    if abs(angle_deg) < 0.3:
        return gray, 0.0

    H, W   = gray.shape
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg     = int(np.median(border))

    cx, cy = W / 2.0, H / 2.0
    M      = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    cos_a  = abs(float(M[0, 0]))
    sin_a  = abs(float(M[0, 1]))
    new_W  = int((H * sin_a) + (W * cos_a))
    new_H  = int((H * cos_a) + (W * sin_a))
    M[0, 2] += (new_W / 2.0) - cx
    M[1, 2] += (new_H / 2.0) - cy

    rotated = cv2.warpAffine(
        gray, M, (new_W, new_H),
        flags       = cv2.INTER_CUBIC,
        borderMode  = cv2.BORDER_CONSTANT,
        borderValue = bg,
    )
    return rotated, angle_deg


# ----------------------------------------------------------------------------
# Binarización robusta
# ----------------------------------------------------------------------------

def _binarize_with_retry(
    gray:        np.ndarray,
    window:      int,
    k_init:      float,
    pre_blur:    float,
    target_ink:  tuple[float, float],
    max_iter:    int   = 4,
) -> tuple[np.ndarray, float]:
    """
    Sauvola con auto-ajuste de `k`: si la fracción de tinta sale fuera del
    rango deseado, baja `k` (más tinta) o lo sube (menos tinta) hasta entrar
    o agotar iteraciones.

    Tracking del mejor candidato: si ninguna iteración cae en el rango, se
    devuelve la que más cerca quedó del centro del rango.
    """
    lo, hi = target_ink
    center = 0.5 * (lo + hi)

    k       = k_init
    best    = None
    best_d  = float("inf")

    for _ in range(max_iter):
        binary    = sauvola(gray, window=window, k=k, pre_blur=pre_blur)
        ink_ratio = float((binary < 128).mean())

        d = abs(ink_ratio - center)
        if d < best_d:
            best_d = d
            best   = (binary, k, ink_ratio)

        if lo <= ink_ratio <= hi:
            return binary, k

        # Ajuste suave; clamp para evitar fugas a valores extremos
        if ink_ratio > hi:
            k = min(0.40, k * 1.15)
        else:  # ink_ratio < lo
            k = max(0.05, k * 0.85)

    assert best is not None
    return best[0], best[1]


# ----------------------------------------------------------------------------
# Limpieza post-binarización
# ----------------------------------------------------------------------------

def _remove_horizontal_rule_lines(
    binary:               np.ndarray,
    max_h:                int,
    min_coverage_frac:    float,
    max_angle_deg:        float = 4.0,
) -> np.ndarray:
    """
    Elimina rayas de pauta de cuaderno (rule lines), incluso si están
    ligeramente inclinadas respecto al texto.

    Las pautas son trazos horizontales muy delgados (1-3 px) que recorren la
    página. Cuando el texto cruza la pauta, ésta queda partida en muchos
    fragmentos cortos. Los fragmentos pueden estar desalineados en Y porque la
    pauta no es perfectamente horizontal y porque el deskew se ajusta al
    baseline del texto (no a la pauta).

    Estrategia robusta a inclinación:

      1. Aislar componentes conexos "delgados" (altura ≤ `max_h`) —
         candidatos a fragmento de pauta. Letras, acentos, puntos sobre "i",
         comas, puntos finales (h ≥ 4-5 px) quedan protegidos desde el inicio.
      2. Clausura morfológica horizontal SUAVE sobre los píxeles delgados
         para unir fragmentos vecinos en la misma fila.
      3. Detección Hough de segmentos casi horizontales: una raya verdadera
         genera líneas con |ángulo| ≤ `max_angle_deg` y longitud ≥
         `min_coverage_frac × W`. Letras y puntuación nunca producen segmentos
         tan largos.
      4. Construir una banda alrededor de cada raya detectada y eliminar
         todos los componentes delgados que la atraviesen. La banda tiene
         grosor `max_h + 2` para abarcar la raya y su antialias.

    Esta aproximación tolera inclinación arbitraria de la pauta sin afectar
    al texto.
    """
    H, W = binary.shape
    if H < 4 or W < 60 or max_h < 1:
        return binary

    fg = (binary < 128).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n_labels < 2:
        return binary

    # Componentes delgados — candidatos a fragmento de pauta
    thin_ids = [lid for lid in range(1, n_labels) if stats[lid, 3] <= max_h]
    if not thin_ids:
        return binary

    thin_mask = np.isin(labels, thin_ids).astype(np.uint8) * 255

    # Clausura horizontal: une fragmentos consecutivos en la misma fila.
    # Kernel angosto (W/40) — solo puentea pequeños espacios entre fragmentos
    # de una raya inclinada; no une cosas que no estén ya casi colineales.
    k_w = max(8, W // 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
    closed = cv2.morphologyEx(thin_mask, cv2.MORPH_CLOSE, kernel)

    # Hough probabilístico: busca segmentos largos casi horizontales.
    # `minLineLength` exige longitud ≥ min_coverage_frac × W → letras y
    # adornos sueltos (longitud típica < 30 px) quedan filtrados.
    # `maxLineGap` tolera cortes por texto.
    min_len = max(40, int(W * min_coverage_frac))
    max_gap = max(20, W // 15)
    hough_thr = max(20, int(min_len * 0.4))

    segments = cv2.HoughLinesP(
        closed,
        rho            = 1,
        theta          = np.pi / 720,   # resolución 0.25°
        threshold      = hough_thr,
        minLineLength  = min_len,
        maxLineGap     = max_gap,
    )

    if segments is None:
        return binary

    # Filtrar por ángulo: pauta es casi horizontal. Además, extrapolamos cada
    # segmento detectado a TODO el ancho de la imagen: una pauta de cuaderno
    # casi siempre recorre la página completa, y Hough puede recortar los
    # extremos cuando los fragmentos finales son muy escasos. Extender el
    # segmento por su pendiente captura esos restos sin riesgo (la banda sigue
    # siendo angosta verticalmente: max_h + 2 px).
    rule_segments = []
    for seg in segments:
        x1, y1, x2, y2 = seg[0]
        dx = x2 - x1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, dx))
        if abs(angle) > max_angle_deg:
            continue
        # Extrapolar a [0, W-1]
        slope = (y2 - y1) / dx
        y_at_0   = int(round(y1 - slope * x1))
        y_at_W   = int(round(y1 + slope * (W - 1 - x1)))
        # Clip dentro de la imagen
        y_at_0 = max(0, min(H - 1, y_at_0))
        y_at_W = max(0, min(H - 1, y_at_W))
        rule_segments.append((0, y_at_0, W - 1, y_at_W))

    if not rule_segments:
        return binary

    # Banda de eliminación: para cada raya detectada, definir un corredor
    # vertical alrededor de la línea extrapolada. Como la pauta puede tener
    # ligera curvatura no capturada por el modelo lineal, ensanchamos la banda
    # a (2·max_h + 3) px. Los componentes altos (letras, acentos, comas,
    # puntos sobre i) siguen protegidos porque no están en `thin_ids`.
    band = np.zeros((H, W), dtype=np.uint8)
    band_thickness = max(5, 2 * max_h + 3)
    for x1, y1, x2, y2 in rule_segments:
        cv2.line(band, (x1, y1), (x2, y2), 1, thickness=band_thickness)

    # Eliminar componentes delgados cuyo centroide cae dentro de la banda
    # (criterio más tolerante que "cualquier píxel en banda": evita falsos
    # positivos cuando un trazo delgado de letra apenas roza la banda) y,
    # también, cualquier componente delgado que tenga ≥ 50 % de sus píxeles
    # dentro de la banda.
    out = binary.copy()
    for lid in thin_ids:
        x, y, w, h, _ = stats[lid]
        cy = y + h // 2
        cx = x + w // 2
        in_band_centroid = (band[cy, cx] > 0) if 0 <= cy < H and 0 <= cx < W else False

        sub_band   = band[y: y + h, x: x + w]
        sub_labels = labels[y: y + h, x: x + w]
        comp_pix   = (sub_labels == lid)
        comp_area  = comp_pix.sum()
        if comp_area == 0:
            continue
        in_band_frac = float((comp_pix & (sub_band > 0)).sum()) / float(comp_area)

        if in_band_centroid or in_band_frac >= 0.5:
            out_region = out[y: y + h, x: x + w]
            out_region[comp_pix] = 255
            out[y: y + h, x: x + w] = out_region
    return out


def _clean_speckles(
    binary:       np.ndarray,
    max_area:     int,
    preserve_edges: bool,
) -> np.ndarray:
    """
    Elimina componentes conexos con área ≤ `max_area` (típicamente ruido de
    granulado del escáner, motas de polvo o pixelitos sueltos del JPEG).

    Conserva intactos:
      · Cualquier componente con área > max_area.
      · Si `preserve_edges`: cualquier componente que toque el borde izquierdo
        o derecho de la imagen. Razón: el recortado previo de la línea pudo
        cortar una letra justo en el borde y dejar un fragmento pequeño que
        SÍ es parte del texto. Eliminarlo se comería ink real — el bug que
        precisamente queremos evitar.

    No filtra por bordes superior/inferior porque ahí los huérfanos sí suelen
    ser ruido (proyección de líneas vecinas en el recortado).
    """
    H, W = binary.shape
    if H < 3 or W < 3 or max_area < 1:
        return binary

    fg = (binary < 128).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n_labels < 2:
        return binary

    keep    = np.zeros(n_labels, dtype=bool)
    keep[0] = True  # fondo

    for lid in range(1, n_labels):
        x, y, w, h, area = stats[lid]
        if area > max_area:
            keep[lid] = True
            continue
        if preserve_edges and (x == 0 or x + w == W):
            # Toca el borde lateral → posible letra parcialmente cortada
            keep[lid] = True
            continue
        # ruido aislado: se descarta

    keep_pixels      = keep[labels]
    result           = np.full_like(binary, 255)
    result[keep_pixels & (fg > 0)] = 0
    return result


def _vertical_trim_only(binary: np.ndarray, margin: int) -> np.ndarray:
    """
    Recorta solo arriba y abajo a las filas con tinta + margen. Conserva el
    ancho completo de la imagen — el ancho NUNCA se toca aquí.
    """
    H, W = binary.shape
    rows_with_ink = np.where((binary < 128).any(axis=1))[0]
    if rows_with_ink.size == 0:
        return binary
    top = max(0, int(rows_with_ink[0])  - margin)
    bot = min(H, int(rows_with_ink[-1]) + margin + 1)
    return binary[top:bot, :]


# ----------------------------------------------------------------------------
# Redimensionado final
# ----------------------------------------------------------------------------

def _resize_to_height(img: np.ndarray, target_h: int, min_w: int) -> np.ndarray:
    """
    Redimensiona conservando relación de aspecto.

    Cuando se reduce de tamaño (scale < 1) usamos INTER_AREA (antialiasing
    correcto). Cuando se aumenta, INTER_CUBIC produce trazos más suaves que
    INTER_LINEAR.
    """
    H, W = img.shape[:2]
    if H == 0 or W == 0:
        return np.full((target_h, min_w), 255, dtype=img.dtype)
    scale  = target_h / float(H)
    new_w  = max(min_w, int(round(W * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, (new_w, target_h), interpolation=interp)


# ----------------------------------------------------------------------------
# Entrada pública
# ----------------------------------------------------------------------------

def preprocess_line(
    img:  Union[np.ndarray, str, Path],
    cfg:  Optional[LineConfig] = None,
) -> LineResult:
    """
    Procesa una imagen que ya es una sola línea manuscrita recortada.

    Acepta numpy array (BGR o grayscale) o ruta a archivo. Devuelve un
    `LineResult` con la imagen lista para OCR y la binaria a resolución plena.

    El pipeline preserva el ancho original en TODAS las etapas — solo se
    recorta arriba y abajo.
    """
    if cfg is None:
        cfg = LineConfig()

    if isinstance(img, (str, Path)):
        from preprocessing.pipeline import load_image
        img = load_image(str(img))
    if img is None or img.size == 0:
        raise ValueError("Imagen vacía o no cargable")

    warns: list[str] = []

    # 1. Grayscale (canal con mejor contraste si es color)
    gray = _to_gray_best(img, pick_best=cfg.pick_best_channel)

    # 2. Denoising suave preservando bordes
    if cfg.use_bilateral:
        gray = cv2.bilateralFilter(
            gray,
            d           = cfg.bilateral_d,
            sigmaColor  = cfg.bilateral_sigma,
            sigmaSpace  = cfg.bilateral_sigma,
        )

    # 3. Normalización de iluminación (compensa sombras del escáner)
    if cfg.normalize_bg:
        gray = normalize_illumination(gray)

    # 4. Deskew sobre grayscale (rotación con bicubic, sin aliasing)
    angle = 0.0
    if cfg.deskew:
        gray, angle = _deskew_grayscale(gray, max_angle=cfg.max_skew_angle)
        if cfg.debug and abs(angle) > 0.1:
            print(f"  [line_preprocess] deskew={angle:.2f}°")

    # 5. Estimar altura de texto y ventana de Sauvola
    text_h = _estimate_text_height(gray)
    win    = cfg.sauvola_window if cfg.sauvola_window > 0 else max(15, int(text_h * 1.2))
    if win % 2 == 0:
        win += 1

    # 6. Binarización con auto-ajuste de k
    binary, k_used = _binarize_with_retry(
        gray,
        window     = win,
        k_init     = cfg.sauvola_k,
        pre_blur   = cfg.sauvola_pre_blur,
        target_ink = cfg.target_ink_range,
    )
    if cfg.debug:
        ink_ratio = float((binary < 128).mean())
        print(f"  [line_preprocess] text_h~{text_h}px win={win} k={k_used:.3f} ink={ink_ratio:.3f}")

    # 7. Straightening polinomial de baseline (corrige curvatura suave)
    if cfg.straighten:
        binary = straighten_line(binary, poly_degree=2)

    # 8. Eliminar rayas de pauta (cuaderno con renglones)
    if cfg.remove_rule_lines:
        binary = _remove_horizontal_rule_lines(
            binary,
            max_h             = cfg.rule_line_max_h,
            min_coverage_frac = cfg.rule_min_coverage,
            max_angle_deg     = cfg.rule_max_angle_deg,
        )

    # 9. Limpieza de ruido sin tocar bordes laterales
    if cfg.remove_noise:
        binary = _clean_speckles(
            binary,
            max_area       = cfg.noise_max_area,
            preserve_edges = cfg.preserve_edges,
        )

    # 10. Recorte VERTICAL ÚNICAMENTE
    binary_trim = binary
    if cfg.vertical_trim:
        binary_trim = _vertical_trim_only(binary, margin=cfg.vertical_margin)

    # Si el recorte vertical dejó el bbox vacío (no había tinta), aviso y se
    # devuelve un lienzo blanco con tamaño objetivo para no romper el batch.
    if (binary_trim < 128).sum() == 0:
        warns.append("No se detectó tinta tras la binarización; se devuelve un lienzo en blanco.")
        out = np.full((cfg.target_height, cfg.min_output_width), 255, dtype=np.uint8)
        return LineResult(
            image           = out,
            binary_full     = binary,
            angle_deg       = angle,
            text_height_est = text_h,
            warnings        = warns,
        )

    # 11. Resize a altura objetivo, ancho proporcional al original
    out = _resize_to_height(binary_trim, target_h=cfg.target_height, min_w=cfg.min_output_width)

    return LineResult(
        image           = out,
        binary_full     = binary,
        angle_deg       = angle,
        text_height_est = text_h,
        warnings        = warns,
    )
