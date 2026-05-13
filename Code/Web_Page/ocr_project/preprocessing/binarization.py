import numpy as np
import cv2
from scipy.ndimage import uniform_filter


DEFAULT_WINDOW = 51
DEFAULT_K      = 0.18
DEFAULT_R      = 128.0  # rango dinámico fijo para imágenes 8-bit


def sauvola(
    img_gray:         np.ndarray,
    window:           int   = DEFAULT_WINDOW,
    k:                float = DEFAULT_K,
    r:                float = DEFAULT_R,
    global_floor_pct: float = 0.0,
    pre_blur:         float = 0.6,
) -> np.ndarray:
    """
    Binarización Sauvola con desenfoque previo opcional para suavizar bordes.

    `pre_blur` aplica un Gaussian de sigma muy pequeño (≤ 1 px) sobre la
    imagen en grises ANTES de calcular media y varianza. Sauvola reacciona
    localmente a la varianza, así que si la imagen tiene ruido fino de JPEG
    (compresión, halo cromático, granulado del escáner) Sauvola amplifica
    esas micro-variaciones a píxeles binarios sueltos: el resultado son
    bordes "dentados" cuando los caracteres deberían tener trazos limpios.
    Un blur de ~0.6 σ suaviza ese ruido sin difuminar los trazos reales
    (los trazos del texto suelen tener ≥ 2 px de grosor) y deja los bordes
    de Sauvola mucho más regulares. Se aplica solo a grayscale puro; en
    binarización pura (cuando la imagen ya viene como blanco/negro) hay que
    llamar con `pre_blur=0`.
    """
    if img_gray.ndim != 2:
        raise ValueError(f"Se esperaba imagen 2D, recibido shape {img_gray.shape}")
    if window % 2 == 0:
        window += 1

    if pre_blur and pre_blur > 0.0:
        # ksize=0 deja a OpenCV elegir el tamaño a partir de sigma; con
        # σ ≤ 1 produce una ventana 3×3 efectiva, así que la pérdida de
        # detalle es despreciable.
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

    # Piso global: fuerza a fondo los píxeles por encima del percentil dado.
    # Solo se aplica si el valor de piso cae claramente por encima del umbral
    # de Otsu + margen; de lo contrario podría borrar tinta real en imágenes
    # de bajo contraste.
    if global_floor_pct > 0.0:
        floor_val = float(np.percentile(img_gray, global_floor_pct))
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
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def bilateral_denoise(
    gray:        np.ndarray,
    diameter:    int   = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    return cv2.bilateralFilter(gray, diameter, sigma_color, sigma_space)


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

    bg_f       = np.maximum(background.astype(np.float32), 1.0)
    normalized = np.clip(gray.astype(np.float32) / bg_f * 255.0, 0, 255).astype(np.uint8)
    return normalized


def _background_uniformity(gray: np.ndarray) -> float:
    """
    Mide la uniformidad de iluminación del fondo.

    Estrategia: estimar el "fondo" tomando el percentil 90 local en una
    rejilla 4×4 sobre la imagen (esos píxeles son por construcción los más
    claros de cada zona, es decir, papel sin tinta). Si la iluminación es
    uniforme, esos valores serán muy parecidos en toda la rejilla; si hay
    sombras, gradientes o vignetting, variarán mucho.

    Retorna la desviación estándar (en niveles 0-255) entre las 16 muestras
    de fondo. Valores típicos:
       <  6  → fondo muy uniforme  (escaneo plano de libro, hoja blanca)
       6-15 → fondo levemente desigual
       > 15 → iluminación irregular, sombras, encuadernación pesada
    """
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
    """
    Binariza una imagen. `method`:
      - 'auto'    : decide automáticamente. Si el fondo es uniforme
                    (escaneo limpio de libro/hoja), usa Otsu — produce
                    trazos del grosor natural sin engordarlos. Si la
                    iluminación es irregular (sombras, gradientes), usa
                    Sauvola que se adapta localmente.
      - 'otsu'    : fuerza Otsu global (rápido, trazos finos, exige fondo
                    uniforme)
      - 'sauvola' : fuerza Sauvola con auto-tune de k (robusto frente a
                    iluminación variable, pero engorda trazos un 15-25%
                    sobre el grosor natural por su propia formulación)

    Justificación del default 'auto': Sauvola es la opción "segura" pero
    produce binarios consistentemente más gruesos que Otsu en escaneos con
    fondo uniforme — su umbral local mean*(1 + k*(std/r - 1)) queda por
    encima del umbral global Otsu cuando el contraste local es alto, y eso
    convierte píxeles del borde del trazo (semioscuros por anti-aliasing del
    escáner) en tinta. En libros con fondo limpio eso se traduce en letras
    "infladas" y manchadas. Otsu evita ese sesgo a costa de no adaptarse a
    iluminación variable, así que enrutamos por uniformidad de fondo.
    """

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # normalize_illumination debe ir antes del bilateral para que este trabaje
    # sobre una imagen sin variaciones lentas de iluminación.
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
        # Umbral 12: por debajo el fondo es razonablemente uniforme y Otsu
        # gana en fidelidad de grosor; por encima Sauvola compensa mejor
        # las variaciones locales. Empíricamente 8-12 es la frontera.
        chosen = "otsu" if bg_std < 12.0 else "sauvola"

    if chosen == "otsu":
        # Otsu sobre la imagen entera: produce trazos del grosor natural
        # del raster original. El threshold sale del bimodal del histograma.
        # Un blur muy ligero (σ=0.5) reduce ruido fino del escáner sin
        # alterar el bimodal — lo mismo que hace Sauvola con pre_blur.
        gray_for_otsu = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.5, sigmaY=0.5)
        _, binary = cv2.threshold(gray_for_otsu, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # global_floor_pct sigue funcionando si el caller lo pidió
        if global_floor_pct > 0.0:
            floor_val = float(np.percentile(gray, global_floor_pct))
            otsu_thr  = float(np.unique(gray_for_otsu[binary == 255]).min()) if (binary == 255).any() else 128.0
            if floor_val > otsu_thr + 10:
                binary[gray >= floor_val] = 255
    else:  # sauvola
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
    """
    Enmascara las franjas oscuras de encuadernación o el borde del libro en los
    laterales izquierdo y derecho de la imagen binarizada. Esas franjas suelen
    aparecer como columnas con tinta en >30% de las filas (binding del libro
    comprimido al escanear).

    Algoritmo:
      1. Marcar cada columna en [0, max_x] como `binding` si su densidad de
         tinta supera `density_thr`.
      2. Enmascarar desde el borde izquierdo hasta la última columna `binding`
         detectada (inclusive). Esto cubre tanto el núcleo opaco como las
         columnas residuales tras una zona de transición — caso común de
         escáneres en los que el binding deja una raya fina más al interior.
      3. Extensión por umbral secundario: tras la última columna binding,
         seguir hacia el interior mientras la siguiente columna tenga densidad
         >= density_thr * 0.7. Captura rayas finas residuales (~0.20-0.29) que
         quedan justo por debajo del umbral principal pero claramente por
         encima del fondo limpio (<0.05). Permite hasta 2 columnas vacías
         intermedias antes de detenerse.
      4. Mismo procedimiento para el lado derecho.

    Esta limpieza se hace una sola vez en el pipeline, antes de detectar líneas
    y bloques, lo que elimina el artefacto en TODA la cadena (proyecciones,
    bbox, crops finales).

    Retorna:
        (binary_limpia, x_left_safe, x_right_safe)
        x_left_safe: primera columna válida desde la izquierda (0 si no hay binding).
        x_right_safe: última columna válida + 1 (W si no hay binding derecho).
    """
    H, W = binary.shape
    if W < 20:
        return binary, 0, W

    ink_col = (binary < 128).mean(axis=0)
    max_x   = max(1, int(W * max_frac))
    sec_thr = density_thr * 0.7  # umbral secundario para extensión

    # Lado izquierdo
    left_band = ink_col[:max_x] >= density_thr
    if left_band.any():
        rightmost = int(np.where(left_band)[0].max())
        # Extensión hacia el interior con umbral secundario
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

    # Lado derecho (simétrico)
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
    """
    Limpia un strip de línea de texto eliminando componentes conexos cuyo
    centroide Y caiga fuera de la banda principal de tinta. Esto remueve las
    "patas" de descenders o tildes de líneas adyacentes que se cuelan en los
    bordes superior/inferior del strip cuando la segmentación deja padding.

    Algoritmo:
      1. Banda principal = filas con tinta >= `band_thr` × pico de proyección H.
      2. Tolerancia vertical = `y_tol_frac` × altura de la banda.
      3. Componentes con centroide Y dentro de [main_top - tol, main_bot + tol]
         se conservan. Los demás se borran (puestos a 255).

    El centroide Y de un componente con descender (p, g, q) cae dentro de la
    banda principal porque la mayor parte de su masa está en x-height. Lo mismo
    para acentos: el centroide del componente unido (cuerpo + tilde) está dentro
    de la banda. Solo se eliminan trozos disjuntos de líneas adyacentes.

    Si la banda principal no se puede determinar de forma robusta, el strip se
    devuelve sin cambios.
    """
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

    # Aislar el bloque CONTIGUO más denso. Un fragmento de tinta aislado en un
    # solo extremo del strip (manchas del borde del escáner, tildes residuales)
    # supera el umbral pero forma una "isla" muy pequeña; si se permitiera
    # extender la banda principal hasta esa fila, `main_top`/`main_bot` se
    # estiran de borde a borde y la limpieza por banda Y deja de eliminar nada.
    # Buscamos por tanto la corrida más larga de filas consecutivas que
    # superan el umbral y la usamos como banda principal; un margen de 1 fila
    # admite bajadas puntuales (cortes de "g", "p", etc.).
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
    band_h   = main_bot - main_top
    tol      = max(2, int(band_h * y_tol_frac))
    y_min    = main_top - tol
    y_max    = main_bot + tol

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n_labels < 2:
        return strip_binary

    # Heurística adicional contra el lomo / encuadernación central: en un libro
    # escaneado a doble página queda un trazo vertical fino entre las dos
    # páginas. Cuando el separador de columnas se sitúa muy cerca de él, ese
    # trazo se cuela en el borde lateral del strip de cada línea de la página
    # vecina como uno o varios fragmentos muy estrechos (1-4 px de ancho) que
    # rotan + straighten suelen partir en trozos de altura variable.
    # Descartamos cualquier componente que cumpla:
    #   – es estrecho: w ≤ 4 px;
    #   – pega contra el borde izquierdo o derecho del strip (≤ 5 % del ancho);
    #   – está claramente desplazado del cuerpo del texto: su centroide X cae
    #     fuera del rango de centroides del resto de componentes de la banda
    #     principal (es decir, no convive con tinta real cercana en X).
    # Esta combinación es muy específica del lomo: textos legítimos cerca del
    # borde (p. ej. una "i" al principio de la línea) tienen otros componentes
    # vecinos en X (las letras siguientes), así que el centroide del fragmento
    # cae dentro del rango.
    edge_band = max(2, int(W * 0.05))

    main_band_cxs: list[float] = []
    for label_id in range(1, n_labels):
        cy = float(centroids[label_id, 1])
        if y_min <= cy <= y_max:
            x, _, w, _, _ = stats[label_id]
            if w > 4:  # solo letras "anchas" definen el cuerpo del texto
                main_band_cxs.append(float(centroids[label_id, 0]))

    keep_mask = np.zeros(n_labels, dtype=bool)
    keep_mask[0] = True  # fondo
    for label_id in range(1, n_labels):
        cy = float(centroids[label_id, 1])
        if not (y_min <= cy <= y_max):
            continue
        x, y, w, h, _ = stats[label_id]
        x_right = x + w
        is_thin    = (w <= 4)
        hugs_edge  = (x < edge_band) or (x_right > W - edge_band)
        if is_thin and hugs_edge and main_band_cxs:
            cx = float(centroids[label_id, 0])
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