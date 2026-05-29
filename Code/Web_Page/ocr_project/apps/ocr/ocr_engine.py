"""Motor OCR del proyecto. Carga los modelos de forma diferida y expone
funciones para impresos, manuscritos y regiones definidas por el usuario."""

import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing  import Iterable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# RUTAS Y CONFIG GLOBAL
# ──────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent

PRINTED_MODEL_PATH    = BASE_DIR / 'models' / 'printed'    / 'best_model.pt'
MANUSCRIPT_MODEL_PATH = BASE_DIR / 'models' / 'manuscript' / 'best_model.pt'
LM_PATH               = BASE_DIR / 'models' / 'kenLM.arpa'

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ── Cap de threads de PyTorch ─────────────────────────────────────────────
def _cap_torch_threads(n: int = 2) -> None:
    """
    Limita los threads internos de PyTorch.

    Idempotente para `set_num_threads`. `set_num_interop_threads` solo
    es válido ANTES de la primera operación tensor; lo intentamos pero
    capturamos su RuntimeError si ya hay actividad.
    """
    try:
        import torch  # noqa: PLC0415
        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(n)
        except RuntimeError:
            pass
        logger.info("PyTorch threads limitados a %d.", n)
    except ImportError:
        pass

_cap_torch_threads(int(os.environ.get('OCR_TORCH_THREADS', '2')))


# ──────────────────────────────────────────────────────────────────────────
# SINGLETONS DE PREDICTORES — CARGA DIFERIDA THREAD-SAFE
# ──────────────────────────────────────────────────────────────────────────

_printed_predictor      = None
_manuscript_predictor   = None
_arpa_lm                = None
_arpa_lm_attempted      = False

# Un lock por singleton para no serializar cargas independientes.
_printed_lock           = threading.Lock()
_manuscript_lock        = threading.Lock()
_arpa_lm_lock           = threading.Lock()


def _get_arpa_lm():
    """
    Devuelve la instancia de `ArpaLM` cargada desde `models/kenLM.arpa`,
    o `None` si el archivo no existe o falla la carga. Carga diferida y
    cacheada (una sola vez por proceso); thread-safe.
    """
    global _arpa_lm, _arpa_lm_attempted
    if _arpa_lm_attempted:
        return _arpa_lm
    with _arpa_lm_lock:
        if _arpa_lm_attempted:
            return _arpa_lm
        _arpa_lm_attempted = True

        if not LM_PATH.exists():
            logger.info(
                "kenLM no encontrado en '%s'. Beam search funcionará sin LM.",
                LM_PATH,
            )
            return None
        try:
            from ocr_predict import ArpaLM  # noqa: PLC0415
            _arpa_lm = ArpaLM(str(LM_PATH), max_order=2, verbose=True)
            logger.info("kenLM cargado desde '%s' (compartido entre OCR/HTR).",
                        LM_PATH)
        except Exception as exc:
            logger.error("Error al cargar kenLM: %s", exc, exc_info=True)
            _arpa_lm = None
        return _arpa_lm


def _get_printed_predictor():
    global _printed_predictor
    if _printed_predictor is not None:
        return _printed_predictor
    with _printed_lock:
        if _printed_predictor is not None:
            return _printed_predictor

        if not PRINTED_MODEL_PATH.exists():
            logger.warning(
                "Modelo OCR impreso no encontrado en '%s'.",
                PRINTED_MODEL_PATH,
            )
            return None
        try:
            from ocr_predict import OCRPredictor  # noqa: PLC0415
            inst = OCRPredictor(
                checkpoint_path=str(PRINTED_MODEL_PATH),
                lm_path=None,
                verbose=True,
            )
            inst.lm = _get_arpa_lm()
            _printed_predictor = inst
            logger.info("Modelo OCR impreso cargado desde '%s'.",
                        PRINTED_MODEL_PATH)
        except Exception as exc:
            logger.error("Error al cargar el modelo OCR impreso: %s",
                         exc, exc_info=True)
            return None
        return _printed_predictor


def _get_manuscript_predictor():
    """
    Carga el modelo HTR manuscrito (arquitectura CRNN-Lite v2).
    Comparte la instancia de `ArpaLM` con el predictor impreso.
    Thread-safe.
    """
    global _manuscript_predictor
    if _manuscript_predictor is not None:
        return _manuscript_predictor
    with _manuscript_lock:
        if _manuscript_predictor is not None:
            return _manuscript_predictor

        if not MANUSCRIPT_MODEL_PATH.exists():
            logger.warning(
                "Modelo HTR manuscrito no encontrado en '%s'.",
                MANUSCRIPT_MODEL_PATH,
            )
            return None
        try:
            from apps.ocr.manuscript_predictor import HTRPredictor  # noqa: PLC0415
            inst = HTRPredictor(
                checkpoint_path=str(MANUSCRIPT_MODEL_PATH),
                lm=_get_arpa_lm(),
                verbose=True,
            )
            _manuscript_predictor = inst
            logger.info("Modelo HTR manuscrito cargado desde '%s'.",
                        MANUSCRIPT_MODEL_PATH)
        except Exception as exc:
            logger.error("Error al cargar el modelo HTR manuscrito: %s",
                         exc, exc_info=True)
            return None
        return _manuscript_predictor


def _get_predictor_for(doc_type: str):
    """Devuelve el predictor adecuado o None si no hay modelo."""
    if doc_type == 'manuscript':
        return _get_manuscript_predictor()
    return _get_printed_predictor()


# ──────────────────────────────────────────────────────────────────────────
# CARGA DE IMAGEN UNICODE-SAFE
# ──────────────────────────────────────────────────────────────────────────

def _imread_unicode(path: str):
    """cv2.imread que tolera rutas con caracteres no-ASCII en Windows."""
    import cv2  # noqa: PLC0415
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {path!r}")
    return img


# ──────────────────────────────────────────────────────────────────────────
# PREPROCESADO — IMPRESO (pipeline.run, igual que antes)
# ──────────────────────────────────────────────────────────────────────────

def _preprocess_array_printed(img_bgr: np.ndarray) -> list:
    """
    Pipeline de impresos: page → blocks → lines → binarize → crop.
    Devuelve la lista `line_crops` (uint8 binarios).
    """
    from preprocessing.pipeline import auto_config, run  # noqa: PLC0415
    cfg    = auto_config(img_bgr)
    result = run(img_bgr, cfg)
    return result.line_crops


# ──────────────────────────────────────────────────────────────────────────
# PREPROCESADO — MANUSCRITO (detect_manuscript_lines + line_preprocess)
# ──────────────────────────────────────────────────────────────────────────

def _preprocess_array_manuscript(img_bgr: np.ndarray) -> list:
    """
    Pipeline de manuscrito:

      1. `detect_manuscript_lines(img)` → binario deskewed +
         lista de bboxes (y_top, y_bot, x_left, x_right) de cada línea.
         No recorta lateralmente; no usa mask_binding_strips; no
         confunde descendentes con líneas separadas.

      2. Para cada bbox, cropea el GRIS deskewed por esa banda Y
         (con el ancho completo de la página). Esa banda se pasa por
         `line_preprocess.preprocess_line()`, que produce el mismo
         tipo de imagen que se usó para entrenar el modelo
         (binaria 64 px de altura, ancho variable).

      3. Devuelve la lista de arrays uint8 listos para predict.
    """
    from preprocessing.manuscript_lines import detect_manuscript_lines  # noqa: PLC0415
    from preprocessing.line_preprocess  import preprocess_line, LineConfig  # noqa: PLC0415

    det = detect_manuscript_lines(img_bgr)
    if not det.line_boxes:
        logger.info("Manuscrito: no se detectaron líneas.")
        return []

    # El detector ya hizo deskew global; aquí solo permitimos un ajuste
    # residual pequeño por si el baseline de UNA línea quedó algo
    # inclinado tras el deskew global.
    line_cfg = LineConfig(deskew=True, max_skew_angle=2.5)

    crops: list[np.ndarray] = []
    H_full, W_full = det.gray_deskewed.shape
    for (y_top, y_bot, x_left, x_right) in det.line_boxes:
        y_top = max(0, y_top); y_bot = min(H_full, y_bot)
        x_left = max(0, x_left); x_right = min(W_full, x_right)
        if y_bot - y_top < 8 or x_right - x_left < 16:
            continue
        crop_gray = det.gray_deskewed[y_top:y_bot, x_left:x_right]
        try:
            result = preprocess_line(crop_gray, cfg=line_cfg)
        except Exception as exc:
            logger.warning("Error preprocesando línea (y=%d..%d): %s",
                           y_top, y_bot, exc)
            continue
        crops.append(result.image)

    return crops


# ──────────────────────────────────────────────────────────────────────────
# DISPATCHER + COMPATIBILIDAD
# ──────────────────────────────────────────────────────────────────────────

def _preprocess_array(img_bgr: np.ndarray,
                      doc_type: str = 'printed') -> list:
    """
    Dispatcher: elige el pipeline adecuado según el tipo de documento.

    Para `manuscript` usa el flujo nuevo (detect_manuscript_lines +
    line_preprocess). Para `printed` usa pipeline.run como siempre.
    """
    if doc_type == 'manuscript':
        return _preprocess_array_manuscript(img_bgr)
    return _preprocess_array_printed(img_bgr)


def _run_preprocessing(image_path: str, doc_type: str = 'printed') -> list:
    """Compatibilidad: preprocesa una imagen completa desde su ruta."""
    return _preprocess_array(_imread_unicode(image_path), doc_type=doc_type)


# ──────────────────────────────────────────────────────────────────────────
# PREDICT POR LÍNEA
# ──────────────────────────────────────────────────────────────────────────

def _predict_line(predictor, line_array: np.ndarray) -> str:
    """
    Persiste un crop de línea como PNG temporal y llama a
    `predictor.predict()` para obtener el texto. Persistir a fichero
    (en lugar de pasar el ndarray) replica exactamente el camino de
    inferencia del standalone.
    """
    from PIL import Image  # noqa: PLC0415

    if line_array.dtype == np.uint8:
        uint8_arr = line_array
    else:
        uint8_arr = (line_array * 255).clip(0, 255).astype(np.uint8)

    pil_img = Image.fromarray(uint8_arr, mode='L')

    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.png')
    os.close(tmp_fd)
    try:
        pil_img.save(tmp_path)
        return predictor.predict(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────────
# API PÚBLICA — PÁGINA COMPLETA
# ──────────────────────────────────────────────────────────────────────────

def _ocr_full_page(image_path: str, doc_type: str,
                   use_bert: bool = False) -> str:
    predictor = _get_predictor_for(doc_type)
    if predictor is None:
        return _fallback_text(printed=(doc_type != 'manuscript'))

    try:
        lines = _run_preprocessing(image_path, doc_type=doc_type)
    except Exception as exc:
        logger.error("Error en preprocesado para '%s': %s", image_path, exc)
        return _fallback_text(printed=(doc_type != 'manuscript'))

    if not lines:
        logger.warning("El preprocesado no detectó líneas en '%s'.", image_path)
        return ''

    predicted = []
    for idx, line_arr in enumerate(lines):
        try:
            predicted.append(_predict_line(predictor, line_arr))
        except Exception as exc:
            logger.warning("Error prediciendo línea %d de '%s': %s",
                           idx, image_path, exc)
            predicted.append('')

    from apps.ocr.spell_correct import correct_text  # noqa: PLC0415
    return correct_text('\n'.join(predicted), use_bert=use_bert)


def ocr_printed(image_path: str, use_bert: bool = False) -> str:
    """OCR para documentos impresos/tipográficos. Página completa."""
    return _ocr_full_page(image_path, 'printed', use_bert=use_bert)


def ocr_manuscript(image_path: str, use_bert: bool = False) -> str:
    """OCR para documentos manuscritos. Página completa."""
    return _ocr_full_page(image_path, 'manuscript', use_bert=use_bert)


# ──────────────────────────────────────────────────────────────────────────
# API PÚBLICA — REGIONES DEFINIDAS POR EL USUARIO
# ──────────────────────────────────────────────────────────────────────────

def ocr_regions(
    image_path: str,
    regions:    Iterable[dict],
    doc_type:   str = 'printed',
    use_bert:   bool = False,
) -> str:
    """
    Ejecuta OCR sobre una lista de regiones recortadas del facsimilar.

    Cada región es un dict con claves x, y, width, height (en píxeles del
    facsimilar original) y opcionalmente order. Se procesan en el orden
    indicado por `order` (o por su posición en la lista si no está).

    Para cada recorte:
      1. Lo extraemos de la imagen original.
      2. Pasamos el recorte por el pipeline → líneas → OCR
         (usando el dispatcher; para manuscritos cada región se trata
         como una sub-página que puede contener varias líneas).
      3. Concatenamos las líneas con '\\n'.
    Las salidas de cada región se separan con '\\n\\n'.
    """
    predictor = _get_predictor_for(doc_type)
    if predictor is None:
        return _fallback_text(printed=(doc_type != 'manuscript'))

    try:
        img_bgr = _imread_unicode(image_path)
    except Exception as exc:
        logger.error("Error cargando '%s' para OCR por regiones: %s",
                     image_path, exc)
        return f'(Error al cargar la imagen: {exc})'

    H, W = img_bgr.shape[:2]

    def _to_dict(r):
        if isinstance(r, dict):
            return r
        return {
            'order':  getattr(r, 'order', 0),
            'x':      getattr(r, 'x', 0),
            'y':      getattr(r, 'y', 0),
            'width':  getattr(r, 'width', 0),
            'height': getattr(r, 'height', 0),
        }

    region_list = [_to_dict(r) for r in regions]
    region_list.sort(key=lambda r: (r.get('order') or 0))
    if not region_list:
        return ''

    region_outputs: List[str] = []
    from apps.ocr.spell_correct import correct_text  # noqa: PLC0415

    for r_idx, r in enumerate(region_list, start=1):
        try:
            x = max(0, int(r.get('x', 0)))
            y = max(0, int(r.get('y', 0)))
            w = max(1, int(r.get('width', 0)))
            h = max(1, int(r.get('height', 0)))
        except (TypeError, ValueError):
            logger.warning("Región #%d con coordenadas inválidas, omitida.",
                           r_idx)
            continue

        x2 = min(W, x + w);  y2 = min(H, y + h)
        x  = min(x, W - 1);  y  = min(y, H - 1)
        if x2 <= x or y2 <= y:
            logger.warning("Región #%d fuera de la imagen, omitida.", r_idx)
            continue

        crop = img_bgr[y:y2, x:x2]

        try:
            line_arrays = _preprocess_array(crop, doc_type=doc_type)
        except Exception as exc:
            logger.warning("Error preprocesando región #%d: %s", r_idx, exc)
            region_outputs.append('')
            continue

        if not line_arrays:
            region_outputs.append('')
            continue

        predicted = []
        for li, line_arr in enumerate(line_arrays):
            try:
                predicted.append(_predict_line(predictor, line_arr))
            except Exception as exc:
                logger.warning("Error prediciendo línea %d de región #%d: %s",
                               li, r_idx, exc)
                predicted.append('')

        region_outputs.append(
            correct_text('\n'.join(predicted), use_bert=use_bert)
        )

    return '\n\n'.join(region_outputs)


# ──────────────────────────────────────────────────────────────────────────
# TEXTO DE RESPALDO
# ──────────────────────────────────────────────────────────────────────────

def _fallback_text(printed: bool) -> str:
    if printed:
        return (
            "(Modelo OCR impreso no disponible.\n"
            " Copia best_model.pt en ocr_project/models/printed/ y reinicia el servidor.)"
        )
    return (
        "(Modelo HTR manuscrito no disponible.\n"
        " Copia best_model.pt (CRNN-Lite v2) en ocr_project/models/manuscript/ "
        "y reinicia el servidor.)"
    )
