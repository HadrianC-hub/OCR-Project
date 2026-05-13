"""
apps/ocr/manuscript_predictor.py — Inferencia del modelo HTR manuscrito.
============================================================================

Define la arquitectura CRNN-Lite v2 (entrenada por el usuario en Kaggle) y
expone la clase ``HTRPredictor`` con la **misma interfaz** que
``OCRPredictor`` de ``ocr_predict.py``::

    pred = HTRPredictor(checkpoint_path="models/manuscript/best_model.pt")
    texto = pred.predict("crop_de_linea.png")

De ese modo ``apps/ocr/ocr_engine.py`` puede usar uno u otro predictor
intercambiablemente según el tipo de documento (impreso / manuscrito).

Decoder compartido
------------------
La clave de este módulo es que **no reimplementa el beam search**.
Reutiliza ``decode_greedy`` y ``decode_beam`` definidos en ``ocr_predict``
(módulo al nivel raíz del proyecto), añadiendo simplemente el parámetro
``blank_idx=0`` para esta arquitectura (el modelo CRNN-Lite v2 coloca el
blank al *principio* del vocabulario, mientras que el modelo impreso lo
coloca al final). Eso implica que el modelo de lenguaje ARPA (kenLM) y
toda la lógica de length-norm, blank-bonus, etc. se aplican exactamente
igual a manuscritos que a impresos.

Post-procesado
--------------
La corrección ortográfica (SymSpell + BETO opcional) se aplica en
``ocr_engine._ocr_full_page`` / ``ocr_regions`` a través de
``apps.ocr.spell_correct.correct_text``, idéntica para ambos modelos.

Arquitectura CRNN-Lite v2 (≈3.2 M parámetros)
---------------------------------------------
::

  conv_init 3×3   →  (B,  32, 64, W)
  layer1 pool(2,2)→  (B,  32, 32, W/2)   ← W÷2: T=W/2 al LSTM
  layer2 pool(2,1)→  (B,  64, 16, W/2)
  layer3 pool(2,1)→  (B, 128,  8, W/2)
  layer4 AdaptAvg →  (B, 256,  1, W/2)
  squeeze+permute →  (B, W/2, 256)
  BiLSTM(256→256) →  (B, W/2, 256)
  Linear(256→V+1) →  (B, W/2, V+1)
  log_softmax+permute → (T=W/2, B, V+1)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════
# Vocabulario por defecto — DEBE coincidir con el del entrenamiento
# ═════════════════════════════════════════════════════════════════════════
# CHARSET usado durante el entrenamiento del CRNN-Lite v2.
# El modelo predice len(CHARSET) + 1 clases: el blank está en el índice 0
# y CHARSET[i] está en el índice i+1.

DEFAULT_CHARSET = list(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " $#&%*.,;:¡!¿?«»-_'\"()[]"
    "áéíóúüñÁÉÍÓÚÜÑ"
)

DEFAULT_IMG_HEIGHT = 64        # H que vio el modelo durante el entrenamiento
DEFAULT_IMG_WIDTH  = 256       # W de entrenamiento (referencia; en inferencia
                               # se acepta W variable: la red es totalmente
                               # convolucional en el eje horizontal gracias al
                               # AdaptiveAvgPool2d((1, None) del layer4).

HTR_BLANK_IDX = 0              # ⚠ Distinto del modelo impreso (que usa 100).


# ═════════════════════════════════════════════════════════════════════════
# BLOQUE 1 — ARQUITECTURA CRNN-Lite v2
# ═════════════════════════════════════════════════════════════════════════

class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out  = F.relu(self.bn1(self.conv1(x)))
        out  = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class _CNNFeatureExtractor(nn.Module):
    """CNN aligerada con pool(2,2) en layer1. Ver docstring del módulo."""

    def __init__(self, input_channels: int = 1, hidden_channels: int = 32):
        super().__init__()

        self.conv_init = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        c = hidden_channels  # 32

        # layer1: pool(2,2) — reduce H y W a la mitad → T = W/2 al LSTM
        self.layer1 = nn.Sequential(
            _ResidualBlock(c, c),
            _ResidualBlock(c, c),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        # layer2 y layer3: pool(2,1) — sólo reducen H, mantienen W
        self.layer2 = nn.Sequential(
            _ResidualBlock(c, c * 2, stride=1),
            _ResidualBlock(c * 2, c * 2),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        c = c * 2  # 64

        self.layer3 = nn.Sequential(
            _ResidualBlock(c, c * 2, stride=1),
            _ResidualBlock(c * 2, c * 2),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )
        c = c * 2  # 128

        # layer4: colapsa H a 1 con AdaptiveAvgPool (permite W variable)
        self.layer4 = nn.Sequential(
            _ResidualBlock(c, c * 2, stride=1),
            _ResidualBlock(c * 2, c * 2),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        c = c * 2  # 256

        self.out_channels = c

    def forward(self, x):
        x = self.conv_init(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.squeeze(2)         # (B, 256, W/2)
        x = x.permute(0, 2, 1)   # (B, W/2, 256)
        return x


class _BiLSTMBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_features = hidden_size * 2

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class CRNN_HTR(nn.Module):
    """
    CRNN-Lite v2 para reconocimiento de texto manuscrito.

    Devuelve log-probs con forma (T, B, num_classes+1), donde la última
    dimensión es el vocabulario más el token blank (índice 0).
    """

    def __init__(self, num_classes: int, img_height: int = 64,
                 hidden_channels: int = 32, lstm_hidden: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.cnn    = _CNNFeatureExtractor(1, hidden_channels)
        cnn_out     = self.cnn.out_channels  # 256
        self.bilstm = _BiLSTMBlock(cnn_out, lstm_hidden, num_layers=1, dropout=dropout)
        self.dropout    = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes + 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features  = self.cnn(x)               # (B, T, 256)
        recurrent = self.bilstm(features)     # (B, T, 256)
        recurrent = self.dropout(recurrent)
        logits    = self.classifier(recurrent)  # (B, T, V+1)
        log_probs = F.log_softmax(logits, dim=2)
        return log_probs.permute(1, 0, 2)     # (T, B, V+1)


# ═════════════════════════════════════════════════════════════════════════
# BLOQUE 2 — Pre-procesado de imagen
# ═════════════════════════════════════════════════════════════════════════

def _imread_grayscale(path: Union[str, Path]) -> np.ndarray:
    """
    cv2.imread en grayscale, tolerante a rutas Unicode (Windows-safe).

    Las rutas que se generan en ``ocr_engine._predict_line`` vienen de
    ``tempfile.mkstemp``, así que no contienen caracteres no-ASCII, pero
    mantenemos la lectura unicode-safe por consistencia con el resto del
    motor.
    """
    import cv2  # noqa: PLC0415
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {path!r}")
    return img


def _preprocess_for_htr(img_path_or_array, img_height: int,
                        normalize) -> torch.Tensor:
    """
    Pre-procesado para el modelo HTR CRNN-Lite v2.

    Replica el pipeline que el autor del modelo usó en
    ``HTRInference.preprocess`` (ver ``clases_necesarias_para_uso.txt``):

      1. Lectura en escala de grises.
      2. Si la altura no es la esperada, redimensionar manteniendo el
         aspect-ratio (el ancho fluye a través de la red gracias al
         AdaptiveAvgPool2d((1, None)) del CNN; así aceptamos crops de
         líneas con cualquier longitud).
      3. ``tensor = img / 255``.
      4. Añadir canal (1, H, W) y normalizar con mean=0.5, std=0.5
         (mapea {0, 255} → {-1, +1}, igual que el código de entrenamiento).
      5. Añadir batch (1, 1, H, W).

    Acepta ruta (str/Path) o ndarray 2-D ya en grayscale uint8.
    """
    if isinstance(img_path_or_array, (str, Path)):
        gray = _imread_grayscale(img_path_or_array)
    elif isinstance(img_path_or_array, np.ndarray):
        arr = img_path_or_array
        if arr.ndim == 3:
            import cv2  # noqa: PLC0415
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            gray = arr
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
    else:
        raise TypeError(
            f"_preprocess_for_htr espera str/Path/ndarray, recibido "
            f"{type(img_path_or_array).__name__}"
        )

    # Asegurar altura objetivo (los crops del pipeline ya vienen a 64 px,
    # pero si llega algo distinto preservamos el aspect-ratio).
    h, w = gray.shape
    if h != img_height:
        import cv2  # noqa: PLC0415
        new_w = max(4, int(round(w * img_height / h)))
        gray = cv2.resize(gray, (new_w, img_height),
                          interpolation=cv2.INTER_CUBIC)

    tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)  # (H, W)
    tensor = tensor.unsqueeze(0)        # (1, H, W) — canal
    tensor = normalize(tensor)          # Normalize(mean=0.5, std=0.5)
    tensor = tensor.unsqueeze(0)        # (1, 1, H, W) — batch
    return tensor


# ═════════════════════════════════════════════════════════════════════════
# BLOQUE 3 — HTRPredictor (interfaz idéntica a OCRPredictor)
# ═════════════════════════════════════════════════════════════════════════

# Imports diferidos a nivel de método para evitar ciclos: ``ocr_predict``
# vive en la raíz del proyecto y a su vez podría querer importar utilidades
# del paquete ``apps.ocr`` en el futuro.

class HTRPredictor:
    """
    Predictor para el modelo HTR CRNN-Lite v2.

    Reutiliza el **mismo decoder** (greedy / beam-search con kenLM ARPA)
    que ``OCRPredictor``, pasando ``blank_idx=0`` para acomodar la
    convención de vocabulario del modelo manuscrito. La corrección
    ortográfica post-OCR (SymSpell + BETO opcional) se aplica fuera, en
    ``apps.ocr.ocr_engine`` → ``apps.ocr.spell_correct``.

    Parámetros
    ----------
    checkpoint_path : str | Path
        Ruta al .pt entrenado en Kaggle.
    charset : list[str] | None
        Vocabulario en el orden usado durante el entrenamiento. Si es
        ``None`` se usa ``DEFAULT_CHARSET`` (el del entrenamiento original).
        El blank NO va incluido — se asume que ocupa el índice 0 del
        tensor de salida del modelo y que CHARSET[i] está en el índice
        i+1.
    device : str | None
        ``'cuda'``, ``'cpu'`` o ``None`` (autodetectar).
    img_height : int
        Altura a la que se redimensionan las imágenes (el modelo CNN
        espera 64 píxeles de alto).
    hidden_channels : int
        Canales base del CNN. El checkpoint puede sobrescribirlo vía
        ``config.hidden_channels`` si lo guarda.
    lstm_hidden : int
        Tamaño oculto del BiLSTM. También se puede leer de ``config``.
    beam_width : int
        Ancho del beam-search (0/1 = greedy, ≥2 = beam).
    beam_bonus : float
        Bonus al token blank en beam search.

        El valor por defecto es 0.0, calibrado empíricamente sobre datos
        reales del proyecto. Históricamente estaba a 2.0 (heredado del
        decoder del modelo impreso), pero ese valor causa
        sub-decodificación masiva en manuscrito cursivo: el blank gana
        casi siempre y muchos caracteres reales se borran del output.
        En la matriz de confusión de test_final, las 10 confusiones más
        frecuentes eran TODAS deleciones (' '→DEL: 236 casos,
        'r'→DEL: 178, etc.). Bajando a 0.0 se elimina ese sesgo y el
        CER mejora ~4 puntos en evaluación sobre documentos no vistos
        (medido sobre los docs de prueba del proyecto en 2026-05-12).
    length_norm : float
        Exponente de normalización de longitud.
    lm : ArpaLM | None
        Modelo de lenguaje ya cargado. Si se pasa, se usa en el beam.
        Cuando ``ocr_engine`` carga ambos predictores comparte una sola
        instancia entre ellos para no duplicar memoria.
    lm_path : str | Path | None
        Alternativa a ``lm``: ruta al kenLM.arpa. Se ignora si ``lm``
        ya viene rellenado.
    lm_alpha : float
        Peso del modelo de lenguaje (0.3 – 0.5 típico).
    lm_max_order : int
        Orden máximo de n-gramas a cargar si ``lm_path`` se usa.
    verbose : bool
        Si True, imprime info al cargar.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        charset: Optional[List[str]] = None,
        device: Optional[str] = None,
        img_height: int = DEFAULT_IMG_HEIGHT,
        hidden_channels: int = 32,
        lstm_hidden: int = 128,
        beam_width: int = 10,
        beam_bonus: float = 0.0,    # ⇐ 2.0 antes; ver docstring
        length_norm: float = 0.65,
        lm=None,
        lm_path: Optional[Union[str, Path]] = None,
        lm_alpha: float = 0.4,
        lm_max_order: int = 2,
        verbose: bool = True,
    ):
        self.beam_width  = beam_width
        self.beam_bonus  = beam_bonus
        self.length_norm = length_norm
        self.lm_alpha    = lm_alpha
        self.img_height  = img_height

        # ── Dispositivo ────────────────────────────────────────────
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── Vocabulario ────────────────────────────────────────────
        self.charset  = list(charset) if charset is not None else list(DEFAULT_CHARSET)
        # blank → idx 0;  CHARSET[i] → idx i+1
        self.idx2char = {i + 1: c for i, c in enumerate(self.charset)}
        # Nota: NO mapeamos el blank a ningún carácter; el decoder lo
        # ignora explícitamente (blank_idx=0).
        num_classes   = len(self.charset)

        # ── Checkpoint ─────────────────────────────────────────────
        ckpt_path = Path(checkpoint_path)
        # weights_only=False es obligatorio: el .pt contiene un dict
        # con 'model_state_dict' y 'config' (no sólo tensores), y a
        # partir de PyTorch 2.6 weights_only=True es el default.
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device,
                                    weights_only=False)
        except TypeError:
            # PyTorch < 2.4 no acepta weights_only — caer al modo viejo
            checkpoint = torch.load(ckpt_path, map_location=self.device)

        cfg = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}

        # Si el checkpoint guarda config con num_classes / dimensiones,
        # respetarlo en lugar del CHARSET pasado por parámetro (para
        # tolerar futuros entrenamientos con vocabularios distintos).
        cfg_num_classes = cfg.get('num_classes')
        if cfg_num_classes is not None and cfg_num_classes != num_classes:
            logger.warning(
                "HTRPredictor: el checkpoint declara num_classes=%d pero "
                "el CHARSET tiene %d caracteres. Usando %d (del checkpoint).",
                cfg_num_classes, num_classes, cfg_num_classes,
            )
            num_classes = cfg_num_classes

        self.model = CRNN_HTR(
            num_classes     = num_classes,
            img_height      = cfg.get('img_height', img_height),
            hidden_channels = cfg.get('hidden_channels', hidden_channels),
            lstm_hidden     = cfg.get('lstm_hidden',    lstm_hidden),
        )

        # Quitar prefijo "_orig_mod." que añade torch.compile()
        state_dict = (checkpoint.get('model_state_dict')
                      if isinstance(checkpoint, dict) else None)
        if state_dict is None:
            # Compatibilidad: algunos checkpoints guardan directamente el state_dict
            state_dict = checkpoint
        if any(k.startswith('_orig_mod.') for k in state_dict):
            state_dict = {k.replace('_orig_mod.', '', 1): v
                          for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        # Transform fijo (los Normalize son lightweight, sin estado entrenable)
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

        # ── Modelo de lenguaje ─────────────────────────────────────
        # Se prioriza la instancia ya cargada (lm). Si no se da, se
        # intenta cargar desde lm_path. Si ambas son None, el beam
        # search funciona sin LM (puro CTC).
        self.lm = lm
        if self.lm is None and lm_path is not None:
            lm_path = Path(lm_path)
            if not lm_path.is_file():
                if verbose:
                    print(f"[AVISO] LM no encontrado en: {lm_path}\n"
                          "        Beam search funcionará SIN modelo de lenguaje.")
            else:
                from ocr_predict import ArpaLM  # noqa: PLC0415
                self.lm = ArpaLM(lm_path, max_order=lm_max_order, verbose=verbose)

        if verbose:
            n_params = sum(p.numel() for p in self.model.parameters())
            best_epoch = checkpoint.get('epoch', '?') if isinstance(checkpoint, dict) else '?'
            best_cer   = checkpoint.get('best_cer', None) if isinstance(checkpoint, dict) else None
            cer_str  = f"  |  CER entrenamiento: {best_cer:.4f}" if best_cer else ""
            print(f"HTRPredictor cargado: época {best_epoch}  "
                  f"|  {n_params:,} parámetros  "
                  f"|  dispositivo: {self.device}{cer_str}")

    # ──────────────────────────────────────────────────────────────
    # Predicción
    # ──────────────────────────────────────────────────────────────

    def predict(self, img_path_or_array) -> str:
        """
        Devuelve el texto reconocido en una imagen (ruta o ndarray).

        Usa el mismo decoder que ``OCRPredictor`` (ocr_predict.decode_beam /
        decode_greedy), con ``blank_idx=0`` para esta arquitectura.
        """
        # Imports diferidos: el decoder vive en ocr_predict.py al nivel raíz.
        from ocr_predict import decode_greedy, decode_beam  # noqa: PLC0415

        tensor = _preprocess_for_htr(
            img_path_or_array, self.img_height, self.normalize
        ).to(self.device)

        with torch.no_grad():
            log_probs = self.model(tensor)   # (T, 1, V+1)

        # Para HTR-Lite v2 todo el ancho es válido (T = W/2). No hay
        # padding implícito como en el modelo impreso.
        T = log_probs.shape[0]

        if self.beam_width <= 1:
            indices = log_probs[:, 0].argmax(dim=1).cpu().tolist()
            return decode_greedy(indices, self.idx2char, blank_idx=HTR_BLANK_IDX)

        lp_np = log_probs[:, 0].cpu().float().numpy()
        seq   = [lp_np[t].tolist() for t in range(T)]
        return decode_beam(
            seq, self.idx2char,
            beam_width  = self.beam_width,
            blank_bonus = self.beam_bonus,
            length_norm = self.length_norm,
            lm          = self.lm,
            lm_alpha    = self.lm_alpha,
            blank_idx   = HTR_BLANK_IDX,
        )

    def predict_batch(self, img_paths):
        return [self.predict(p) for p in img_paths]
