"""HTR para manuscritos cursivos en español basado en TrOCR fine-tuneado."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

logger = logging.getLogger(__name__)


def _is_trocr_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / 'config.json').is_file():
        return False
    has_weights = (
        (path / 'model.safetensors').is_file() or
        (path / 'pytorch_model.bin').is_file()
    )
    has_tokenizer = (
        (path / 'tokenizer.json').is_file() or
        (path / 'sentencepiece.bpe.model').is_file()
    )
    return has_weights and has_tokenizer


def _find_trocr_model(hint_path: Optional[str] = None) -> Path:
    """Localiza la carpeta del modelo TrOCR.

    Busca en este orden: variable de entorno OCR_TROCR_PATH, hint_path,
    settings.OCR_TROCR_PATH, BASE_DIR/models/trocr_es_finetuned,
    BASE_DIR/trocr_es_finetuned.
    """
    candidates: list[Path] = []

    if env := os.environ.get('OCR_TROCR_PATH'):
        candidates.append(Path(env))

    if hint_path:
        candidates.append(Path(hint_path))

    base_dir = None
    try:
        from django.conf import settings
        if hasattr(settings, 'OCR_TROCR_PATH'):
            candidates.append(Path(settings.OCR_TROCR_PATH))
        if hasattr(settings, 'BASE_DIR'):
            base_dir = Path(settings.BASE_DIR)
            candidates.append(base_dir / 'models' / 'trocr_es_finetuned')
            candidates.append(base_dir / 'trocr_es_finetuned')
    except Exception:
        cwd = Path.cwd()
        candidates.append(cwd / 'models' / 'trocr_es_finetuned')
        candidates.append(cwd / 'trocr_es_finetuned')

    for c in candidates:
        if _is_trocr_dir(c):
            return c

    msg = ['No se encontró el modelo TrOCR. Rutas probadas:']
    for c in candidates:
        if not c.exists():
            msg.append(f'  - {c}  (no existe)')
        elif not c.is_dir():
            msg.append(f'  - {c}  (no es directorio)')
        elif not (c / 'config.json').is_file():
            msg.append(f'  - {c}  (falta config.json)')
        else:
            msg.append(f'  - {c}  (faltan pesos o tokenizer)')
    msg.append('')
    msg.append('Descomprime el modelo en una de estas rutas.')
    if base_dir:
        msg.append(f'Recomendado: {base_dir / "models" / "trocr_es_finetuned"}')
    raise FileNotFoundError('\n'.join(msg))


def _cap_torch_threads(n: int = 2) -> None:
    torch.set_num_threads(n)
    os.environ.setdefault('OMP_NUM_THREADS', str(n))
    os.environ.setdefault('MKL_NUM_THREADS', str(n))


class HTRPredictor:
    """Predictor TrOCR thread-safe (singleton)."""

    _instance: Optional['HTRPredictor'] = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_beams: int = 4,
        max_length: int = 128,
        **kwargs,
    ):
        if getattr(self, '_initialized', False):
            return

        _cap_torch_threads(int(os.environ.get('OCR_TORCH_THREADS', '2')))

        self.model_path = _find_trocr_model(hint_path=model_path)
        self.num_beams = num_beams
        self.max_length = max_length

        t0 = time.perf_counter()
        logger.info(f'Cargando modelo TrOCR desde {self.model_path}')

        use_fast = (self.model_path / 'tokenizer.json').is_file()
        self.processor = TrOCRProcessor.from_pretrained(
            str(self.model_path), use_fast=use_fast
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            str(self.model_path)
        )
        self.model.eval()

        gc = self.model.generation_config
        tok = self.processor.tokenizer
        gc.decoder_start_token_id = tok.bos_token_id
        gc.pad_token_id           = tok.pad_token_id
        gc.eos_token_id           = tok.eos_token_id
        gc.bos_token_id           = tok.bos_token_id
        gc.max_length             = self.max_length
        gc.num_beams              = self.num_beams
        gc.early_stopping         = True
        gc.no_repeat_ngram_size   = 3
        gc.length_penalty         = 2.0

        self._inference_lock = threading.Lock()
        self._initialized = True

        elapsed = time.perf_counter() - t0
        logger.info(f'Modelo TrOCR cargado en {elapsed:.1f}s')

    @staticmethod
    def _prepare_image(image_path: str | Path) -> Image.Image:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def predict(self, image_path: str | Path) -> str:
        img = self._prepare_image(image_path)
        with self._inference_lock:
            pixel_values = self.processor(
                images=img, return_tensors='pt'
            ).pixel_values
            with torch.no_grad():
                ids = self.model.generate(pixel_values)
            text = self.processor.batch_decode(
                ids, skip_special_tokens=True
            )[0]
        return text.strip()

    def predict_batch(
        self,
        image_paths: Iterable[str | Path],
        batch_size: int = 4,
    ) -> list[str]:
        image_paths = list(image_paths)
        if not image_paths:
            return []

        results: list[str] = []
        with self._inference_lock:
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                imgs = [self._prepare_image(p) for p in batch_paths]
                pixel_values = self.processor(
                    images=imgs, return_tensors='pt'
                ).pixel_values
                with torch.no_grad():
                    ids = self.model.generate(pixel_values)
                texts = self.processor.batch_decode(
                    ids, skip_special_tokens=True
                )
                results.extend(t.strip() for t in texts)
        return results


_predictor: Optional[HTRPredictor] = None
_predictor_lock = threading.Lock()


def get_predictor(model_path: Optional[str] = None) -> HTRPredictor:
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                _predictor = HTRPredictor(model_path=model_path)
    return _predictor


def predict(image_path) -> str:
    return get_predictor().predict(image_path)


def predict_batch(image_paths, batch_size: int = 4) -> list[str]:
    return get_predictor().predict_batch(image_paths, batch_size=batch_size)
