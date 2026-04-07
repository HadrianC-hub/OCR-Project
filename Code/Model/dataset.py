"""
dataset.py  —  Dataset con autocrop de espacio blanco + beam search CTC corregido

Notas:
  - autocrop_whitespace(): recorta márgenes blancos antes de redimensionar
  - CNN_STRIDE = 4 (coordinado con model.py — bloque 1 y 2 hacen MaxPool 2×2)
  - decode_ctc_beam(): beam search CTC con correcciones para sobre-generación
  - get_font_label(): extrae nombre de fuente del stem del archivo
  - NoAugSubset: Subset sin augmentación (para validación en CV/LOFO)
"""

import os
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image

# ================================================================== #
DATA_DIR   = os.environ.get("OCR_DATA_DIR",   "images")
VOCAB_PATH = os.environ.get("OCR_VOCAB_PATH", "vocab/vocab.txt")
IMG_HEIGHT = 32
CNN_STRIDE = 4   # Bloque 1 (MaxPool 2×2) + Bloque 2 (MaxPool 2×2) = stride horizontal 4
BLANK_IDX  = 100
# ================================================================== #


# ------------------------------------------------------------------ #
#  Auto-recorte de márgenes blancos                                   #
# ------------------------------------------------------------------ #

def autocrop_whitespace(img: Image.Image, threshold: int = 200, padding: int = 2) -> Image.Image:
    """
    Recorta el espacio blanco alrededor del texto.
    En imágenes binarizadas: blanco=255 (fondo), negro=0 (tinta).
    """
    arr = np.array(img)
    ink_mask = arr < threshold
    rows = np.any(ink_mask, axis=1)
    cols = np.any(ink_mask, axis=0)

    if not rows.any():
        return img

    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    H, W = arr.shape
    r_min = max(0, r_min - padding)
    r_max = min(H - 1, r_max + padding)
    c_min = max(0, c_min - padding)
    c_max = min(W - 1, c_max + padding)

    return img.crop((c_min, r_min, c_max + 1, r_max + 1))


# ------------------------------------------------------------------ #
#  Etiqueta de fuente tipográfica                                     #
# ------------------------------------------------------------------ #

def get_font_label(img_path: Path) -> str:
    """
    Extrae el nombre de la fuente tipográfica del stem del archivo.
    Formato esperado: '<NombreFuente>_<número>.png'
    Ejemplo: 'TimesNewRoman_000042.png' → 'TimesNewRoman'
    Si el stem no tiene '_', devuelve el stem completo.
    """
    parts = Path(img_path).stem.split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else Path(img_path).stem


# ------------------------------------------------------------------ #
#  Vocabulario                                                        #
# ------------------------------------------------------------------ #

def load_vocab(path: str | Path = VOCAB_PATH) -> Tuple[dict, dict]:
    vocab_path = Path(path)
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de vocabulario en '{vocab_path}'.\n"
            f"Ajusta VOCAB_PATH o la variable de entorno OCR_VOCAB_PATH."
        )
    chars = []
    for line in vocab_path.read_text(encoding="utf-8").splitlines():
        c = line if line != "" else " "
        if c not in chars:
            chars.append(c)
    if len(chars) == 0:
        raise ValueError(f"El archivo de vocabulario '{vocab_path}' está vacío.")

    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char  = {i: c for i, c in enumerate(chars)}
    print(f"Vocabulario cargado: {len(chars)} símbolos desde '{vocab_path}'")
    print(f"  idx 0 → {repr(chars[0])}  |  idx {len(chars)-1} → {repr(chars[-1])}  |  blank CTC → idx {BLANK_IDX}")
    return char2idx, idx2char


CHAR2IDX, IDX2CHAR = load_vocab(VOCAB_PATH)


def encode(text: str) -> List[int]:
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]


# ------------------------------------------------------------------ #
#  Decodificación CTC                                                 #
# ------------------------------------------------------------------ #

def _log_add(a: float, b: float) -> float:
    """Suma numericamente estable en espacio logarítmico."""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def decode_ctc(indices: List[int]) -> str:
    """Decodificación greedy CTC — rápida, usada durante entrenamiento."""
    result, prev = [], None
    for idx in indices:
        if idx != BLANK_IDX and idx != prev:
            result.append(IDX2CHAR.get(idx, ""))
        prev = idx
    return "".join(result)


def decode_ctc_beam(
    log_probs_seq: List[List[float]],
    beam_width: int = 10,
    blank_bonus: float = 2.0,
    length_norm_alpha: float = 0.65,
) -> str:
    """
    Beam search CTC corregido contra sobre-generación.

      1. blank_bonus: suma al log_prob del blank en cada paso,
         penalizando inserciones.
      2. length_norm_alpha: divide el score final por len^alpha,
         evitando que secuencias largas acumulen mayor prob marginal.
    """
    NEG_INF = float('-inf')
    beams = {(): (0.0, NEG_INF)}

    for log_probs_t in log_probs_seq:
        new_beams: dict = {}

        for seq, (p_b, p_nb) in beams.items():
            p_total = _log_add(p_b, p_nb)

            for c, lp in enumerate(log_probs_t):
                if c == BLANK_IDX:
                    lp_blank = lp + blank_bonus
                    pb, pnb = new_beams.get(seq, (NEG_INF, NEG_INF))
                    new_beams[seq] = (_log_add(pb, p_total + lp_blank), pnb)
                else:
                    ext = seq + (c,)
                    if seq and seq[-1] == c:
                        pb, pnb = new_beams.get(seq, (NEG_INF, NEG_INF))
                        new_beams[seq] = (pb, _log_add(pnb, p_nb + lp))
                        pb, pnb = new_beams.get(ext, (NEG_INF, NEG_INF))
                        new_beams[ext] = (pb, _log_add(pnb, p_b + lp))
                    else:
                        pb, pnb = new_beams.get(ext, (NEG_INF, NEG_INF))
                        new_beams[ext] = (pb, _log_add(pnb, p_total + lp))

        beams = dict(
            sorted(new_beams.items(),
                   key=lambda x: _log_add(x[1][0], x[1][1]),
                   reverse=True)[:beam_width]
        )

    def _normed_score(seq):
        raw  = _log_add(beams[seq][0], beams[seq][1])
        norm = max(len(seq), 1) ** length_norm_alpha
        return raw / norm

    best = max(beams.keys(), key=_normed_score)
    return "".join(IDX2CHAR.get(c, "") for c in best)


# ------------------------------------------------------------------ #
#  Dataset                                                            #
# ------------------------------------------------------------------ #

class OCRDataset(Dataset):
    """Carga pares (imagen, transcripción) con autocrop de espacio blanco."""

    def __init__(
        self,
        data_dir:   str | Path | None = None,
        img_height: int  = IMG_HEIGHT,
        augment:    bool = False,
    ):
        self.root       = Path(data_dir or DATA_DIR)
        self.img_height = img_height
        self.augment    = augment

        if not self.root.exists():
            raise FileNotFoundError(f"La carpeta de datos no existe: '{self.root}'")

        _IMG_EXTS = [".png", ".jpg", ".jpeg"]
        _seen_stems: dict = {}
        for ext in _IMG_EXTS:
            for p in self.root.glob(f"*{ext}"):
                if p.stem not in _seen_stems:
                    _seen_stems[p.stem] = p
        all_images = sorted(_seen_stems.values(), key=lambda p: p.name)

        self.samples: List[Tuple[Path, str]] = []
        skipped_no_txt, skipped_empty, skipped_no_vocab = 0, 0, 0

        for img_path in all_images:
            txt_path = img_path.with_suffix(".txt")
            if not txt_path.exists():
                skipped_no_txt += 1; continue
            text = txt_path.read_text(encoding="utf-8").strip()
            if not text:
                skipped_empty += 1; continue
            if len(encode(text)) == 0:
                skipped_no_vocab += 1; continue
            self.samples.append((img_path, text))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No se encontraron pares válidos en '{self.root}'.\n"
                f"  Sin .txt: {skipped_no_txt} | Vacíos: {skipped_empty} | Sin vocab: {skipped_no_vocab}"
            )

        total_skipped = skipped_no_txt + skipped_empty + skipped_no_vocab
        print(
            f"Dataset cargado: {len(self.samples)} muestras desde '{self.root}'"
            + (f"  (ignorados: {total_skipped})" if total_skipped > 0 else "")
        )

        self._base = transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self._aug = transforms.Compose([
            transforms.Grayscale(1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=1, translate=(0.01, 0.01)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, text = self.samples[idx]
        img = Image.open(img_path).convert("L")
        img = autocrop_whitespace(img, threshold=200, padding=2)
        w, h = img.size
        new_w = max(4, int(round(w * self.img_height / h)))
        img = img.resize((new_w, self.img_height), Image.BICUBIC)
        tfm = self._aug if (self.augment and torch.rand(1).item() > 0.5) else self._base
        tensor = tfm(img)
        label = torch.tensor(encode(text), dtype=torch.long)
        return tensor, label, text

    def get_font_labels(self, indices: List[int]) -> List[str]:
        """Devuelve los nombres de fuente para los índices dados."""
        return [get_font_label(self.samples[i][0]) for i in indices]


# ------------------------------------------------------------------ #
#  NoAugSubset — Subset que siempre usa transformaciones base         #
# ------------------------------------------------------------------ #

class NoAugSubset(Subset):
    """
    Wrapper de torch.utils.data.Subset que ignora el flag augment
    del dataset padre y siempre aplica las transformaciones base.
    Úsalo para los conjuntos de validación en CV y LOFO.
    """
    def __getitem__(self, idx: int):
        img_path, text = self.dataset.samples[self.indices[idx]]
        img = Image.open(img_path).convert("L")
        img = autocrop_whitespace(img, threshold=200, padding=2)
        w, h = img.size
        new_w = max(4, int(round(w * self.dataset.img_height / h)))
        img = img.resize((new_w, self.dataset.img_height), Image.BICUBIC)
        tensor = self.dataset._base(img)
        label  = torch.tensor(encode(text), dtype=torch.long)
        return tensor, label, text


# ------------------------------------------------------------------ #
#  collate_fn                                                         #
# ------------------------------------------------------------------ #

def collate_fn(batch):
    """Padding horizontal a la derecha hasta el ancho máximo del batch."""
    images, labels, texts = zip(*batch)
    max_w = max(img.shape[2] for img in images)
    B, H  = len(images), images[0].shape[1]
    padded = torch.zeros(B, 1, H, max_w)
    for i, img in enumerate(images):
        padded[i, :, :, :img.shape[2]] = img

    input_lengths  = torch.tensor([img.shape[2] // CNN_STRIDE for img in images], dtype=torch.long)
    target_lengths = torch.tensor([len(lbl) for lbl in labels], dtype=torch.long)
    flat_labels    = torch.cat(labels)
    return padded, flat_labels, input_lengths, target_lengths, list(texts)