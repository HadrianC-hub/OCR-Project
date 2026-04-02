"""
dataset.py  —  Dataset para pares .png/.jpg / .txt en una carpeta plana

Formato esperado
----------------
DATA_DIR/
    imagen_001.png   ← o .jpg / .jpeg
    imagen_001.txt    ← contiene la transcripción (una sola línea de texto)
    imagen_002.jpg
    imagen_002.txt
    ...

En Kaggle, DATA_DIR y VOCAB_PATH apuntan a /kaggle/input/<tu-dataset>/...
"""

import os
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# ================================================================== #
#  VARIABLES GLOBALES                                                  #
#  Pueden sobreescribirse vía variables de entorno o directamente     #
# ================================================================== #
DATA_DIR   = os.environ.get("OCR_DATA_DIR",   "images")
VOCAB_PATH = os.environ.get("OCR_VOCAB_PATH", "vocab/vocab.txt")
IMG_HEIGHT = 32
CNN_STRIDE = 4
BLANK_IDX  = 100
# ================================================================== #


def load_vocab(path: str | Path = VOCAB_PATH) -> Tuple[dict, dict]:
    """
    Lee vocab.txt línea a línea y construye char2idx / idx2char.
    Las líneas vacías se interpretan como espacio ' '.
    """
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


def decode_ctc(indices: List[int]) -> str:
    result, prev = [], None
    for idx in indices:
        if idx != BLANK_IDX and idx != prev:
            result.append(IDX2CHAR.get(idx, ""))
        prev = idx
    return "".join(result)


class OCRDataset(Dataset):
    """
    Carga pares (imagen, transcripción) desde una carpeta plana.

    Parámetros
    ----------
    data_dir : str | Path | None
    img_height : int
    augment : bool
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        img_height: int = IMG_HEIGHT,
        augment: bool = False,
    ):
        self.root = Path(data_dir or DATA_DIR)
        self.img_height = img_height
        self.augment = augment

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
                skipped_no_txt += 1
                continue
            text = txt_path.read_text(encoding="utf-8").strip()
            if not text:
                skipped_empty += 1
                continue
            if len(encode(text)) == 0:
                skipped_no_vocab += 1
                continue
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
        w, h = img.size
        new_w = max(4, int(round(w * self.img_height / h)))
        img = img.resize((new_w, self.img_height), Image.BICUBIC)
        tfm = self._aug if (self.augment and torch.rand(1).item() > 0.5) else self._base
        tensor = tfm(img)
        label = torch.tensor(encode(text), dtype=torch.long)
        return tensor, label, text


def collate_fn(batch):
    """
    Padding horizontal a la derecha hasta el ancho máximo del batch.
    """
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