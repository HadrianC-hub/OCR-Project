import os
import math
from pathlib import Path
from typing import List, Tuple

import random
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, Sampler
from torchvision import transforms
from PIL import Image, ImageFilter


# --- Transformaciones de aumento ---

class RandomHorizontalScale:
    def __init__(self, scale_range: tuple = (0.55, 1.60)):
        self.lo, self.hi = scale_range

    def __call__(self, img: Image.Image) -> Image.Image:
        factor = random.uniform(self.lo, self.hi)
        w, h   = img.size
        new_w  = max(4, int(round(w * factor)))
        return img.resize((new_w, h), Image.BICUBIC)


class RandomStrokeWidth:
    """Simula variaciones de grosor de trazo con erosión/dilatación morfológica."""
    def __call__(self, img: Image.Image) -> Image.Image:
        r = random.random()
        if r < 0.20:
            return img.filter(ImageFilter.MinFilter(3))   # adelgaza trazos
        elif r < 0.40:
            return img.filter(ImageFilter.MaxFilter(3))   # engrosa trazos
        return img


class RandomNoise:
    def __init__(self, std_range: tuple = (0.0, 0.08)):
        self.lo, self.hi = std_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        std = random.uniform(self.lo, self.hi)
        return (tensor + torch.randn_like(tensor) * std).clamp(-1.0, 1.0)


# --- Constantes globales ---

DATA_DIR   = os.environ.get("OCR_DATA_DIR",   "images")
VOCAB_PATH = os.environ.get("OCR_VOCAB_PATH", "vocab/vocab.txt")
IMG_HEIGHT = 64
CNN_STRIDE = 4
BLANK_IDX  = 100


# --- Preprocesamiento de imagen ---

def autocrop_whitespace(img: Image.Image, threshold: int = 200, padding: int = 2) -> Image.Image:
    """Recorta márgenes blancos conservando un margen de `padding` píxeles."""
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


def get_font_label(img_path: Path) -> str:
    """Extrae la etiqueta de fuente del nombre de archivo (todo menos el sufijo numérico)."""
    parts = Path(img_path).stem.split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else Path(img_path).stem


# --- Vocabulario ---

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


# --- Decodificación CTC ---

def _log_add(a: float, b: float) -> float:
    """Suma estable en espacio logarítmico: log(exp(a) + exp(b))."""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def decode_ctc(indices: List[int]) -> str:
    """Decodificación greedy CTC: colapsa repetidos y elimina blanks."""
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
    Beam search CTC con normalización de longitud.

    Mantiene dos scores por hipótesis: p_b (termina en blank) y p_nb (no blank).
    Esta separación es necesaria para manejar correctamente la duplicación de
    caracteres en CTC: el mismo carácter puede repetirse si va separado por un blank.

    blank_bonus: penaliza expansiones de caracteres reales, evita hiper-segmentación.
    length_norm_alpha: exponente de normalización, análogo al GNMT length penalty.
    """
    NEG_INF = float('-inf')
    beams = {(): (0.0, NEG_INF)}  # {seq: (p_b, p_nb)}

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
                        # Repetición del último carácter: solo p_nb contribuye a ext,
                        # p_total contribuye a mantener seq (blank intermedio implícito).
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


# --- Dataset ---

class OCRDataset(Dataset):
    def __init__(
        self,
        data_dir:   str | Path | None = None,
        img_height: int  = IMG_HEIGHT,
        augment:    bool = False,
    ):
        self.root       = Path(data_dir or DATA_DIR)
        self.img_height = img_height
        self.augment    = augment

        # Caché de anchos (idx → px escalado a img_height).
        # Se llena una sola vez con precompute_widths() y se reutiliza
        # en todos los folds de CV y LOFO.
        self._width_cache: dict[int, int] = {}

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
            transforms.Lambda(lambda img: RandomStrokeWidth()(img)),
            RandomHorizontalScale(scale_range=(0.70, 1.40)),
            transforms.RandomPerspective(distortion_scale=0.05, p=0.25),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
            transforms.RandomAffine(
                degrees=3,
                translate=(0.02, 0.03),
                shear=(-3, 3),
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.Lambda(lambda t: RandomNoise(std_range=(0.0, 0.05))(t)),
        ])

    def precompute_widths(self) -> None:
        """Pre-calcula los anchos escalados de todas las imágenes.
        Llamar una sola vez antes de construir los BucketBatchSamplers;
        los folds de CV y LOFO reutilizan el caché.
        """
        n = len(self.samples)
        missing = [i for i in range(n) if i not in self._width_cache]
        if not missing:
            return
        print(f"  [OCRDataset] Precalculando anchos de {len(missing):,} imágenes...", end="", flush=True)
        h = self.img_height
        for i in missing:
            img_path, _ = self.samples[i]
            with Image.open(img_path) as im:
                orig_w, orig_h = im.size
            self._width_cache[i] = max(4, int(round(orig_w * h / orig_h)))
        print(f" listo. ({n:,} imágenes en caché)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, text = self.samples[idx]
        img = Image.open(img_path).convert("L")
        img = autocrop_whitespace(img, threshold=200, padding=2)
        w, h = img.size

        # Jitter de altura durante augmentación — obliga al modelo a ser robusto
        # a variaciones de escala vertical, no solo horizontal.
        if self.augment and torch.rand(1).item() > 0.40:
            jitter_h = random.randint(
                max(44, self.img_height // 2 + 5),
                self.img_height * 2,
            )
            jitter_w = max(4, int(round(w * jitter_h / h)))
            img = img.resize((jitter_w, jitter_h), Image.BICUBIC)
            w, h = img.size

        new_w = max(4, int(round(w * self.img_height / h)))
        img = img.resize((new_w, self.img_height), Image.BICUBIC)
        tfm = self._aug if (self.augment and torch.rand(1).item() > 0.45) else self._base
        tensor = tfm(img)
        label = torch.tensor(encode(text), dtype=torch.long)
        return tensor, label, text

    def get_font_labels(self, indices: List[int]) -> List[str]:
        return [get_font_label(self.samples[i][0]) for i in indices]


# --- NoAugSubset ---

class NoAugSubset(Subset):
    """Subset que siempre aplica la transformación base (sin augmentación).
    Se usa para el conjunto de validación aunque el dataset padre tenga augment=True.
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


# --- BucketBatchSampler ---

class BucketBatchSampler(Sampler):
    """
    Agrupa imágenes con anchos similares en el mismo batch.

    Reduce el padding horizontal desperdiciado en cada batch, lo que disminuye
    el uso de VRAM y acelera el entrenamiento. Las imágenes se ordenan por ancho,
    se dividen en n_buckets grupos y dentro de cada grupo se mezclan aleatoriamente.

    width_cache: dict {idx_global → ancho} precomputado por OCRDataset.precompute_widths().
    Si se pasa, evita reabrir miles de imágenes en cada construcción del sampler.
    """
    def __init__(self, subset, batch_size: int, shuffle: bool = True,
                 n_buckets: int = 10, seed: int = 42,
                 width_cache: dict | None = None):
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.rng        = np.random.default_rng(seed)

        if hasattr(subset, "indices"):
            base_ds  = subset.dataset
            indices  = np.array(subset.indices)
        else:
            base_ds  = subset
            indices  = np.arange(len(subset))

        h = base_ds.img_height
        cache = width_cache if width_cache is not None else getattr(base_ds, "_width_cache", None)

        if cache is not None:
            missing_mask = np.array([i not in cache for i in indices])
            n_missing = missing_mask.sum()
            if n_missing > 0:
                print(f"  [BucketBatchSampler] Calculando anchos de {n_missing:,} imágenes no cacheadas...", end="", flush=True)
                for i in indices[missing_mask]:
                    img_path, _ = base_ds.samples[i]
                    with Image.open(img_path) as im:
                        orig_w, orig_h = im.size
                    cache[i] = max(4, int(round(orig_w * h / orig_h)))
                print(" listo.")
            widths = np.array([cache[i] for i in indices])
            print(f"  [BucketBatchSampler] Anchos desde caché ({len(indices):,} imgs). "
                  f"min={widths.min()} max={widths.max()} media={widths.mean():.0f}")
        else:
            print(f"  [BucketBatchSampler] Calculando anchos de {len(indices):,} imágenes...", end="", flush=True)
            widths = np.empty(len(indices), dtype=np.int32)
            for i, idx in enumerate(indices):
                img_path, _ = base_ds.samples[idx]
                with Image.open(img_path) as im:
                    orig_w, orig_h = im.size
                widths[i] = max(4, int(round(orig_w * h / orig_h)))
            print(f" listo. Ancho min={widths.min()} max={widths.max()} media={widths.mean():.0f}")

        sorted_order   = np.argsort(widths)
        sorted_indices = sorted_order

        bucket_size = max(batch_size, len(sorted_indices) // n_buckets)
        self.batches = []
        for start in range(0, len(sorted_indices), bucket_size):
            bucket = sorted_indices[start : start + bucket_size].copy()
            if self.shuffle:
                self.rng.shuffle(bucket)
            for b_start in range(0, len(bucket), batch_size):
                b = bucket[b_start : b_start + batch_size]
                if len(b) > 0:
                    self.batches.append(b.tolist())

    def __iter__(self):
        if self.shuffle:
            order = np.arange(len(self.batches))
            self.rng.shuffle(order)
            for i in order:
                yield self.batches[i]
        else:
            yield from self.batches

    def __len__(self):
        return len(self.batches)


# --- collate_fn ---

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