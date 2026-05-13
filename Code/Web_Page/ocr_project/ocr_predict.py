"""
ocr_predict.py — Inferencia del modelo OCR CRNN+CTC
=====================================================
Uso rápido:
    python ocr_predict.py imagen.png --checkpoint checkpoints/best_model.pt

Uso con modelo de lenguaje en español (decoder):
    python ocr_predict.py imagen.png --checkpoint checkpoints/best_model.pt \\
                          --lm modelos/kenLM.arpa

Uso desde Python:
    from ocr_predict import OCRPredictor
    predictor = OCRPredictor(
        "checkpoints/best_model.pt",
        lm_path="C:/mis_modelos/kenLM.arpa",  # cualquier ruta
    )
    texto = predictor.predict("imagen.png")
    print(texto)

Dependencias base:  torch, torchvision, Pillow, numpy
Decoder español:    NINGUNA dependencia extra — lector .arpa incluido en este archivo.
                    Descarga el modelo desde:
                    https://huggingface.co/kensho/5gram-spanish-kenLM/resolve/main/kenLM.arpa
                    y pásalo con --lm <ruta>.
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────
#  Constantes por defecto (se sobreescriben con los valores del ckpt)
# ─────────────────────────────────────────────────────────────────────
_DEFAULT_IMG_HEIGHT = 64
_DEFAULT_VOCAB_SIZE = 101
_BLANK_IDX          = 100   # índice del token blank en el modelo
_CNN_STRIDE         = 4


# ═════════════════════════════════════════════════════════════════════
#  MODELO CRNN
# ═════════════════════════════════════════════════════════════════════

class CRNN(nn.Module):
    """Misma arquitectura que en model.py del entrenamiento."""

    def __init__(self, vocab_size=101, img_height=64,
                 hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        assert img_height % 16 == 0, "img_height debe ser divisible entre 16"

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn  = nn.LSTM(256, hidden_size, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        mid = hidden_size
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, mid), nn.ReLU(True),
            nn.Dropout(dropout / 2),
            nn.Linear(mid, vocab_size),
        )

    def forward(self, x):
        f = self.pool(self.cnn(x))
        B, C, _, T = f.shape
        f, _ = self.rnn(f.squeeze(2).permute(0, 2, 1))
        return torch.log_softmax(self.fc(f).permute(1, 0, 2), dim=2)


# ═════════════════════════════════════════════════════════════════════
#  LECTOR .ARPA PURO PYTHON  (sin kenlm, sin compilar nada)
# ═════════════════════════════════════════════════════════════════════

class ArpaLM:
    """
    Lector de modelos de lenguaje en formato ARPA.
    Funciona con cualquier versión de Python sin dependencias adicionales.

    Solo carga unigramas y bigramas por defecto para mantener la memoria
    razonable. Con max_order=2 carga ~2.4 M entradas del archivo de 964 MB,
    que ya mejoran notablemente el beam search.

    Parámetros
    ----------
    arpa_path : str | Path
        Ruta al archivo kenLM.arpa (puede estar en cualquier carpeta).
    max_order : int
        Orden máximo de n-gramas a cargar (por defecto 2 = bigramas).
        Subir a 3 mejora calidad pero usa ~1 GB extra de RAM.
    verbose : bool
        Si True, muestra progreso de carga.
    """

    def __init__(self, arpa_path: Union[str, Path],
                 max_order: int = 2, verbose: bool = True):
        self.max_order = max_order
        # probs[order][ngram_tuple]   = log10(prob)
        self.probs:    Dict[int, Dict[tuple, float]] = {}
        # backoffs[order][ngram_tuple] = log10(backoff)
        self.backoffs: Dict[int, Dict[tuple, float]] = {}
        self._load(Path(arpa_path), verbose)

    def _load(self, path: Path, verbose: bool):
        current_order = None
        loaded = 0

        if verbose:
            size_mb = path.stat().st_size / 1_048_576
            print(f"Cargando modelo de lenguaje: {path.name}  ({size_mb:.0f} MB)")
            print(f"  Leyendo órdenes 1–{self.max_order} (puede tardar 1–2 min)...")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                # Cabecera de sección: \N-grams:
                if line.startswith("\\") and line.endswith("-grams:"):
                    try:
                        current_order = int(line[1:line.index("-")])
                    except ValueError:
                        current_order = None
                    if current_order is not None and current_order > self.max_order:
                        break   # no necesitamos órdenes superiores
                    if current_order is not None:
                        self.probs[current_order]    = {}
                        self.backoffs[current_order] = {}
                    continue

                if current_order is None or current_order > self.max_order:
                    continue
                if line.startswith("\\"):
                    continue

                # Formato ARPA: log10prob<TAB>w1 [w2 ...]<TAB>[backoff]
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                try:
                    log_prob = float(parts[0])
                except ValueError:
                    continue

                words = tuple(parts[1].split())
                if len(words) != current_order:
                    continue

                self.probs[current_order][words] = log_prob

                if len(parts) >= 3:
                    try:
                        self.backoffs[current_order][words] = float(parts[2])
                    except ValueError:
                        pass

                loaded += 1
                if verbose and loaded % 500_000 == 0:
                    print(f"  ... {loaded:,} n-gramas leídos", end="\r")

        if verbose:
            total = sum(len(v) for v in self.probs.values())
            print(f"  Modelo listo: {total:,} n-gramas en memoria.           ")

    def score_word(self, word: str,
                   context: Optional[Tuple[str, ...]] = None) -> float:
        """
        Devuelve log10 P(word | context) con back-off.
        context es una tupla de palabras previas (puede ser vacía o None).
        """
        if context is None:
            context = ()

        # Intentar del orden más alto al más bajo
        for order in range(min(len(context) + 1, self.max_order), 0, -1):
            ctx   = context[-(order - 1):] if order > 1 else ()
            ngram = ctx + (word,)
            if ngram in self.probs.get(order, {}):
                return self.probs[order][ngram]

        # Unigrama
        uni = (word,)
        if uni in self.probs.get(1, {}):
            return self.probs[1][uni]

        # OOV — usar <unk> si existe
        unk = ("<unk>",)
        if unk in self.probs.get(1, {}):
            return self.probs[1][unk]

        return -6.0   # probabilidad muy baja para palabras totalmente desconocidas


# ═════════════════════════════════════════════════════════════════════
#  DECODIFICACIÓN CTC
# ═════════════════════════════════════════════════════════════════════

def decode_greedy(indices: List[int], idx2char: dict,
                  blank_idx: int = _BLANK_IDX) -> str:
    """
    Decodifica CTC greedy con un índice de blank configurable.

    El parámetro `blank_idx` permite reutilizar este decoder para modelos
    con distintas convenciones de vocabulario: el modelo OCR impreso usa
    blank=100 (último índice), mientras que el modelo HTR manuscrito usa
    blank=0 (primer índice).
    """
    result, prev = [], None
    for idx in indices:
        if idx != blank_idx and idx != prev:
            result.append(idx2char.get(idx, ""))
        prev = idx
    return "".join(result)


def _log_add(a: float, b: float) -> float:
    if a == float("-inf"): return b
    if b == float("-inf"): return a
    if a > b: return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def decode_beam(log_probs_seq: List[List[float]], idx2char: dict,
                beam_width: int = 10, blank_bonus: float = 2.0,
                length_norm: float = 0.65,
                lm: Optional[ArpaLM] = None,
                lm_alpha: float = 0.4,
                blank_idx: int = _BLANK_IDX) -> str:
    """
    Beam search CTC con soporte opcional de modelo de lenguaje ARPA.
    Si lm es None funciona como beam search puro (sin dependencias).

    El parámetro `blank_idx` permite reutilizar este decoder para modelos
    con distintas convenciones de vocabulario (impreso: blank=100,
    manuscrito CRNN-Lite v2: blank=0).
    """
    NEG   = float("-inf")
    LN10  = math.log(10)   # para convertir log10 → ln natural

    beams = {(): (0.0, NEG)}

    for lp_t in log_probs_seq:
        new_beams: dict = {}
        for seq, (p_b, p_nb) in beams.items():
            p_tot = _log_add(p_b, p_nb)
            for c, lp in enumerate(lp_t):
                if c == blank_idx:
                    pb, pnb = new_beams.get(seq, (NEG, NEG))
                    new_beams[seq] = (_log_add(pb, p_tot + lp + blank_bonus), pnb)
                else:
                    ext = seq + (c,)
                    if seq and seq[-1] == c:
                        pb, pnb = new_beams.get(seq, (NEG, NEG))
                        new_beams[seq] = (pb, _log_add(pnb, p_nb + lp))
                        pb, pnb = new_beams.get(ext, (NEG, NEG))
                        new_beams[ext] = (pb, _log_add(pnb, p_b + lp))
                    else:
                        pb, pnb = new_beams.get(ext, (NEG, NEG))
                        new_beams[ext] = (pb, _log_add(pnb, p_tot + lp))
        beams = dict(sorted(new_beams.items(),
                            key=lambda x: _log_add(x[1][0], x[1][1]),
                            reverse=True)[:beam_width])

    def lm_score(seq: tuple) -> float:
        if lm is None or not seq:
            return 0.0
        text  = "".join(idx2char.get(c, "") for c in seq)
        words = text.strip().split()
        if not words:
            return 0.0
        total = 0.0
        for i, word in enumerate(words):
            ctx = tuple(words[max(0, i - lm.max_order + 1): i])
            total += lm.score_word(word, ctx) * LN10   # log10 → ln
        return lm_alpha * total

    def score(seq):
        raw  = _log_add(beams[seq][0], beams[seq][1])
        norm = max(len(seq), 1) ** length_norm
        return raw / norm + lm_score(seq)

    best = max(beams.keys(), key=score)
    return "".join(idx2char.get(c, "") for c in best)


# ═════════════════════════════════════════════════════════════════════
#  PRE-PROCESADO DE IMAGEN
# ═════════════════════════════════════════════════════════════════════

def autocrop(img: Image.Image, threshold=200, padding=2) -> Image.Image:
    arr = np.array(img)
    ink = arr < threshold
    rows, cols = np.any(ink, axis=1), np.any(ink, axis=0)
    if not rows.any():
        return img
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    H, W = arr.shape
    return img.crop((max(0, c0-padding), max(0, r0-padding),
                     min(W, c1+padding+1), min(H, r1+padding+1)))


def preprocess(img_path: Union[str, Path], img_height: int) -> torch.Tensor:
    img = Image.open(img_path).convert("L")
    img = autocrop(img)
    w, h = img.size
    new_w = max(4, int(round(w * img_height / h)))
    img = img.resize((new_w, img_height), Image.BICUBIC)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return tfm(img).unsqueeze(0)


# ═════════════════════════════════════════════════════════════════════
#  CARGA DE VOCABULARIO
# ═════════════════════════════════════════════════════════════════════

def _load_vocab_from_checkpoint(ckpt: dict) -> Optional[dict]:
    return ckpt.get("vocab") or ckpt.get("idx2char") or None


def _load_vocab_from_file(vocab_path: str) -> Tuple[dict, dict]:
    chars = []
    for line in Path(vocab_path).read_text("utf-8").splitlines():
        c = line if line != "" else " "
        if c not in chars:
            chars.append(c)
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char  = {i: c for i, c in enumerate(chars)}
    return char2idx, idx2char


# ═════════════════════════════════════════════════════════════════════
#  CLASE PRINCIPAL: OCRPredictor
# ═════════════════════════════════════════════════════════════════════

class OCRPredictor:
    """
    Predictor OCR listo para usar.

    Parámetros
    ----------
    checkpoint_path : str | Path
        Ruta al archivo .pt guardado durante el entrenamiento.
    vocab_path : str | Path | None
        Ruta al vocab.txt. Si es None, busca junto al checkpoint o usa
        el vocab embebido en el checkpoint.
    device : str | None
        'cuda', 'cpu' o None (autodetectar).
    beam_width : int
        Ancho del beam search (0/1 = greedy, ≥2 = beam).
    beam_bonus : float
        Bonus para el token blank en beam search.
    length_norm : float
        Exponente de normalización de longitud.
    lm_path : str | None
        Ruta al archivo kenLM.arpa en español (cualquier carpeta).
        No requiere dependencias extra — se lee con Python puro.
        Descarga: https://huggingface.co/kensho/5gram-spanish-kenLM/resolve/main/kenLM.arpa
    lm_alpha : float
        Peso del modelo de lenguaje (recomendado 0.3–0.5).
    lm_max_order : int
        Orden máximo de n-gramas a cargar (2 = bigramas, 3 = trigramas).
    verbose : bool
        Si True, imprime info al cargar.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        vocab_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        beam_width: int = 10,
        beam_bonus: float = 2.0,
        length_norm: float = 0.65,
        lm_path: Optional[str] = None,
        lm_alpha: float = 0.4,
        lm_max_order: int = 2,
        verbose: bool = True,
    ):
        self.beam_width  = beam_width
        self.beam_bonus  = beam_bonus
        self.length_norm = length_norm
        self.lm_alpha    = lm_alpha

        # ── Dispositivo ────────────────────────────────────────────
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── Checkpoint ─────────────────────────────────────────────
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg  = ckpt.get("config", {})

        self.img_height  = cfg.get("img_height",  _DEFAULT_IMG_HEIGHT)
        vocab_size       = cfg.get("vocab_size",  _DEFAULT_VOCAB_SIZE)
        hidden_size      = cfg.get("hidden_size", 256)
        num_layers       = cfg.get("num_layers",  2)
        self.best_epoch  = ckpt.get("epoch", "?")
        self.best_cer    = ckpt.get("best_cer", None)

        # ── Vocabulario ────────────────────────────────────────────
        embedded = _load_vocab_from_checkpoint(ckpt)
        if embedded:
            self.idx2char = {int(k): v for k, v in embedded.items()}
        else:
            if vocab_path is None:
                candidates = [
                    Path(checkpoint_path).parent / "vocab.txt",
                    Path(checkpoint_path).parent.parent / "vocab" / "vocab.txt",
                    Path("vocab/vocab.txt"),
                    Path("vocab.txt"),
                ]
                for c in candidates:
                    if c.exists():
                        vocab_path = c
                        break

            if vocab_path is None or not Path(vocab_path).exists():
                raise FileNotFoundError(
                    "No se encontró vocab.txt. Pásalo con vocab_path=... "
                    "o ponlo en el mismo directorio que el checkpoint."
                )
            _, self.idx2char = _load_vocab_from_file(vocab_path)
            if verbose:
                print(f"Vocabulario cargado desde: {vocab_path}  ({len(self.idx2char)} símbolos)")

        # ── Modelo ─────────────────────────────────────────────────
        self.model = CRNN(
            vocab_size=vocab_size,
            img_height=self.img_height,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        if verbose:
            n_params = sum(p.numel() for p in self.model.parameters())
            cer_str  = f"  |  CER entrenamiento: {self.best_cer:.4f}" if self.best_cer else ""
            print(f"Modelo cargado: época {self.best_epoch}  "
                  f"|  {n_params:,} parámetros  "
                  f"|  dispositivo: {self.device}{cer_str}")

        # ── Modelo de lenguaje ARPA (puro Python, sin dependencias) ─
        self.lm = None
        if lm_path:
            lm_path = Path(lm_path)
            if not lm_path.exists():
                print(f"[AVISO] LM no encontrado en: {lm_path}\n"
                      f"        Beam search funcionará SIN modelo de lenguaje.")
            else:
                self.lm = ArpaLM(lm_path, max_order=lm_max_order, verbose=verbose)

    # ── Predicción de una sola imagen ─────────────────────────────

    def predict(self, img_path: Union[str, Path]) -> str:
        tensor = preprocess(img_path, self.img_height).to(self.device)

        with torch.no_grad():
            log_probs = self.model(tensor)   # [T, 1, vocab_size]

        valid_t = tensor.shape[3] // _CNN_STRIDE

        if self.beam_width <= 1:
            indices = log_probs[:valid_t, 0].argmax(dim=1).cpu().tolist()
            return decode_greedy(indices, self.idx2char)

        lp_np = log_probs[:valid_t, 0].cpu().float().numpy()
        seq   = [lp_np[t].tolist() for t in range(len(lp_np))]
        return decode_beam(
            seq, self.idx2char,
            beam_width  = self.beam_width,
            blank_bonus = self.beam_bonus,
            length_norm = self.length_norm,
            lm          = self.lm,
            lm_alpha    = self.lm_alpha,
        )

    # ── Predicción de múltiples imágenes ──────────────────────────

    def predict_batch(self, img_paths: List[Union[str, Path]]) -> List[str]:
        return [self.predict(p) for p in img_paths]

    def predict_folder(self, folder: Union[str, Path],
                       extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
                       show_progress: bool = True) -> dict:
        folder = Path(folder)
        paths  = sorted(p for p in folder.iterdir()
                        if p.suffix.lower() in extensions)
        if not paths:
            print(f"No se encontraron imágenes en '{folder}'")
            return {}

        results = {}
        for i, p in enumerate(paths, 1):
            results[p.name] = self.predict(p)
            if show_progress:
                print(f"[{i}/{len(paths)}] {p.name:<40} → {results[p.name]}")
        return results

    # ── Evaluación con ground-truth ───────────────────────────────

    def evaluate(self, img_paths: List[Union[str, Path]],
                 references: List[str]) -> dict:
        assert len(img_paths) == len(references)
        predictions = [self.predict(p) for p in img_paths]

        def lev(a, b):
            m, n = len(a), len(b)
            prev = list(range(n + 1))
            for i in range(1, m + 1):
                curr = [i] + [0] * n
                for j in range(1, n + 1):
                    curr[j] = prev[j-1] if a[i-1]==b[j-1] else 1+min(prev[j],curr[j-1],prev[j-1])
                prev = curr
            return prev[n]

        cer_sum = wer_sum = exact = 0
        for hyp, ref in zip(predictions, references):
            cer_sum += lev(list(hyp), list(ref)) / max(len(ref), 1)
            wer_sum += lev(hyp.split(), ref.split()) / max(len(ref.split()), 1)
            exact   += int(hyp == ref)

        n = len(references)
        return {
            "CER":         cer_sum / n,
            "WER":         wer_sum / n,
            "LineAcc":     exact / n,
            "n_samples":   n,
            "predictions": predictions,
        }


# ═════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════

def _build_parser():
    p = argparse.ArgumentParser(
        description="OCR CRNN+CTC — inferencia con decoder español opcional",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input",
                   help="Imagen (.png/.jpg) o carpeta con imágenes")
    p.add_argument("--checkpoint", "-c", required=True,
                   help="Ruta al checkpoint .pt")
    p.add_argument("--vocab",      "-v", default=None,
                   help="Ruta al vocab.txt (se busca automáticamente si no se da)")
    p.add_argument("--beam",       "-b", type=int,   default=10,
                   help="Ancho del beam search (0/1 = greedy, ≥2 = beam)")
    p.add_argument("--bonus",             type=float, default=2.0,
                   help="Bonus para blank en beam search")
    p.add_argument("--alpha",             type=float, default=0.65,
                   help="Exponente de normalización de longitud")
    p.add_argument("--lm",                default=None,
                   help=(
                       "Ruta al kenLM.arpa (cualquier carpeta, sin dependencias extra).\n"
                       "Descarga: https://huggingface.co/kensho/5gram-spanish-kenLM/resolve/main/kenLM.arpa"
                   ))
    p.add_argument("--lm-alpha",          type=float, default=0.4,
                   help="Peso del modelo de lenguaje (recomendado 0.3–0.5)")
    p.add_argument("--lm-order",          type=int,   default=2,
                   help="Orden máximo de n-gramas a cargar (2=bigramas, 3=trigramas)")
    p.add_argument("--device",            default=None,
                   help="'cuda' o 'cpu' (autodetectar si no se especifica)")
    p.add_argument("--gt",                default=None,
                   help="Carpeta con .txt ground-truth para evaluar (opcional)")
    return p


def main():
    args = _build_parser().parse_args()

    predictor = OCRPredictor(
        checkpoint_path = args.checkpoint,
        vocab_path      = args.vocab,
        device          = args.device,
        beam_width      = args.beam,
        beam_bonus      = args.bonus,
        length_norm     = args.alpha,
        lm_path         = args.lm,
        lm_alpha        = args.lm_alpha,
        lm_max_order    = args.lm_order,
    )

    input_path = Path(args.input)

    if input_path.is_file():
        text = predictor.predict(input_path)
        print(f"\n{'─'*60}")
        print(f"  Imagen : {input_path.name}")
        print(f"  Texto  : {text}")
        print(f"{'─'*60}\n")
        return

    if input_path.is_dir():
        print(f"\n{'─'*60}")
        results = predictor.predict_folder(input_path)

        if args.gt:
            gt_dir = Path(args.gt)
            img_paths, refs = [], []
            for img_name, pred in results.items():
                txt_path = gt_dir / (Path(img_name).stem + ".txt")
                if txt_path.exists():
                    img_paths.append(input_path / img_name)
                    refs.append(txt_path.read_text("utf-8").strip())

            if refs:
                metrics = predictor.evaluate(img_paths, refs)
                print(f"\n{'═'*60}")
                print(f"  EVALUACIÓN  ({metrics['n_samples']} muestras con GT)")
                print(f"  CER     : {metrics['CER']*100:.2f}%")
                print(f"  WER     : {metrics['WER']*100:.2f}%")
                print(f"  LineAcc : {metrics['LineAcc']*100:.1f}%")
                print(f"{'═'*60}\n")
        return

    print(f"[ERROR] '{args.input}' no es un archivo ni una carpeta válida.")
    sys.exit(1)


if __name__ == "__main__":
    main()