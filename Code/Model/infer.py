"""
infer.py  —  Inferencia sobre una imagen o carpeta de imágenes

Usa automáticamente GPU si está disponible.
"""

import argparse
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from model   import CRNN
from dataset import decode_ctc, IMG_HEIGHT


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str, device: torch.device = None):
    if device is None:
        device = get_device()
    ckpt  = torch.load(checkpoint_path, map_location=device)
    cfg   = ckpt.get("config", {})
    model = CRNN(
        vocab_size=cfg.get("vocab_size",  101),
        img_height=cfg.get("img_height",  IMG_HEIGHT),
        hidden_size=cfg.get("hidden_size", 128),
    )
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Modelo cargado desde '{checkpoint_path}'  |  dispositivo: {device}")
    return model, cfg, device


def transcribe_image(model, img_path: str, device: torch.device, img_height: int = IMG_HEIGHT) -> str:
    img = Image.open(img_path).convert("L")
    w, h = img.size
    new_w = max(4, int(round(w * img_height / h)))
    img = img.resize((new_w, img_height), Image.BICUBIC)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    x = tfm(img).unsqueeze(0).to(device)   # [1, 1, H, W]
    with torch.no_grad():
        log_probs = model(x)
        _, best   = log_probs.max(2)
    return decode_ctc(best.squeeze(1).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Ruta al checkpoint .pt")
    parser.add_argument("input",      help="Imagen .png/.jpg o carpeta de imágenes")
    args = parser.parse_args()

    model, cfg, device = load_model(args.checkpoint)
    h = cfg.get("img_height", IMG_HEIGHT)

    input_path = Path(args.input)
    paths = [input_path] if input_path.is_file() else sorted(
        list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
    )

    print(f"\n{'Archivo':<40} Transcripción")
    print("─" * 80)
    for p in paths:
        text = transcribe_image(model, str(p), device, img_height=h)
        print(f"{p.name:<40} {text}")