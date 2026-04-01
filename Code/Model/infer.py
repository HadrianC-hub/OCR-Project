"""
infer.py  —  Inferencia sobre una imagen o carpeta de imágenes
"""

from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image

from model import CRNN
from dataset import decode_ctc, CHAR2IDX, IDX2CHAR, BLANK_IDX, IMG_HEIGHT


def load_model(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg  = ckpt.get("config", {})
    model = CRNN(
        vocab_size=cfg.get("vocab_size", 101),
        img_height=cfg.get("img_height", IMG_HEIGHT),
        hidden_size=cfg.get("hidden_size", 128),
    )
    # Eliminar prefijo "_orig_mod." si el modelo fue compilado con torch.compile
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state"].items()}
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def transcribe_image(model, img_path: str, img_height: int = IMG_HEIGHT) -> str:
    img = Image.open(img_path).convert("L")
    w, h = img.size
    new_w = max(4, int(round(w * img_height / h)))
    img = img.resize((new_w, img_height), Image.BICUBIC)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    x = tfm(img).unsqueeze(0)   # [1, 1, H, W]
    with torch.no_grad():
        log_probs = model(x)    # [T, 1, vocab]
        _, best = log_probs.max(2)
    return decode_ctc(best.squeeze(1).tolist())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Ruta al checkpoint .pt")
    parser.add_argument("input",      help="Imagen .png o carpeta de imágenes")
    args = parser.parse_args()

    model, cfg = load_model(args.checkpoint)
    h = cfg.get("img_height", IMG_HEIGHT)

    input_path = Path(args.input)
    paths = [input_path] if input_path.is_file() else sorted(
        list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
    )
    for p in paths:
        text = transcribe_image(model, str(p), img_height=h)
        print(f"{p.name}\t{text}")