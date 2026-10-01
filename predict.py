"""Classifies one or more photos using a model trained with train.py.

Usage:
    python predict.py artifacts/model.pt photo1.jpg [photo2.jpg ...]
"""
import sys

import timm
import torch

from train import eval_transform, load_rgb

FRIENDLY = {"AluCan": "Aluminium can", "Glass": "Glass bottle",
            "HDPEM": "HDPE (milk-jug plastic)", "PET": "PET plastic bottle"}


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = timm.create_model(ckpt["model"], pretrained=False, num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["classes"], ckpt["img_size"]


@torch.no_grad()
def classify(model, classes, img_size, path, device):
    x = eval_transform(img_size)(load_rgb(path)).unsqueeze(0).to(device)
    views = [torch.rot90(v, k, (2, 3)) for v in (x, x.flip(3)) for k in range(4)]
    probs = torch.stack([model(v).softmax(1) for v in views]).mean(0)[0].cpu()
    return dict(zip(classes, probs.tolist()))


def main():
    ckpt_path, paths = sys.argv[1], sys.argv[2:]
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model, classes, img_size = load_model(ckpt_path, device)
    for path in paths:
        probs = classify(model, classes, img_size, path, device)
        best = max(probs, key=probs.get)
        print(f"{path}: {FRIENDLY.get(best, best)} ({probs[best]:.1%})  "
              + "  ".join(f"{c}={p:.1%}" for c, p in probs.items()))


if __name__ == "__main__":
    main()
