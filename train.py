"""Trains the drinking-waste classifier (AluCan / Glass / HDPEM / PET) using transfer learning.

I'm using the same images and the same 962-image validation split as the original
Colab notebook. Keras' validation_split just takes the first 20% of each class's
files in alphabetical order, so I do the same thing here. That way the validation
accuracy can be compared directly with the old 47.8%.

The saved model is whatever we end up with after the last epoch. I never use the
validation set to pick a checkpoint or to stop early, so the number isn't inflated.

Usage:
    python train.py --data /path/to/Images_of_Waste/rawimgs
"""
import argparse
import json
import math
import os
import time

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# Same image types that Keras' flow_from_directory reads. The 8 HEIC files in the dataset get skipped, same as before.
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".tif", ".tiff")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def split_files(data_dir, val_split=0.2):
    """Copies how Keras does validation_split: sort each class's files and put the first `val_split` fraction in validation."""
    class_names = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )
    train, val = [], []
    for idx, name in enumerate(class_names):
        files = sorted(
            f for f in os.listdir(os.path.join(data_dir, name))
            if f.lower().endswith(IMG_EXTS)
        )
        n_val = int(val_split * len(files))
        paths = [os.path.join(data_dir, name, f) for f in files]
        val += [(p, idx) for p in paths[:n_val]]
        train += [(p, idx) for p in paths[n_val:]]
    return class_names, train, val


def load_rgb(path):
    # Some of the photos are saved sideways with an EXIF rotation tag, and exif_transpose straightens them out.
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


class Rot90:
    """Rotates by a random multiple of 90°. These are photos of litter taken from above,
    so there's no real "right way up" and any orientation is fine."""

    def __call__(self, img):
        k = np.random.randint(4)
        return img if k == 0 else img.rotate(90 * k, expand=True)


def train_transform(size):
    return T.Compose([
        Rot90(),
        # The cans/bottles are often pretty small in the shot, so don't crop too hard or we might cut them out.
        T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.75, 1.333)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(0.3, 0.3, 0.2, 0.03),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])


def eval_transform(size):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


class Files(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(load_rgb(path)), label


def seed_worker(_):
    # Give each worker its own numpy seed, otherwise they'd all pick the same Rot90 rotations.
    np.random.seed(torch.initial_seed() % 2**32)


@torch.no_grad()
def predict(model, loader, device, tta=False):
    """Returns class probabilities. With tta=True it averages over all 8 flips/rotations of each image."""
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device)
        if tta:
            views = [torch.rot90(v, k, (2, 3)) for v in (x, x.flip(3)) for k in range(4)]
            p = torch.stack([model(v).softmax(1) for v in views]).mean(0)
        else:
            p = model(x).softmax(1)
        out.append(p.float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="trade ~40%% speed for much less memory (e.g. 8GB Macs)")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    class_names, train_items, val_items = split_files(args.data)
    n_cls = len(class_names)
    print(f"device={device} classes={class_names} train={len(train_items)} val={len(val_items)}")

    train_dl = DataLoader(
        Files(train_items, train_transform(args.img_size)), batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=args.workers,
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
    )
    val_dl = DataLoader(
        Files(val_items, eval_transform(args.img_size)), batch_size=args.batch_size * 2,
        num_workers=args.workers, persistent_workers=args.workers > 0,
    )

    # Start from a backbone pretrained on ImageNet-22k and put a new 4-class head on top.
    model = timm.create_model(args.model, pretrained=True, num_classes=n_cls,
                              drop_path_rate=0.1).to(device)
    if args.grad_checkpointing:
        model.set_grad_checkpointing(True)

    # The classes are a bit unbalanced (PET has about 1.5x as many images as HDPEM), so weight the loss to even it out.
    counts = np.bincount([l for _, l in train_items], minlength=n_cls)
    weights = torch.tensor(len(train_items) / (n_cls * counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)

    # The head is brand new, so give it a 10x higher learning rate than the pretrained backbone.
    head = model.get_classifier()
    head_ids = {id(p) for p in head.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.lr},
         {"params": head.parameters(), "lr": args.lr * 10}],
        weight_decay=args.weight_decay,
    )
    total = args.epochs * len(train_dl)
    warmup = len(train_dl)  # warm up linearly for one epoch, then cosine decay

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    y_val = np.array([l for _, l in val_items])
    history = []

    for epoch in range(args.epochs):
        model.train()
        t0, run_loss, correct, seen = time.time(), 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        # Just to keep an eye on how it's doing. This never gets used to choose a checkpoint.
        val_acc = float((predict(model, val_dl, device).argmax(1) == y_val).mean())
        history.append({"epoch": epoch + 1, "loss": run_loss / seen,
                        "acc": correct / seen, "val_acc": val_acc})
        print(f"epoch {epoch + 1}/{args.epochs} loss={run_loss / seen:.4f} "
              f"acc={correct / seen:.4f} val_acc={val_acc:.4f} ({time.time() - t0:.0f}s)",
              flush=True)

    torch.save({"model": args.model, "img_size": args.img_size, "classes": class_names,
                "state_dict": model.state_dict()}, os.path.join(args.out, "model.pt"))
    with open(os.path.join(args.out, "labels.txt"), "w") as f:
        f.write("\n".join(class_names))

    from sklearn.metrics import classification_report, confusion_matrix
    probs = predict(model, val_dl, device, tta=True)
    pred = probs.argmax(1)
    results = {
        "model": args.model, "img_size": args.img_size, "epochs": args.epochs,
        "val_acc": history[-1]["val_acc"],
        "val_acc_tta": float((pred == y_val).mean()),
        "confusion_matrix": confusion_matrix(y_val, pred).tolist(),
        "history": history,
    }
    print(classification_report(y_val, pred, target_names=class_names, digits=4))
    print(confusion_matrix(y_val, pred))
    print(f"val acc: {results['val_acc']:.4f}   with 8x TTA: {results['val_acc_tta']:.4f}")
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
