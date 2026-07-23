"""Train / fine-tune OLTechniqueNet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from oline_cv.nn.dataset import build_training_arrays, train_val_split
from oline_cv.nn.features import POSTURE_CLASSES
from oline_cv.nn.model import OLTechniqueNet, build_model, count_parameters

from oline_cv.nn.checkpoint import DEFAULT_WEIGHTS


def evaluate(model: OLTechniqueNet, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    correct = 0
    total = 0
    per_class = {c: {"tp": 0, "n": 0} for c in POSTURE_CLASSES}
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss_sum += float(crit(logits, yb).item()) * len(yb)
            pred = logits.argmax(dim=-1)
            correct += int((pred == yb).sum().item())
            total += len(yb)
            for p, t in zip(pred.tolist(), yb.tolist()):
                name = POSTURE_CLASSES[t]
                per_class[name]["n"] += 1
                if p == t:
                    per_class[name]["tp"] += 1
    acc = correct / max(total, 1)
    return {
        "accuracy": acc,
        "loss": loss_sum / max(total, 1),
        "n": total,
        "per_class_recall": {
            c: (v["tp"] / v["n"] if v["n"] else None) for c, v in per_class.items()
        },
    }


def train(
    video_path: str | None = "footage.mp4",
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden: int = 128,
    num_blocks: int = 4,
    synth_n: int = 4000,
    out_path: str | Path = DEFAULT_WEIGHTS,
    device: str | None = None,
    pose_model: str = "yolov8n-pose.pt",
) -> dict:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Building dataset (synth={synth_n}, video={video_path})…")
    X, y = build_training_arrays(
        video_path=video_path if video_path and Path(video_path).exists() else None,
        synth_n=synth_n,
        pose_model=pose_model,
    )
    train_ds, val_ds = train_val_split(X, y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model(hidden=hidden, num_blocks=num_blocks).to(device_t)
    print(f"Model params: {count_parameters(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc = -1.0
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device_t), yb.to(device_t)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * len(yb)
            n += len(yb)
        sched.step()
        val = evaluate(model, val_loader, device_t)
        row = {
            "epoch": epoch,
            "train_loss": running / max(n, 1),
            "val_accuracy": val["accuracy"],
            "val_loss": val["loss"],
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}  train_loss={row['train_loss']:.4f}  "
            f"val_acc={val['accuracy']:.3f}  val_loss={val['loss']:.4f}"
        )
        if val["accuracy"] > best_acc:
            best_acc = val["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": best_state,
        "meta": {
            "feature_dim": int(X.shape[-1]),
            "window": int(X.shape[1]),
            "hidden": hidden,
            "num_blocks": num_blocks,
            "classes": list(POSTURE_CLASSES),
            "val_accuracy": best_acc,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
        },
    }
    torch.save(payload, out_path)
    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps({"best_val_accuracy": best_acc, "history": history, "meta": payload["meta"]}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {out_path}  best_val_acc={best_acc:.3f}")
    return {"best_val_accuracy": best_acc, "path": str(out_path), "history": history}


def load_model(path: str | Path = DEFAULT_WEIGHTS, device: str | None = None) -> OLTechniqueNet:
    from oline_cv.nn.checkpoint import load_model as _load

    return _load(path, device)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--video", default="footage.mp4")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--synth", type=int, default=4000)
    p.add_argument("--model", default="yolov8n-pose.pt")
    args = p.parse_args()
    train(video_path=args.video, epochs=args.epochs, synth_n=args.synth, pose_model=args.model)
