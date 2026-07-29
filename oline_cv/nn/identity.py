"""Appearance identity network — pixel embedding to freeze OL lock.

Trained at lock time on the locked athlete's torso crops (positives) vs other
people in-frame (negatives). Used as a hard gate: once locked, a candidate
must match the embedding + classic filters or we keep the old identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityEmbedNet(nn.Module):
    """Small multi-filter CNN → L2-normalized embedding."""

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        z = self.head(h)
        return F.normalize(z, dim=-1)


def _crop_torso(frame: np.ndarray, bbox: np.ndarray, size: int = 96) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = [int(float(v)) for v in bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    bw, bh = x1 - x0, y1 - y0
    if bw < 12 or bh < 20:
        return None
    ry0 = y0 + int(0.08 * bh)
    ry1 = y0 + int(0.58 * bh)
    rx0 = x0 + int(0.10 * bw)
    rx1 = x1 - int(0.10 * bw)
    if rx1 - rx0 < 8 or ry1 - ry0 < 8:
        return None
    crop = frame[ry0:ry1, rx0:rx1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def _to_tensor(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Channel-wise normalize lightly
    rgb = (rgb - 0.45) / 0.25
    t = torch.from_numpy(rgb).permute(2, 0, 1)
    return t


def _augment(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = bgr.copy()
    # Brightness / contrast
    alpha = float(rng.uniform(0.75, 1.25))
    beta = float(rng.uniform(-25, 25))
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    # Small shift crop
    h, w = out.shape[:2]
    dx = int(rng.integers(-4, 5))
    dy = int(rng.integers(-4, 5))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    # Color jitter via HSV
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + float(rng.uniform(-8, 8))) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(rng.uniform(0.8, 1.2)), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * float(rng.uniform(0.85, 1.15)), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


def color_hist_signature(bgr: np.ndarray) -> np.ndarray:
    """Multi-filter color signature for cross-check (not the NN)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    labs = []
    for ch, bins, rng in ((0, 18, [0, 180]), (1, 16, [0, 256]), (2, 16, [0, 256])):
        hist = cv2.calcHist([hsv], [ch], None, [bins], rng)
        hist = cv2.normalize(hist, hist).flatten()
        labs.append(hist)
    # Edge energy fingerprint
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    eh = cv2.calcHist([edges], [0], None, [8], [0, 256])
    eh = cv2.normalize(eh, eh).flatten()
    labs.append(eh)
    sig = np.concatenate(labs).astype(np.float32)
    n = float(np.linalg.norm(sig) + 1e-6)
    return sig / n


def hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


@dataclass
class IdentityVerifier:
    net: IdentityEmbedNet
    template_embed: torch.Tensor
    template_hist: np.ndarray
    device: torch.device
    min_embed: float = 0.58
    min_hist: float = 0.52

    @torch.inference_mode()
    def score_crop(self, bgr: np.ndarray) -> dict[str, float]:
        x = _to_tensor(bgr).unsqueeze(0).to(self.device)
        z = self.net(x)[0]
        embed = float(F.cosine_similarity(z, self.template_embed, dim=0).item())
        hist = hist_similarity(self.template_hist, color_hist_signature(bgr))
        # Combined: both filters must be healthy
        ok = embed >= self.min_embed and hist >= self.min_hist
        return {"embed": embed, "hist": hist, "ok": 1.0 if ok else 0.0}

    def score_bbox(self, frame: np.ndarray, bbox: np.ndarray) -> dict[str, float] | None:
        crop = _crop_torso(frame, bbox)
        if crop is None:
            return None
        return self.score_crop(crop)


def train_identity_verifier(
    frame: np.ndarray,
    lock_bbox: np.ndarray,
    other_boxes: list[np.ndarray] | None = None,
    steps: int = 80,
    embed_dim: int = 64,
    min_embed: float = 0.58,
    device: str | None = None,
) -> IdentityVerifier | None:
    """Quick lock-time fine-tune: same athlete vs other people + pixel filters."""
    lock_crop = _crop_torso(frame, lock_bbox)
    if lock_crop is None:
        return None

    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = IdentityEmbedNet(embed_dim=embed_dim).to(device_t)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)

    neg_crops: list[np.ndarray] = []
    for b in other_boxes or []:
        c = _crop_torso(frame, b)
        if c is not None:
            neg_crops.append(c)
    # Synthetic hard negatives from scrambled lock crop
    rng = np.random.default_rng(0)
    scramble = lock_crop.copy()
    scramble = scramble[:, ::-1]  # mirror alone is still similar — also shift hue hard
    hsv = cv2.cvtColor(scramble, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int32) + 90) % 180
    scramble = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    neg_crops.append(scramble)

    net.train()
    for step in range(steps):
        pos = [_to_tensor(_augment(lock_crop, rng)) for _ in range(4)]
        # Second positive view
        pos2 = [_to_tensor(_augment(lock_crop, rng)) for _ in range(4)]
        neg_idx = int(rng.integers(0, len(neg_crops)))
        negs = [_to_tensor(_augment(neg_crops[neg_idx], rng)) for _ in range(4)]

        xp = torch.stack(pos).to(device_t)
        xp2 = torch.stack(pos2).to(device_t)
        xn = torch.stack(negs).to(device_t)

        zp, zp2, zn = net(xp), net(xp2), net(xn)
        # InfoNCE-ish: pull pos together, push neg
        pos_sim = F.cosine_similarity(zp, zp2)
        neg_sim = F.cosine_similarity(zp, zn)
        loss = (1.0 - pos_sim).mean() + F.relu(neg_sim - 0.15).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        # Average a few clean lock views as template
        views = torch.stack(
            [_to_tensor(lock_crop)]
            + [_to_tensor(_augment(lock_crop, rng)) for _ in range(5)]
        ).to(device_t)
        tmpl = net(views).mean(dim=0)
        tmpl = F.normalize(tmpl, dim=0)

    return IdentityVerifier(
        net=net,
        template_embed=tmpl.detach(),
        template_hist=color_hist_signature(lock_crop),
        device=device_t,
        min_embed=min_embed,
        min_hist=0.52,
    )
