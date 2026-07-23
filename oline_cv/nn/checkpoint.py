"""Checkpoint load/save helpers (kept separate to avoid -m runpy warnings)."""

from __future__ import annotations

from pathlib import Path

import torch

from oline_cv.nn.model import OLTechniqueNet, build_model

DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent.parent / "models" / "ol_technique_net.pt"


def load_model(path: str | Path = DEFAULT_WEIGHTS, device: str | None = None) -> OLTechniqueNet:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(path, map_location=device_t, weights_only=False)
    meta = payload.get("meta", {})
    model = build_model(
        hidden=int(meta.get("hidden", 128)),
        num_blocks=int(meta.get("num_blocks", 4)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device_t)
    model.eval()
    return model
