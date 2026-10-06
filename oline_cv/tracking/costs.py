"""Association costs for Deep HM-SORT.

All distances live in ``[0, 1]``. A value of 1 means "cannot match".
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def harmonic_mean_distance(a, b) -> np.ndarray:
    """Harmonic mean ``2/(1/a + 1/b)`` of two distance arrays.

    Written as ``2ab/(a+b)`` so zero distances give 0 instead of a division
    by zero, and NaN inputs are treated as "cannot match" (1). The harmonic
    mean is pulled toward the smaller term, so a strong cue (tight overlap or
    near-identical appearance) can carry a weaker one, but a distance of 1 on
    one side still leaves the result high when the other side is mediocre.
    """
    a = np.nan_to_num(np.asarray(a, dtype=np.float64), nan=1.0, posinf=1.0, neginf=0.0)
    b = np.nan_to_num(np.asarray(b, dtype=np.float64), nan=1.0, posinf=1.0, neginf=0.0)
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)
    s = a + b
    return np.where(s > _EPS, 2.0 * a * b / np.maximum(s, _EPS), 0.0)


def normalize_rows(x) -> np.ndarray:
    """L2-normalise rows; zero or non-finite rows become NaN (missing)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.size == 0:
        return x.reshape(len(x), -1)
    norm = np.linalg.norm(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), axis=1, keepdims=True)
    bad = (norm[:, 0] <= _EPS) | ~np.all(np.isfinite(x), axis=1)
    out = x / np.maximum(norm, _EPS)
    out[bad] = np.nan
    return out


def cosine_distance_matrix(a, b) -> np.ndarray:
    """``1 - cos`` between embedding rows, clipped to ``[0, 1]``.

    Missing (NaN / zero) embeddings on either side give distance 1. Opposite
    vectors are clipped to 1 rather than 2, so the appearance term stays on
    the same scale as the IoU distance.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = 0 if a.size == 0 else (1 if a.ndim == 1 else a.shape[0])
    nb = 0 if b.size == 0 else (1 if b.ndim == 1 else b.shape[0])
    if na == 0 or nb == 0:
        return np.ones((na, nb), dtype=np.float64)
    a = normalize_rows(a)
    b = normalize_rows(b)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"embedding size mismatch: {a.shape[1]} vs {b.shape[1]}")
    sim = np.nan_to_num(a, nan=0.0) @ np.nan_to_num(b, nan=0.0).T
    d = np.clip(1.0 - sim, 0.0, 1.0)
    missing = np.isnan(a[:, 0])[:, None] | np.isnan(b[:, 0])[None, :]
    d[missing] = 1.0
    return d


def gated_hm_cost(
    d_iou,
    d_app,
    *,
    appearance_threshold: float,
    proximity_threshold: float | None,
) -> np.ndarray:
    """Deep HM-SORT cost with the Deep-EIoU appearance gates.

    Appearance distances above ``appearance_threshold`` are set to 1, and
    appearance is ignored (set to 1) for pairs whose spatial distance exceeds
    ``proximity_threshold`` so a look-alike far away cannot be matched on
    appearance alone.
    """
    d_iou = np.clip(np.nan_to_num(np.asarray(d_iou, dtype=np.float64), nan=1.0), 0.0, 1.0)
    d_app = np.clip(np.nan_to_num(np.asarray(d_app, dtype=np.float64), nan=1.0), 0.0, 1.0)
    d_app = np.where(d_app > appearance_threshold, 1.0, d_app)
    if proximity_threshold is not None:
        d_app = np.where(d_iou > proximity_threshold, 1.0, d_app)
    return harmonic_mean_distance(d_iou, d_app)


def linear_assignment(cost, threshold: float):
    """Hungarian assignment keeping only pairs with ``cost <= threshold``.

    Returns ``(matches, unmatched_rows, unmatched_cols)`` where ``matches`` is
    a list of ``(row, col)``. Deterministic for a given matrix.
    """
    from scipy.optimize import linear_sum_assignment

    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        cost = cost.reshape(0, 0)
    n_r, n_c = cost.shape
    if n_r == 0 or n_c == 0:
        return [], list(range(n_r)), list(range(n_c))
    big = float(threshold) + 1e4
    c = np.where(np.isfinite(cost) & (cost <= threshold), cost, big)
    rows, cols = linear_sum_assignment(c)
    matches = [(int(r), int(k)) for r, k in zip(rows, cols) if c[r, k] <= threshold]
    mr = {r for r, _ in matches}
    mc = {k for _, k in matches}
    return (
        matches,
        [r for r in range(n_r) if r not in mr],
        [k for k in range(n_c) if k not in mc],
    )
