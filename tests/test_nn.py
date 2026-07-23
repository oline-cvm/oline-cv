"""Tests for OLTechniqueNet and feature pipeline."""

from __future__ import annotations

import numpy as np
import torch

from oline_cv.nn.features import (
    FEATURE_DIM,
    POSTURE_CLASSES,
    synthesize_feature_batch,
    window_features,
)
from oline_cv.nn.model import ResidualTemporalBlock, build_model, count_parameters
from oline_cv.nn.train import evaluate
from oline_cv.pose_tracker import FramePose


def test_feature_dim_constant():
    assert FEATURE_DIM == 24
    assert len(POSTURE_CLASSES) == 4


def test_synthesize_shapes():
    X, y = synthesize_feature_batch(32, seed=1)
    assert X.shape == (32, 16, FEATURE_DIM)
    assert y.shape == (32,)
    assert y.min() >= 0 and y.max() <= 2


def test_model_forward_shape():
    model = build_model(hidden=64, num_blocks=4)
    x = torch.randn(8, 16, FEATURE_DIM)
    logits = model(x)
    assert logits.shape == (8, 4)
    assert count_parameters(model) > 50_000


def test_residual_block_preserves_length():
    block = ResidualTemporalBlock(32, dilation=4)
    x = torch.randn(2, 16, 32)
    y = block(x)
    assert y.shape == x.shape


def test_predict_label():
    model = build_model(hidden=32, num_blocks=2)
    model.eval()
    x = torch.randn(3, 16, FEATURE_DIM)
    labels = model.predict_label(x)
    assert len(labels) == 3
    assert all(l in POSTURE_CLASSES for l in labels)


def test_window_features_padding():
    poses = [
        FramePose(
            frame_idx=i,
            timestamp_ms=i * 33.0,
            keypoints_xy=np.random.randn(17, 2) * 10 + 100,
            keypoints_conf=np.ones(17) * 0.8,
            bbox_xyxy=np.array([50.0, 50.0, 150.0, 200.0]),
            person_confidence=0.9,
            low_confidence=False,
            usable=True,
        )
        for i in range(5)
    ]
    feats = window_features(poses, standing_height_px=120.0, start=0, length=16)
    assert feats.shape == (16, FEATURE_DIM)
    # padded tail should be zeros-ish for missing frames
    assert np.allclose(feats[10:], 0.0)


def test_train_smoke(tmp_path):
    from oline_cv.nn.train import train

    out = tmp_path / "net.pt"
    result = train(
        video_path=None,
        epochs=2,
        batch_size=32,
        synth_n=256,
        hidden=32,
        num_blocks=2,
        out_path=out,
    )
    assert out.exists()
    assert result["best_val_accuracy"] >= 0.0
    # reload
    from oline_cv.nn.checkpoint import load_model

    model = load_model(out)
    x = torch.randn(2, 16, FEATURE_DIM)
    assert model(x).shape == (2, 4)


def test_evaluate_dict_keys():
    model = build_model(hidden=32, num_blocks=2)
    X, y = synthesize_feature_batch(64, seed=0)
    from oline_cv.nn.dataset import PoseWindowDataset
    from torch.utils.data import DataLoader

    loader = DataLoader(PoseWindowDataset(X, y), batch_size=16)
    metrics = evaluate(model, loader, torch.device("cpu"))
    assert "accuracy" in metrics and "per_class_recall" in metrics
