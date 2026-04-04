# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.1.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
"""
tests/test_vision.py
====================
Tests for vision distillation modules.
All tests use synthetic data; no CIFAR-10 download required.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sara.core.losses import DistillationLoss
from sara.vision.response_based import ResponseBasedDistiller, VisionDistillConfig
from sara.vision.feature_based import FeatureBasedDistiller, attach_feature_hooks


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_loader(n: int = 32, c: int = 3, h: int = 8, w: int = 8,
                num_classes: int = 5, batch_size: int = 8) -> DataLoader:
    x = torch.randn(n, c, h, w)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def tiny_cnn(in_ch: int, out: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_ch, 8, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, out),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ResponseBasedDistiller
# ══════════════════════════════════════════════════════════════════════════════

class TestResponseBasedDistiller:

    def test_train_returns_history(self, tmp_path):
        teacher = tiny_cnn(3, 5)
        student = tiny_cnn(3, 5)
        cfg     = VisionDistillConfig(epochs=1, checkpoint_dir=str(tmp_path))
        d       = ResponseBasedDistiller(teacher, student, config=cfg)
        history = d.train(make_loader(), make_loader(), verbose=False)
        assert len(history) == 1
        assert "loss" in history[0]
        assert "val_acc" in history[0]

    def test_best_val_acc_tracked(self, tmp_path):
        teacher = tiny_cnn(3, 5)
        student = tiny_cnn(3, 5)
        cfg     = VisionDistillConfig(epochs=2, checkpoint_dir=str(tmp_path))
        d       = ResponseBasedDistiller(teacher, student, config=cfg)
        d.train(make_loader(), make_loader(), verbose=False)
        assert 0.0 <= d.best_val_acc <= 1.0

    def test_teacher_grad_is_disabled(self, tmp_path):
        teacher = tiny_cnn(3, 5)
        student = tiny_cnn(3, 5)
        cfg     = VisionDistillConfig(epochs=1, checkpoint_dir=str(tmp_path))
        d       = ResponseBasedDistiller(teacher, student, config=cfg)
        for p in d.teacher.parameters():
            assert not p.requires_grad

    def test_from_config_factory(self, tmp_path):
        teacher = tiny_cnn(3, 5)
        student = tiny_cnn(3, 5)
        cfg     = VisionDistillConfig(epochs=1, checkpoint_dir=str(tmp_path))
        d       = ResponseBasedDistiller.from_config(teacher, student, cfg)
        assert d.config is cfg

    def test_profile_returns_ratios(self, tmp_path):
        teacher = tiny_cnn(3, 5)
        student = tiny_cnn(3, 5)
        cfg     = VisionDistillConfig(epochs=0, checkpoint_dir=str(tmp_path))
        d       = ResponseBasedDistiller(teacher, student, config=cfg)
        dummy   = torch.randn(1, 3, 8, 8)
        result  = d.profile(dummy)
        assert "speedup" in result
        assert "compression" in result

    def test_history_property_returns_list(self, tmp_path):
        teacher = tiny_cnn(3, 5)
        student = tiny_cnn(3, 5)
        cfg     = VisionDistillConfig(epochs=2, checkpoint_dir=str(tmp_path))
        d       = ResponseBasedDistiller(teacher, student, config=cfg)
        d.train(make_loader(), make_loader(), verbose=False)
        assert isinstance(d.history, list)
        assert len(d.history) == 2


# ══════════════════════════════════════════════════════════════════════════════
# attach_feature_hooks
# ══════════════════════════════════════════════════════════════════════════════

class TestAttachFeatureHooks:

    def test_hook_captures_output(self):
        model = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)
        )
        store = attach_feature_hooks(model, ["1"])   # index "1" = ReLU layer
        _     = model(torch.randn(2, 4))
        assert "1" in store
        assert store["1"].shape == (2, 8)

    def test_multiple_hooks(self):
        model = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 5), nn.ReLU()
        )
        store = attach_feature_hooks(model, ["0", "2"])
        _     = model(torch.randn(3, 4))
        assert "0" in store and "2" in store
        assert store["0"].shape == (3, 8)
        assert store["2"].shape == (3, 5)


# ══════════════════════════════════════════════════════════════════════════════
# AttentionTransferDistiller
# ══════════════════════════════════════════════════════════════════════════════

class TestAttentionTransferDistiller:

    def _build_distiller(self, tmp_path):
        from sara.vision.attention_transfer import AttentionTransferDistiller
        teacher = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 5),
        )
        student = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 5),
        )
        # Hook layer "1" (ReLU after first conv) on both
        return AttentionTransferDistiller(
            teacher=teacher, student=student,
            teacher_layers=["1"], student_layers=["1"],
            lambda_at=100.0, device="cpu",
        )

    def test_instantiation_succeeds(self, tmp_path):
        d = self._build_distiller(tmp_path)
        assert d.best_val_acc == 0.0

    def test_mismatched_layer_lists_raise(self):
        from sara.vision.attention_transfer import AttentionTransferDistiller
        teacher = nn.Linear(4, 5)
        student = nn.Linear(4, 5)
        with pytest.raises(ValueError, match="equal length"):
            AttentionTransferDistiller(
                teacher, student,
                teacher_layers=["0", "1"],
                student_layers=["0"],
            )

    def test_train_one_epoch(self, tmp_path):
        d      = self._build_distiller(tmp_path)
        loader = make_loader(n=16, c=3, h=8, num_classes=5, batch_size=8)
        d.train(loader, loader, epochs=1, verbose=False)
        assert 0.0 <= d.best_val_acc <= 1.0

    def test_teacher_frozen_after_init(self, tmp_path):
        d = self._build_distiller(tmp_path)
        for p in d.teacher.parameters():
            assert not p.requires_grad
