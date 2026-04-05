# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.2.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
from __future__ import annotations
"""
sara.vision.feature_based
=======================
FitNets-style feature-based knowledge distillation (Romero et al., 2015).

Two-stage pipeline:
  Stage 1 — hint-layer pre-training: align the student mid-network
             features to the teacher via a learnable adapter.
  Stage 2 — joint distillation: combined feature + response loss.

Use :class:`FeatureBasedDistiller` directly or call
:func:`attach_feature_hooks` to add hooks to any model pair.
"""


import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sara.core.losses import DistillationLoss, FeatureDistillationLoss


def attach_feature_hooks(
    model: nn.Module,
    layer_names: list[str],
) -> dict[str, torch.Tensor]:
    """
    Register forward hooks on the named layers of ``model`` and return
    a dict that will be populated with the layer outputs after each
    forward pass.

    Parameters
    ----------
    model       : Any ``nn.Module``
    layer_names : Dot-separated layer names, e.g. ``["layer2", "features.7"]``

    Returns
    -------
    Dict mapping layer_name → most-recent output tensor.
    The dict is updated in-place on each forward pass.

    Examples
    --------
    >>> store = attach_feature_hooks(resnet, ["layer2", "layer3"])
    >>> _ = resnet(x)
    >>> feat = store["layer2"]   # (B, 512, H, W)
    """
    store: dict[str, torch.Tensor] = {}

    def _make_hook(name: str):
        def hook(module, input, output):
            store[name] = output
        return hook

    for name in layer_names:
        parts  = name.split(".")
        module = model
        for part in parts:
            module = getattr(module, part)
        module.register_forward_hook(_make_hook(name))

    return store


class FeatureBasedDistiller:
    """
    Two-stage FitNets distillation.

    Parameters
    ----------
    teacher         : Frozen teacher model
    student         : Trainable student model
    teacher_layer   : Dot-name of teacher layer to tap (e.g. ``"layer2"``)
    student_layer   : Dot-name of student layer to tap (e.g. ``"features.7"``)
    teacher_channels: Output channels of the teacher layer
    student_channels: Output channels of the student layer
    device          : Torch device string

    Examples
    --------
    >>> distiller = FeatureBasedDistiller(
    ...     teacher, student,
    ...     teacher_layer="layer2", student_layer="features.7",
    ...     teacher_channels=512, student_channels=32,
    ... )
    >>> distiller.train(train_loader, val_loader)
    """

    def __init__(
        self,
        teacher:          nn.Module,
        student:          nn.Module,
        teacher_layer:    str   = "layer2",
        student_layer:    str   = "features.7",
        teacher_channels: int   = 512,
        student_channels: int   = 32,
        device:           Optional[str] = None,
    ) -> None:
        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher = teacher.to(self.device).eval()
        self.student = student.to(self.device)

        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self._t_feats = attach_feature_hooks(self.teacher, [teacher_layer])
        self._s_feats = attach_feature_hooks(self.student, [student_layer])
        self._t_layer = teacher_layer
        self._s_layer = student_layer

        # Adapter: project student → teacher channel dim
        self.adapter = nn.Conv2d(
            student_channels, teacher_channels, kernel_size=1, bias=False
        ).to(self.device)

        self._feat_loss_fn = FeatureDistillationLoss(
            student_channels=teacher_channels,  # adapter output = teacher channels
            teacher_channels=teacher_channels,
        ).to(self.device)
        # Override adapter — use the class-level one defined above
        self._feat_loss_fn.adapter = nn.Identity()

        self._kd_loss_fn = DistillationLoss(alpha=0.5, temperature=4.0)
        self.best_val_acc = 0.0
        self._best_ckpt: Optional[dict] = None

    def _get_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._s_feats[self._s_layer], self._t_feats[self._t_layer]

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        self.student.eval()
        correct = total = 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            preds   = self.student(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
        return correct / max(total, 1)

    def train(
        self,
        train_loader:   DataLoader,
        val_loader:     DataLoader,
        epochs_hint:    int   = 10,
        epochs_joint:   int   = 20,
        lr:             float = 1e-3,
        lambda_feat:    float = 0.1,
        verbose:        bool  = True,
    ) -> None:
        """
        Run two-stage FitNets training.

        Stage 1 trains only the adapter and student's hint layers.
        Stage 2 trains everything jointly with combined feature + KD loss.

        Parameters
        ----------
        train_loader : Training data loader
        val_loader   : Validation data loader
        epochs_hint  : Epochs for Stage 1 (hint-layer pre-training)
        epochs_joint : Epochs for Stage 2 (joint distillation)
        lr           : Initial learning rate
        lambda_feat  : Weight of feature loss in Stage 2
        verbose      : Print per-epoch metrics
        """
        self._stage1_hint(train_loader, epochs=epochs_hint, lr=lr, verbose=verbose)
        self._stage2_joint(
            train_loader, val_loader,
            epochs=epochs_joint, lr=lr * 0.5, lambda_feat=lambda_feat, verbose=verbose,
        )
        if self._best_ckpt is not None:
            self.student.load_state_dict(self._best_ckpt)

    def _stage1_hint(
        self,
        loader:  DataLoader,
        epochs:  int,
        lr:      float,
        verbose: bool,
    ) -> None:
        if verbose:
            print(f"\n[Stage 1] Hint-layer pre-training ({epochs} epochs)")

        opt = torch.optim.Adam(
            list(self.student.parameters()) + list(self.adapter.parameters()), lr=lr
        )
        for epoch in range(1, epochs + 1):
            self.student.train()
            self.adapter.train()
            total = 0.0
            for images, _ in loader:
                images = images.to(self.device)
                with torch.no_grad():
                    self.teacher(images)
                    t_feat = self._t_feats[self._t_layer].clone()
                self.student(images)
                s_feat     = self._s_feats[self._s_layer]
                projected  = self.adapter(s_feat)
                loss       = F.mse_loss(projected, t_feat.detach())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item()
            if verbose:
                print(f"  Epoch {epoch:02d}/{epochs}  feat_loss={total/len(loader):.4f}")

    def _stage2_joint(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int,
        lr:           float,
        lambda_feat:  float,
        verbose:      bool,
    ) -> None:
        if verbose:
            print(f"\n[Stage 2] Joint KD + feature distillation ({epochs} epochs)")

        opt   = torch.optim.Adam(
            list(self.student.parameters()) + list(self.adapter.parameters()),
            lr=lr, weight_decay=1e-4,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        for epoch in range(1, epochs + 1):
            self.student.train(); self.adapter.train()
            epoch_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    t_logits = self.teacher(images)
                    t_feat   = self._t_feats[self._t_layer].clone()
                s_logits = self.student(images)
                s_feat   = self.adapter(self._s_feats[self._s_layer])
                loss_kd  = self._kd_loss_fn(s_logits, t_logits, labels)
                loss_f   = F.mse_loss(s_feat, t_feat.detach())
                loss     = loss_kd + lambda_feat * loss_f
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            sched.step()
            val_acc = self._validate(val_loader)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._best_ckpt   = copy.deepcopy(self.student.state_dict())
            if verbose:
                print(
                    f"  Epoch {epoch:02d}/{epochs}  "
                    f"loss={epoch_loss/len(train_loader):.4f}  "
                    f"val_acc={val_acc:.4f}  best={self.best_val_acc:.4f}"
                )
