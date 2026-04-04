# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.1.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
from __future__ import annotations
"""
sara.vision.attention_transfer
=============================
Attention Transfer distillation (Zagoruyko & Komodakis, ICLR 2017).

Registers forward hooks on paired teacher/student layers to automatically
capture spatial attention maps during the forward pass, then uses
:class:`AttentionTransferLoss` to penalise divergence.
"""


import copy
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sara.core.losses import AttentionTransferLoss, DistillationLoss
from sara.vision.feature_based import attach_feature_hooks


class AttentionTransferDistiller:
    """
    Combines attention-transfer loss with standard response-based KD.

    For each paired layer group, the spatial attention maps (L2-normalised
    sum of squared channel activations) of the student and teacher are
    aligned via MSE.

    Parameters
    ----------
    teacher          : Frozen teacher model
    student          : Trainable student model
    teacher_layers   : List of teacher layer names to tap (dot-separated)
    student_layers   : List of student layer names to tap (must match length)
    lambda_at        : Weight for attention-transfer loss
    device           : Torch device string

    Examples
    --------
    >>> distiller = AttentionTransferDistiller(
    ...     teacher, student,
    ...     teacher_layers=["layer2", "layer3"],
    ...     student_layers=["features.5", "features.10"],
    ...     lambda_at=1000,
    ... )
    >>> distiller.train(train_loader, val_loader)
    """

    def __init__(
        self,
        teacher:         nn.Module,
        student:         nn.Module,
        teacher_layers:  list[str],
        student_layers:  list[str],
        lambda_at:       float = 1000.0,
        alpha:           float = 0.5,
        temperature:     float = 4.0,
        device:          Optional[str] = None,
    ) -> None:
        if len(teacher_layers) != len(student_layers):
            raise ValueError(
                f"teacher_layers ({len(teacher_layers)}) and "
                f"student_layers ({len(student_layers)}) must have equal length"
            )

        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher = teacher.to(self.device).eval()
        self.student = student.to(self.device)

        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self._t_store = attach_feature_hooks(self.teacher, teacher_layers)
        self._s_store = attach_feature_hooks(self.student, student_layers)
        self._t_layers = teacher_layers
        self._s_layers = student_layers

        self.lambda_at   = lambda_at
        self._at_loss_fn = AttentionTransferLoss()
        self._kd_loss_fn = DistillationLoss(alpha=alpha, temperature=temperature)

        self.best_val_acc = 0.0
        self._best_ckpt: Optional[dict] = None

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int   = 30,
        lr:           float = 1e-3,
        weight_decay: float = 1e-4,
        verbose:      bool  = True,
    ) -> None:
        """
        Train the student with joint AT + KD loss.

        Parameters
        ----------
        train_loader : Training data loader
        val_loader   : Validation data loader
        epochs       : Total training epochs
        lr           : Initial learning rate
        weight_decay : L2 regularisation
        verbose      : Print per-epoch metrics
        """
        optimizer = torch.optim.Adam(
            self.student.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(1, epochs + 1):
            self.student.train()
            epoch_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.no_grad():
                    t_logits = self.teacher(images)
                    t_feats  = [self._t_store[l].clone() for l in self._t_layers]

                s_logits = self.student(images)
                s_feats  = [self._s_store[l] for l in self._s_layers]

                loss_kd = self._kd_loss_fn(s_logits, t_logits, labels)
                loss_at = self._at_loss_fn(s_feats, t_feats)
                loss    = loss_kd + self.lambda_at * loss_at

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.student.parameters(), 5.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            val_acc = self._validate(val_loader)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._best_ckpt   = copy.deepcopy(self.student.state_dict())

            if verbose:
                print(
                    f"Epoch {epoch:03d}/{epochs}  "
                    f"loss={epoch_loss/len(train_loader):.4f}  "
                    f"val_acc={val_acc:.4f}  best={self.best_val_acc:.4f}"
                )

        if self._best_ckpt is not None:
            self.student.load_state_dict(self._best_ckpt)

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
