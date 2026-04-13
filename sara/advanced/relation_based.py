# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.8.3
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ajsinha/sara
from __future__ import annotations
"""
sara.advanced.relation_based
==========================
Relational Knowledge Distillation (Park et al., CVPR 2019).

Distils pairwise structural relationships between samples in the
teacher's feature space to the student — architecture-agnostic and
effective across large capacity gaps.
"""


import copy
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sara.core.losses import RKDLoss, DistillationLoss
from sara.vision.feature_based import attach_feature_hooks


class RelationalKDDistiller:
    """
    Trains a student using Relational KD (pairwise distance + angle losses)
    over the teacher's penultimate embedding space.

    The teacher and student are hooked at a specified layer to extract
    embedding vectors. The student learns to reproduce the teacher's
    inter-sample relationships rather than individual activations.

    Parameters
    ----------
    teacher        : Frozen teacher model
    student        : Trainable student model
    teacher_embed_layer : Dot-name of teacher's embedding layer
    student_embed_layer : Dot-name of student's embedding layer
    lambda_d       : Weight for RKD distance loss
    lambda_a       : Weight for RKD angle loss
    alpha          : Weight for standard KD loss
    temperature    : KD temperature
    device         : Torch device string

    Examples
    --------
    >>> distiller = RelationalKDDistiller(
    ...     teacher, student,
    ...     teacher_embed_layer="avgpool",
    ...     student_embed_layer="avgpool",
    ... )
    >>> distiller.train(train_loader, val_loader)
    """

    def __init__(
        self,
        teacher:              nn.Module,
        student:              nn.Module,
        teacher_embed_layer:  str   = "avgpool",
        student_embed_layer:  str   = "avgpool",
        lambda_d:             float = 1.0,
        lambda_a:             float = 2.0,
        alpha:                float = 0.5,
        temperature:          float = 4.0,
        device:               Optional[str] = None,
    ) -> None:
        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher = teacher.to(self.device).eval()
        self.student = student.to(self.device)

        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self._t_store = attach_feature_hooks(self.teacher, [teacher_embed_layer])
        self._s_store = attach_feature_hooks(self.student, [student_embed_layer])
        self._t_layer = teacher_embed_layer
        self._s_layer = student_embed_layer

        self._rkd_fn = RKDLoss(lambda_d=lambda_d, lambda_a=lambda_a)
        self._kd_fn  = DistillationLoss(alpha=alpha, temperature=temperature)

        self.best_val_acc = 0.0
        self._best_ckpt: Optional[dict] = None

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int   = 30,
        lr:           float = 1e-3,
        lambda_rkd:   float = 1.0,
        verbose:      bool  = True,
    ) -> None:
        """
        Train with joint RKD + standard KD loss.

        Parameters
        ----------
        train_loader : Training data loader
        val_loader   : Validation data loader
        epochs       : Total training epochs
        lr           : Initial Adam learning rate
        lambda_rkd   : Scale factor for the total RKD loss term
        verbose      : Print per-epoch metrics
        """
        optimizer = torch.optim.Adam(
            self.student.parameters(), lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(1, epochs + 1):
            self.student.train()
            epoch_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.no_grad():
                    t_logits = self.teacher(images)
                    t_embed  = self._t_store[self._t_layer].flatten(1).clone()

                s_logits = self.student(images)
                s_embed  = self._s_store[self._s_layer].flatten(1)

                loss_kd  = self._kd_fn(s_logits, t_logits, labels)
                rkd_dict = self._rkd_fn(s_embed, t_embed)
                loss     = loss_kd + lambda_rkd * rkd_dict["total"]

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
