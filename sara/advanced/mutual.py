# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.8.3
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ajsinha/sara
from __future__ import annotations
"""
sara.advanced.mutual
==================
Online / Mutual Knowledge Distillation (Zhang et al., 2018).

Two student networks of equal capacity train simultaneously, each using
the other's current soft predictions as an additional supervisory signal.
Neither model is pre-trained — both improve together.
"""


import copy
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class MutualDistiller:
    """
    Train two student models via mutual peer distillation.

    Parameters
    ----------
    student1_factory : Callable that returns a fresh model for peer 1
    student2_factory : Callable that returns a fresh model for peer 2
                       (can be the same factory for symmetric learning)
    alpha            : Weight for the KL peer-supervision term in [0, 1]
    device           : Torch device string

    Examples
    --------
    >>> from torchvision import models
    >>> factory = lambda: models.resnet18(weights=None)
    >>> trainer = MutualDistiller(factory, factory, alpha=0.5)
    >>> acc1, acc2 = trainer.train(train_loader, val_loader, epochs=30)
    """

    def __init__(
        self,
        student1_factory: Callable[[], nn.Module],
        student2_factory: Callable[[], nn.Module],
        alpha:  float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.s1     = student1_factory().to(self.device)
        self.s2     = student2_factory().to(self.device)
        self.alpha  = alpha
        self.kl     = nn.KLDivLoss(reduction="batchmean")
        self.ce     = nn.CrossEntropyLoss()

    def _mutual_loss(
        self,
        logits_self: torch.Tensor,
        logits_peer: torch.Tensor,
        labels:      torch.Tensor,
    ) -> torch.Tensor:
        """
        Combined CE (hard labels) + KL (peer soft labels) for one student.

        Parameters
        ----------
        logits_self : (B, C) this student's raw outputs
        logits_peer : (B, C) peer student's raw outputs — will be detached
        labels      : (B,)  integer class indices
        """
        kl_loss = self.kl(
            F.log_softmax(logits_self, dim=-1),
            F.softmax(logits_peer.detach(), dim=-1),
        )
        ce_loss = self.ce(logits_self, labels)
        return (1.0 - self.alpha) * ce_loss + self.alpha * kl_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int   = 30,
        lr:           float = 0.1,
        momentum:     float = 0.9,
        weight_decay: float = 1e-4,
        verbose:      bool  = True,
    ) -> tuple[float, float]:
        """
        Co-train both students.

        Parameters
        ----------
        train_loader : Training data loader (images, labels)
        val_loader   : Validation data loader
        epochs       : Total training epochs
        lr           : Initial SGD learning rate
        momentum     : SGD momentum
        weight_decay : L2 regularisation
        verbose      : Print per-epoch metrics

        Returns
        -------
        (final_val_acc_s1, final_val_acc_s2)
        """
        opt1   = torch.optim.SGD(self.s1.parameters(), lr=lr,
                                  momentum=momentum, weight_decay=weight_decay)
        opt2   = torch.optim.SGD(self.s2.parameters(), lr=lr,
                                  momentum=momentum, weight_decay=weight_decay)
        sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=epochs)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=epochs)

        for epoch in range(1, epochs + 1):
            self.s1.train(); self.s2.train()
            total_l1 = total_l2 = 0.0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                l1 = self.s1(images)
                l2 = self.s2(images)

                loss1 = self._mutual_loss(l1, l2, labels)
                loss2 = self._mutual_loss(l2, l1, labels)

                opt1.zero_grad(set_to_none=True)
                loss1.backward(retain_graph=True)
                opt1.step()

                opt2.zero_grad(set_to_none=True)
                loss2.backward()
                opt2.step()

                total_l1 += loss1.item()
                total_l2 += loss2.item()

            sched1.step(); sched2.step()
            acc1 = self._validate(self.s1, val_loader)
            acc2 = self._validate(self.s2, val_loader)

            if verbose:
                print(
                    f"Epoch {epoch:03d}/{epochs}  "
                    f"L1={total_l1/len(train_loader):.4f}  acc1={acc1:.4f}  "
                    f"L2={total_l2/len(train_loader):.4f}  acc2={acc2:.4f}"
                )

        return (
            self._validate(self.s1, val_loader),
            self._validate(self.s2, val_loader),
        )

    @torch.no_grad()
    def _validate(self, model: nn.Module, loader: DataLoader) -> float:
        model.eval()
        correct = total = 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            out     = model(images)
            logits  = out[0] if isinstance(out, tuple) else out
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
        return correct / max(total, 1)

    @property
    def student1(self) -> nn.Module:
        return self.s1

    @property
    def student2(self) -> nn.Module:
        return self.s2
