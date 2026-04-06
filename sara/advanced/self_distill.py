# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.6.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
from __future__ import annotations
"""
sara.advanced.self_distill
========================
Self-distillation via early-exit branches.

The deepest classifier exit acts as the teacher, providing soft targets
to all shallower exits.  No separate pre-trained teacher required.

The architecture follows a "ResNet trunk + side exits" pattern, but
:class:`SelfDistillTrainer` accepts any model that implements the
multi-exit interface (returns a list of logit tensors, shallow → deep).
"""


import copy
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sara.core.losses import SelfDistillationLoss


# ── Multi-exit ResNet ──────────────────────────────────────────────────────────

class MultiExitResNet(nn.Module):
    """
    ResNet-34 backbone with two early-exit classifiers and one final classifier.

    Exits are attached after layer2, layer3, and layer4 respectively.

    Parameters
    ----------
    num_classes : Number of output classes

    Returns (forward)
    -----------------
    List of three logit tensors: [exit1_logits, exit2_logits, final_logits]
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        from torchvision import models  # type: ignore

        base = models.resnet34(weights=None)
        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.exit1 = self._make_exit(128, num_classes)   # after layer2
        self.exit2 = self._make_exit(256, num_classes)   # after layer3
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(512, num_classes)          # final exit

    @staticmethod
    def _make_exit(in_channels: int, num_classes: int) -> nn.Module:
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x);   l1 = self.exit1(x)
        x = self.layer3(x);   l2 = self.exit2(x)
        x = self.layer4(x);   lf = self.fc(self.pool(x).flatten(1))
        return [l1, l2, lf]


# ── Trainer ────────────────────────────────────────────────────────────────────

class SelfDistillTrainer:
    """
    Train a multi-exit network with self-distillation.

    The trainer expects the model's ``forward()`` to return a list of
    logit tensors ordered from shallowest to deepest exit.

    Parameters
    ----------
    model         : Multi-exit model (e.g. :class:`MultiExitResNet`)
    temperature   : Softening temperature for deepest-exit soft targets
    exit_weights  : Per-exit loss weights (len must match number of exits)
    device        : Torch device string

    Examples
    --------
    >>> model   = MultiExitResNet(num_classes=10)
    >>> trainer = SelfDistillTrainer(model, temperature=3.0)
    >>> trainer.train(train_loader, val_loader, epochs=40)
    >>> print(f"Best accuracy (final exit): {trainer.best_val_acc:.4f}")
    """

    def __init__(
        self,
        model:        nn.Module,
        temperature:  float = 3.0,
        exit_weights: tuple[float, ...] = (0.3, 0.4, 1.0),
        device:       Optional[str] = None,
    ) -> None:
        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model   = model.to(self.device)
        self.loss_fn = SelfDistillationLoss(
            temperature=temperature, exit_weights=exit_weights
        )
        self.best_val_acc = 0.0
        self._best_ckpt: Optional[dict] = None

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        epochs:       int   = 40,
        lr:           float = 0.1,
        weight_decay: float = 5e-4,
        verbose:      bool  = True,
    ) -> None:
        """
        Train the multi-exit model with self-distillation.

        Parameters
        ----------
        train_loader : Training data loader
        val_loader   : Validation data loader
        epochs       : Total training epochs
        lr           : Initial SGD learning rate
        weight_decay : L2 regularisation
        verbose      : Print per-epoch metrics
        """
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                exit_logits    = self.model(images)          # list of tensors
                loss, _        = self.loss_fn(exit_logits, labels)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            val_acc = self._validate(val_loader)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._best_ckpt   = copy.deepcopy(self.model.state_dict())

            if verbose:
                print(
                    f"Epoch {epoch:03d}/{epochs}  "
                    f"loss={epoch_loss/len(train_loader):.4f}  "
                    f"val_acc={val_acc:.4f}  best={self.best_val_acc:.4f}"
                )

        if self._best_ckpt is not None:
            self.model.load_state_dict(self._best_ckpt)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        """Evaluate using only the final (deepest) exit."""
        self.model.eval()
        correct = total = 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            exits   = self.model(images)
            preds   = exits[-1].argmax(dim=1)   # deepest exit
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
        return correct / max(total, 1)

    @torch.no_grad()
    def evaluate_all_exits(self, loader: DataLoader) -> dict[str, float]:
        """
        Return validation accuracy for every exit branch.

        Returns
        -------
        Dict mapping ``"exit_1"``, ``"exit_2"``, …, ``"final"`` to accuracy values
        """
        self.model.eval()
        n_exits      = None
        correct_list = None
        total        = 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            exits = self.model(images)

            if n_exits is None:
                n_exits      = len(exits)
                correct_list = [0] * n_exits

            for i, logits in enumerate(exits):
                correct_list[i] += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        accs = {}
        for i in range(n_exits or 0):
            key         = "final" if i == n_exits - 1 else f"exit_{i+1}"
            accs[key]   = correct_list[i] / max(total, 1)
        return accs
