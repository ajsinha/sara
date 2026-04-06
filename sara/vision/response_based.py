# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.4.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
from __future__ import annotations
"""
sara.vision.response_based
========================
Response-based (soft-logit) knowledge distillation for image classification.

The :class:`ResponseBasedDistiller` wraps the standard training loop with
checkpoint saving, cosine-annealing LR, gradient clipping, and per-epoch
validation. Use :func:`build_cifar10_loaders` to get ready-made CIFAR-10
data loaders.
"""


import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sara.core.losses import DistillationLoss
from sara.core.utils import ProfileResult, profile_model, compare_profiles


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class VisionDistillConfig:
    """Hyperparameters for :class:`ResponseBasedDistiller`."""
    alpha:         float = 0.6
    temperature:   float = 4.0
    epochs:        int   = 30
    lr:            float = 1e-3
    weight_decay:  float = 1e-4
    grad_clip:     float = 5.0
    checkpoint_dir: str  = "./checkpoints"


# ── Data loaders ───────────────────────────────────────────────────────────────

def build_cifar10_loaders(
    data_dir:   str = "./data",
    batch_size: int = 128,
    num_workers:int = 4,
    download:   bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Build CIFAR-10 train / validation data loaders.

    Parameters
    ----------
    data_dir    : Directory to download or find CIFAR-10 data
    batch_size  : Samples per batch
    num_workers : DataLoader worker processes
    download    : Whether to auto-download if not present

    Returns
    -------
    (train_loader, val_loader)
    """
    from torchvision import datasets, transforms  # type: ignore

    MEAN = (0.4914, 0.4822, 0.4465)
    STD  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_ds = datasets.CIFAR10(data_dir, train=True,  download=download, transform=train_tf)
    val_ds   = datasets.CIFAR10(data_dir, train=False, download=download, transform=val_tf)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds,   batch_size=batch_size * 2, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
    )


def build_default_teacher_student(num_classes: int = 10) -> tuple[nn.Module, nn.Module]:
    """
    Build a ResNet-50 teacher and MobileNetV2 student pre-adapted for the
    given number of output classes.

    Parameters
    ----------
    num_classes : Number of output classes

    Returns
    -------
    (teacher, student) — teacher has ImageNet weights; student is random-init
    """
    from torchvision import models  # type: ignore

    teacher = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    teacher.fc = nn.Linear(2048, num_classes)

    student = models.mobilenet_v2(weights=None)
    student.classifier[1] = nn.Linear(1280, num_classes)

    return teacher, student


# ── Trainer ────────────────────────────────────────────────────────────────────

class ResponseBasedDistiller:
    """
    Trains a student model by minimising the combined distillation loss
    (soft teacher logits + hard labels) on a given dataset.

    Parameters
    ----------
    teacher  : Frozen teacher model
    student  : Trainable student model
    loss_fn  : DistillationLoss instance
    config   : VisionDistillConfig
    device   : ``'cuda'``, ``'mps'``, or ``'cpu'``

    Examples
    --------
    >>> distiller = ResponseBasedDistiller(teacher, student)
    >>> distiller.train(train_loader, val_loader)
    >>> print(f"Best accuracy: {distiller.best_val_acc:.4f}")
    """

    def __init__(
        self,
        teacher:  nn.Module,
        student:  nn.Module,
        loss_fn:  Optional[DistillationLoss] = None,
        config:   Optional[VisionDistillConfig] = None,
        device:   Optional[str] = None,
    ) -> None:
        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher = teacher.to(self.device).eval()
        self.student = student.to(self.device)
        self.config  = config or VisionDistillConfig()
        self.loss_fn = loss_fn or DistillationLoss(
            alpha=self.config.alpha, temperature=self.config.temperature
        )
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.best_val_acc:   float            = 0.0
        self._best_ckpt:     Optional[dict]   = None
        self._train_history: list[dict]       = []

    @classmethod
    def from_config(
        cls,
        teacher: nn.Module,
        student: nn.Module,
        config:  VisionDistillConfig,
    ) -> "ResponseBasedDistiller":
        """Factory constructor from a VisionDistillConfig."""
        return cls(
            teacher=teacher,
            student=student,
            loss_fn=DistillationLoss(alpha=config.alpha, temperature=config.temperature),
            config=config,
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        verbose:      bool = True,
    ) -> list[dict]:
        """
        Run the full distillation training loop.

        Parameters
        ----------
        train_loader : Training data loader
        val_loader   : Validation data loader
        verbose      : Print per-epoch metrics

        Returns
        -------
        List of per-epoch metric dicts
        """
        cfg       = self.config
        optimizer = torch.optim.Adam(
            self.student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        ckpt_dir  = Path(cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, cfg.epochs + 1):
            metrics = self._train_epoch(epoch, train_loader, optimizer, cfg.grad_clip)
            metrics["val_acc"] = self._validate(val_loader)
            metrics["lr"]      = scheduler.get_last_lr()[0]
            scheduler.step()
            self._train_history.append(metrics)

            if metrics["val_acc"] > self.best_val_acc:
                self.best_val_acc = metrics["val_acc"]
                self._best_ckpt   = copy.deepcopy(self.student.state_dict())
                torch.save(self._best_ckpt, ckpt_dir / "student_best.pth")

            if verbose:
                print(
                    f"Epoch {epoch:03d}/{cfg.epochs}  "
                    f"loss={metrics['loss']:.4f}  "
                    f"val_acc={metrics['val_acc']:.4f}  "
                    f"best={self.best_val_acc:.4f}"
                )

        if self._best_ckpt is not None:
            self.student.load_state_dict(self._best_ckpt)
        return self._train_history

    def _train_epoch(
        self,
        epoch: int,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
    ) -> dict:
        self.student.train()
        epoch_loss = 0.0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                t_logits = self.teacher(images)

            s_logits = self.student(images)
            loss     = self.loss_fn(s_logits, t_logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.student.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()

        return {"epoch": epoch, "loss": epoch_loss / len(loader)}

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

    def profile(self, dummy_input: Optional[torch.Tensor] = None) -> dict:
        """
        Profile teacher and student latency/size and print comparison.

        Parameters
        ----------
        dummy_input : Optional input tensor for profiling.
                      Defaults to a random (1, 3, 32, 32) CIFAR-10-sized tensor.

        Returns
        -------
        Dict with keys 'teacher', 'student', 'speedup', 'compression'
        """
        if dummy_input is None:
            dummy_input = torch.randn(1, 3, 32, 32, device=self.device)

        t_res = profile_model(self.teacher, dummy_input, model_name="Teacher")
        s_res = profile_model(self.student, dummy_input, model_name="Student")
        ratios = compare_profiles(t_res, s_res)
        return {"teacher": t_res, "student": s_res, **ratios}

    @property
    def history(self) -> list[dict]:
        """Training history as a list of per-epoch metric dicts."""
        return self._train_history
