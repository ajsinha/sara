# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.2.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
from __future__ import annotations
"""
sara.advanced.progressive
========================
Multi-stage (progressive) knowledge distillation pipeline.

Each stage reduces model capacity by a controlled factor and uses the
previous stage's student as the next stage's teacher.
"""


import copy
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sara.core.losses import DistillationLoss


@dataclass
class Stage:
    """Configuration for one distillation stage."""
    teacher_id:  str    # HuggingFace model ID or "previous_stage"
    student_id:  str
    temperature: float = 4.0
    alpha:       float = 0.7
    epochs:      int   = 5
    lr:          float = 2e-5


class ProgressiveDistiller:
    """
    Chain multiple teacher→student distillation stages.

    Each stage's best-checkpoint student becomes the teacher for the next stage.
    Works with any models loaded via a user-supplied factory function.

    Parameters
    ----------
    stages         : List of Stage configs, in order (large → small)
    model_factory  : Callable(model_id: str) → nn.Module
    device         : Torch device string

    Examples
    --------
    >>> from transformers import AutoModelForSequenceClassification
    >>> factory = lambda mid: AutoModelForSequenceClassification.from_pretrained(mid, num_labels=2)
    >>> stages = [
    ...     Stage("bert-large-uncased", "bert-base-uncased",  T=4, alpha=0.7, epochs=3),
    ...     Stage("bert-base-uncased",  "distilbert-base-uncased", T=3, alpha=0.6, epochs=3),
    ... ]
    >>> distiller = ProgressiveDistiller(stages, factory)
    >>> final_student = distiller.run(train_loader, val_loader)
    """

    def __init__(
        self,
        stages:        list[Stage],
        model_factory: Callable[[str], nn.Module],
        device:        str | None = None,
    ) -> None:
        self.stages  = stages
        self.factory = model_factory
        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def run(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        verbose:      bool = True,
    ) -> nn.Module:
        """
        Execute all stages sequentially.

        Returns
        -------
        Final student model (best checkpoint from the last stage)
        """
        teacher = self.factory(self.stages[0].teacher_id).to(self.device).eval()
        student: nn.Module | None = None

        for i, stage in enumerate(self.stages, start=1):
            if verbose:
                print(f"\n[Stage {i}/{len(self.stages)}] "
                      f"{stage.teacher_id} → {stage.student_id}")

            student   = self.factory(stage.student_id).to(self.device)
            loss_fn   = DistillationLoss(alpha=stage.alpha, temperature=stage.temperature)
            optimizer = torch.optim.AdamW(
                student.parameters(), lr=stage.lr, weight_decay=0.01
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage.epochs)

            for p in teacher.parameters():
                p.requires_grad_(False)

            best_acc, best_ckpt = 0.0, None

            for epoch in range(1, stage.epochs + 1):
                student.train()
                epoch_loss = 0.0
                for batch in train_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attn_mask = batch["attention_mask"].to(self.device)
                    labels    = batch["labels"].to(self.device)

                    with torch.no_grad():
                        t_out = teacher(input_ids=input_ids, attention_mask=attn_mask)

                    s_out = student(input_ids=input_ids, attention_mask=attn_mask)
                    loss  = loss_fn(s_out.logits, t_out.logits, labels)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()

                scheduler.step()
                acc = self._validate(student, val_loader)
                if acc > best_acc:
                    best_acc  = acc
                    best_ckpt = copy.deepcopy(student.state_dict())
                if verbose:
                    print(f"  Epoch {epoch:02d}/{stage.epochs}  "
                          f"loss={epoch_loss/len(train_loader):.4f}  acc={acc:.4f}")

            student.load_state_dict(best_ckpt)
            teacher = copy.deepcopy(student).eval()  # promote for next stage

        return student

    @torch.no_grad()
    def _validate(self, model: nn.Module, loader: DataLoader) -> float:
        model.eval()
        correct = total = 0
        for batch in loader:
            ids   = batch["input_ids"].to(self.device)
            mask  = batch["attention_mask"].to(self.device)
            lbls  = batch["labels"].to(self.device)
            out   = model(input_ids=ids, attention_mask=mask)
            correct += (out.logits.argmax(-1) == lbls).sum().item()
            total   += lbls.size(0)
        return correct / max(total, 1)
