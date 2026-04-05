# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.2.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
"""
examples/01_response_based_kd.py
=================================
Distil a ResNet-50 teacher into a MobileNetV2 student on CIFAR-10.

Run:
    python examples/01_response_based_kd.py

Requirements:
    pip install -e ".[vision]"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sara.core.progress import SaraLogger
from sara.vision.response_based import (
    ResponseBasedDistiller, VisionDistillConfig,
    build_cifar10_loaders, build_default_teacher_student,
)


def main():
    log = SaraLogger("Vision KD")
    log.banner(
        "Sara — Response-Based Vision Distillation",
        "Teacher : ResNet-50   →   Student : MobileNetV2",
        "Dataset : CIFAR-10",
    )

    log.section("Data loaders")
    log.step("Building CIFAR-10 train/val loaders")
    train_loader, val_loader = build_cifar10_loaders(batch_size=128)
    log.done(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    log.section("Models")
    log.step("Loading ResNet-50 (teacher) and MobileNetV2 (student)")
    teacher, student = build_default_teacher_student(num_classes=10)
    log.done("Models ready")

    cfg = VisionDistillConfig(alpha=0.6, temperature=4.0, epochs=30, lr=1e-3,
                              checkpoint_dir="./checkpoints/response_based")
    log.info(f"  alpha={cfg.alpha}  T={cfg.temperature}  epochs={cfg.epochs}  lr={cfg.lr}")

    distiller = ResponseBasedDistiller(teacher, student, config=cfg)

    log.section("Teacher baseline")
    log.step("Evaluating pretrained teacher accuracy")
    teacher_acc = distiller._validate(val_loader)
    log.done(f"Teacher accuracy: {teacher_acc:.4f}")

    log.section(f"Distillation — {cfg.epochs} epochs")
    log.info(f"  Checkpoints → {cfg.checkpoint_dir}")
    log.start_heartbeat(interval=60, message="Training epoch in progress…")
    history = distiller.train(train_loader, val_loader, verbose=True)
    log.stop_heartbeat()

    log.section("Results")
    log.metric("Best student accuracy", f"{distiller.best_val_acc:.4f}")
    log.metric("Teacher accuracy",      f"{teacher_acc:.4f}")
    log.metric("Recovery",
               f"{distiller.best_val_acc / max(teacher_acc, 1e-9) * 100:.1f}% of teacher")

    log.section("Model profile")
    import torch
    distiller.profile(torch.randn(1, 3, 32, 32))
    log.summary()


if __name__ == "__main__":
    main()
