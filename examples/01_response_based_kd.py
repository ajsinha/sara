"""
examples/01_response_based_kd.py
=================================
Quickstart: distil a ResNet-50 teacher into a MobileNetV2 student on CIFAR-10.

Run:
    python examples/01_response_based_kd.py

Requirements:
    pip install torch torchvision
"""

from sara.vision.response_based import (
    ResponseBasedDistiller,
    VisionDistillConfig,
    build_cifar10_loaders,
    build_default_teacher_student,
)

def main():
    print("Building data loaders …")
    train_loader, val_loader = build_cifar10_loaders(batch_size=128)

    print("Building teacher (ResNet-50) and student (MobileNetV2) …")
    teacher, student = build_default_teacher_student(num_classes=10)

    cfg = VisionDistillConfig(
        alpha=0.6,
        temperature=4.0,
        epochs=30,
        lr=1e-3,
        checkpoint_dir="./checkpoints/response_based",
    )

    distiller = ResponseBasedDistiller(teacher, student, config=cfg)

    print(f"\nTeacher accuracy (pretrained):")
    teacher_acc = distiller._validate(val_loader)
    print(f"  {teacher_acc:.4f}")

    print("\nStarting distillation …")
    history = distiller.train(train_loader, val_loader, verbose=True)

    print(f"\nBest student accuracy: {distiller.best_val_acc:.4f}")

    print("\nProfiling …")
    import torch
    distiller.profile(torch.randn(1, 3, 32, 32))


if __name__ == "__main__":
    main()
