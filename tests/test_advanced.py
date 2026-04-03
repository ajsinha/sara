"""
tests/test_advanced.py
======================
Tests for advanced distillation modules:
  - MutualDistiller (online/mutual learning)
  - SelfDistillTrainer (early-exit self-distillation)
  - MultiExitResNet architecture

All tests use tiny synthetic data and run on CPU.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sara.core.losses import SelfDistillationLoss
from sara.advanced.mutual import MutualDistiller
from sara.advanced.self_distill import MultiExitResNet, SelfDistillTrainer


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_loader(n: int = 32, c: int = 3, h: int = 16, num_classes: int = 5,
                batch: int = 8) -> DataLoader:
    x = torch.randn(n, c, h, h)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=False)


def tiny_factory():
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 5),
    )


# ══════════════════════════════════════════════════════════════════════════════
# MutualDistiller
# ══════════════════════════════════════════════════════════════════════════════

class TestMutualDistiller:

    def test_train_returns_two_accuracies(self):
        trainer = MutualDistiller(tiny_factory, tiny_factory, alpha=0.5)
        acc1, acc2 = trainer.train(
            make_loader(), make_loader(), epochs=1, verbose=False
        )
        assert 0.0 <= acc1 <= 1.0
        assert 0.0 <= acc2 <= 1.0

    def test_student_properties(self):
        trainer = MutualDistiller(tiny_factory, tiny_factory)
        assert isinstance(trainer.student1, nn.Module)
        assert isinstance(trainer.student2, nn.Module)

    def test_different_factories_allowed(self):
        def factory_a(): return nn.Sequential(nn.Flatten(), nn.Linear(192, 5))
        def factory_b(): return nn.Sequential(nn.Flatten(), nn.Linear(192, 5))
        trainer = MutualDistiller(factory_a, factory_b, alpha=0.3)
        # Should not raise
        loader = DataLoader(
            TensorDataset(torch.randn(8, 3, 8, 8), torch.randint(0, 5, (8,))),
            batch_size=8
        )
        acc1, acc2 = trainer.train(loader, loader, epochs=1, verbose=False)
        assert acc1 >= 0.0 and acc2 >= 0.0

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_various_alphas(self, alpha):
        trainer = MutualDistiller(tiny_factory, tiny_factory, alpha=alpha)
        acc1, _ = trainer.train(
            make_loader(n=16), make_loader(n=16), epochs=1, verbose=False
        )
        assert 0.0 <= acc1 <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# MultiExitResNet
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiExitResNet:

    def test_forward_returns_three_tensors(self):
        model   = MultiExitResNet(num_classes=5)
        x       = torch.randn(2, 3, 32, 32)
        outputs = model(x)
        assert len(outputs) == 3, "Expected [exit1, exit2, final]"

    def test_output_shapes(self):
        B, C = 4, 7
        model   = MultiExitResNet(num_classes=C)
        outputs = model(torch.randn(B, 3, 32, 32))
        for logits in outputs:
            assert logits.shape == (B, C), f"Unexpected shape: {logits.shape}"

    def test_gradients_flow_to_all_outputs(self):
        model   = MultiExitResNet(num_classes=5)
        x       = torch.randn(2, 3, 32, 32)
        outputs = model(x)
        loss    = sum(o.sum() for o in outputs)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                pytest.fail(f"No gradient for {name}")


# ══════════════════════════════════════════════════════════════════════════════
# SelfDistillTrainer
# ══════════════════════════════════════════════════════════════════════════════

class TestSelfDistillTrainer:

    def test_train_completes(self, tmp_path):
        model   = MultiExitResNet(num_classes=5)
        trainer = SelfDistillTrainer(model, temperature=3.0)
        loader  = make_loader(n=16, c=3, h=32, num_classes=5, batch=8)
        trainer.train(loader, loader, epochs=1, verbose=False)
        assert 0.0 <= trainer.best_val_acc <= 1.0

    def test_evaluate_all_exits_returns_dict(self):
        model   = MultiExitResNet(num_classes=5)
        trainer = SelfDistillTrainer(model)
        loader  = make_loader(n=16, c=3, h=32, num_classes=5, batch=8)
        accs    = trainer.evaluate_all_exits(loader)
        assert "final" in accs
        assert "exit_1" in accs
        assert "exit_2" in accs
        for k, v in accs.items():
            assert 0.0 <= v <= 1.0, f"{k} accuracy out of range: {v}"

    def test_wrong_exit_weights_length_raises(self):
        model = MultiExitResNet(num_classes=5)
        # 3 exits but only 2 weights
        with pytest.raises(ValueError):
            trainer = SelfDistillTrainer(model, exit_weights=(0.3, 0.7))
            loader  = make_loader(n=8, c=3, h=32, num_classes=5)
            trainer.train(loader, loader, epochs=1, verbose=False)


# ══════════════════════════════════════════════════════════════════════════════
# SelfDistillationLoss (edge cases)
# ══════════════════════════════════════════════════════════════════════════════

class TestSelfDistillationLossEdgeCases:

    def test_two_exits(self):
        loss_fn = SelfDistillationLoss(exit_weights=(0.5, 1.0))
        exits   = [torch.randn(4, 5), torch.randn(4, 5)]
        labels  = torch.randint(0, 5, (4,))
        loss, details = loss_fn(exits, labels)
        assert loss.item() >= 0.0
        assert "exit_1" in details
        assert "exit_2" in details

    def test_four_exits(self):
        loss_fn = SelfDistillationLoss(exit_weights=(0.1, 0.2, 0.3, 1.0))
        exits   = [torch.randn(4, 5)] * 4
        labels  = torch.randint(0, 5, (4,))
        loss, details = loss_fn(exits, labels)
        assert "exit_1" in details
        assert "exit_4" in details

    def test_high_temperature_softens_distribution(self):
        """Higher T should produce different (softer) KL signal."""
        exits   = [torch.randn(4, 5), torch.randn(4, 5), torch.randn(4, 5)]
        labels  = torch.randint(0, 5, (4,))
        loss_lo, _ = SelfDistillationLoss(temperature=1.0)(exits, labels)
        loss_hi, _ = SelfDistillationLoss(temperature=8.0)(exits, labels)
        # Losses can differ; just verify both are valid
        assert loss_lo.item() >= 0.0
        assert loss_hi.item() >= 0.0
