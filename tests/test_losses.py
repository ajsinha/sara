# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.8.3
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ajsinha/sara
"""
tests/test_losses.py
====================
Unit tests for all loss functions in kd.core.losses.

All tests use small synthetic tensors and run on CPU.
No model downloads or API calls required.
"""

import pytest
import torch
import torch.nn.functional as F

from sara.core.losses import (
    DistillationLoss,
    FeatureDistillationLoss,
    AttentionTransferLoss,
    RKDLoss,
    SelfDistillationLoss,
)


# ══════════════════════════════════════════════════════════════════════════════
# DistillationLoss
# ══════════════════════════════════════════════════════════════════════════════

class TestDistillationLoss:

    def test_output_is_scalar(self):
        loss_fn = DistillationLoss()
        B, C = 8, 10
        loss = loss_fn(
            torch.randn(B, C),
            torch.randn(B, C),
            torch.randint(0, C, (B,)),
        )
        assert loss.shape == torch.Size([]), "Loss must be a scalar"

    def test_loss_is_non_negative(self):
        loss_fn = DistillationLoss()
        B, C = 8, 10
        loss = loss_fn(
            torch.randn(B, C),
            torch.randn(B, C),
            torch.randint(0, C, (B,)),
        )
        assert loss.item() >= 0.0

    def test_identical_logits_kd_term_near_zero(self):
        """When student == teacher, the KL term should be ~0."""
        loss_fn  = DistillationLoss(alpha=1.0, temperature=4.0)  # pure KD
        logits   = torch.randn(16, 10)
        labels   = torch.randint(0, 10, (16,))
        loss     = loss_fn(logits, logits.clone(), labels)
        assert loss.item() < 1e-4, f"Expected ~0 KD loss, got {loss.item()}"

    def test_alpha_zero_uses_only_ce(self):
        """alpha=0 should equal cross-entropy only."""
        B, C = 8, 5
        s_logits = torch.randn(B, C)
        t_logits = torch.randn(B, C)
        labels   = torch.randint(0, C, (B,))
        loss_fn  = DistillationLoss(alpha=0.0, temperature=4.0)
        loss_kd  = loss_fn(s_logits, t_logits, labels)
        loss_ce  = F.cross_entropy(s_logits, labels)
        assert abs(loss_kd.item() - loss_ce.item()) < 1e-5

    def test_alpha_one_ignores_ce(self):
        """alpha=1 should not include any CE contribution."""
        B, C = 8, 5
        labels       = torch.randint(0, C, (B,))
        s_logits     = torch.randn(B, C)
        t_logits_a   = torch.randn(B, C)
        t_logits_b   = torch.randn(B, C)
        loss_fn = DistillationLoss(alpha=1.0, temperature=4.0)
        # Same student logits, different labels → CE differs but shouldn't matter
        labels_b = (labels + 1) % C
        l1 = loss_fn(s_logits, t_logits_a, labels)
        l2 = loss_fn(s_logits, t_logits_a, labels_b)
        # KD term uses same teacher and student → losses should be equal
        assert abs(l1.item() - l2.item()) < 1e-4

    def test_gradient_flows_to_student(self):
        """Backpropagation should update student logits only."""
        B, C = 8, 5
        s_logits = torch.randn(B, C, requires_grad=True)
        t_logits = torch.randn(B, C)
        labels   = torch.randint(0, C, (B,))
        loss_fn  = DistillationLoss()
        loss     = loss_fn(s_logits, t_logits, labels)
        loss.backward()
        assert s_logits.grad is not None
        assert t_logits.grad is None, "Teacher logits must NOT accumulate gradients"

    @pytest.mark.parametrize("T", [1.0, 2.0, 4.0, 8.0])
    def test_various_temperatures(self, T):
        loss_fn = DistillationLoss(temperature=T)
        loss = loss_fn(torch.randn(8, 10), torch.randn(8, 10), torch.randint(0, 10, (8,)))
        assert loss.item() >= 0.0

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            DistillationLoss(alpha=1.5)

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            DistillationLoss(temperature=-1.0)

    def test_repr_contains_params(self):
        loss_fn = DistillationLoss(alpha=0.7, temperature=5.0)
        assert "0.7" in repr(loss_fn) and "5.0" in repr(loss_fn)


# ══════════════════════════════════════════════════════════════════════════════
# FeatureDistillationLoss
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureDistillationLoss:

    def test_output_is_scalar(self):
        loss_fn = FeatureDistillationLoss(student_channels=32, teacher_channels=64)
        s_feat  = torch.randn(4, 32, 8, 8)
        t_feat  = torch.randn(4, 64, 8, 8)
        loss    = loss_fn(s_feat, t_feat)
        assert loss.shape == torch.Size([])

    def test_loss_is_non_negative(self):
        loss_fn = FeatureDistillationLoss(student_channels=16, teacher_channels=32)
        loss    = loss_fn(torch.randn(4, 16, 4, 4), torch.randn(4, 32, 4, 4))
        assert loss.item() >= 0.0

    def test_grad_flows_to_student_not_teacher(self):
        loss_fn = FeatureDistillationLoss(student_channels=8, teacher_channels=8)
        s = torch.randn(2, 8, 4, 4, requires_grad=True)
        t = torch.randn(2, 8, 4, 4, requires_grad=True)
        loss_fn(s, t).backward()
        assert s.grad is not None
        assert t.grad is None

    def test_same_channels_adapter_is_identity_like(self):
        """With same channels, loss should be computable (no shape error)."""
        loss_fn = FeatureDistillationLoss(student_channels=16, teacher_channels=16)
        loss    = loss_fn(torch.randn(2, 16, 4, 4), torch.randn(2, 16, 4, 4))
        assert loss.item() >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# AttentionTransferLoss
# ══════════════════════════════════════════════════════════════════════════════

class TestAttentionTransferLoss:

    def test_output_is_scalar(self):
        loss_fn = AttentionTransferLoss()
        s_feats = [torch.randn(4, 8, 4, 4), torch.randn(4, 16, 2, 2)]
        t_feats = [torch.randn(4, 16, 4, 4), torch.randn(4, 32, 2, 2)]
        loss    = loss_fn(s_feats, t_feats)
        assert loss.shape == torch.Size([])

    def test_identical_attention_maps_loss_is_zero(self):
        loss_fn = AttentionTransferLoss()
        feat    = torch.randn(4, 16, 4, 4)
        loss    = loss_fn([feat], [feat.clone()])
        assert loss.item() < 1e-5

    def test_mismatched_list_lengths_raises(self):
        loss_fn = AttentionTransferLoss()
        with pytest.raises(ValueError):
            loss_fn([torch.randn(2, 8, 4, 4)], [])

    def test_empty_lists_raises(self):
        loss_fn = AttentionTransferLoss()
        with pytest.raises(ValueError):
            loss_fn([], [])

    def test_attention_map_shape(self):
        feat = torch.randn(4, 16, 8, 8)
        attn = AttentionTransferLoss._attention_map(feat)
        assert attn.shape == (4, 64), f"Expected (4, 64), got {attn.shape}"

    def test_attention_map_is_l2_normalised(self):
        feat  = torch.randn(4, 16, 8, 8)
        attn  = AttentionTransferLoss._attention_map(feat)
        norms = attn.norm(dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# RKDLoss
# ══════════════════════════════════════════════════════════════════════════════

class TestRKDLoss:

    def test_returns_dict_with_required_keys(self):
        loss_fn = RKDLoss()
        result  = loss_fn(torch.randn(8, 32), torch.randn(8, 32))
        assert "total" in result
        assert "distance" in result
        assert "angle" in result

    def test_total_is_weighted_sum(self):
        ld, la  = 2.0, 3.0
        loss_fn = RKDLoss(lambda_d=ld, lambda_a=la)
        result  = loss_fn(torch.randn(6, 16), torch.randn(6, 16))
        expected = ld * result["distance"] + la * result["angle"]
        assert torch.allclose(result["total"], expected, atol=1e-5)

    def test_all_terms_non_negative(self):
        loss_fn = RKDLoss()
        result  = loss_fn(torch.randn(8, 32), torch.randn(8, 32))
        for key, val in result.items():
            assert val.item() >= 0.0, f"{key} should be non-negative"

    def test_gradient_flows_to_student_only(self):
        loss_fn = RKDLoss()
        s = torch.randn(6, 16, requires_grad=True)
        t = torch.randn(6, 16, requires_grad=True)
        result = loss_fn(s, t)
        result["total"].backward()
        assert s.grad is not None
        assert t.grad is None


# ══════════════════════════════════════════════════════════════════════════════
# SelfDistillationLoss
# ══════════════════════════════════════════════════════════════════════════════

class TestSelfDistillationLoss:

    def test_returns_scalar_and_details(self):
        loss_fn = SelfDistillationLoss(exit_weights=(0.3, 0.4, 1.0))
        exits   = [torch.randn(8, 5), torch.randn(8, 5), torch.randn(8, 5)]
        labels  = torch.randint(0, 5, (8,))
        loss, details = loss_fn(exits, labels)
        assert loss.shape == torch.Size([])
        assert "exit_1" in details
        assert "exit_2" in details
        assert "exit_3" in details

    def test_loss_non_negative(self):
        loss_fn = SelfDistillationLoss()
        exits   = [torch.randn(4, 5), torch.randn(4, 5), torch.randn(4, 5)]
        labels  = torch.randint(0, 5, (4,))
        loss, _ = loss_fn(exits, labels)
        assert loss.item() >= 0.0

    def test_wrong_number_of_exits_raises(self):
        loss_fn = SelfDistillationLoss(exit_weights=(0.3, 0.7))
        exits   = [torch.randn(4, 5)] * 3   # 3 exits but 2 weights
        labels  = torch.randint(0, 5, (4,))
        with pytest.raises(ValueError, match="exit logits"):
            loss_fn(exits, labels)

    def test_gradient_flows_to_all_exits(self):
        loss_fn = SelfDistillationLoss()
        exits   = [torch.randn(4, 5, requires_grad=True) for _ in range(3)]
        labels  = torch.randint(0, 5, (4,))
        loss, _ = loss_fn(exits, labels)
        loss.backward()
        for i, e in enumerate(exits):
            assert e.grad is not None, f"exit_{i+1} has no gradient"

    @pytest.mark.parametrize("T", [1.0, 3.0, 6.0])
    def test_various_temperatures(self, T):
        loss_fn = SelfDistillationLoss(temperature=T)
        exits   = [torch.randn(4, 5)] * 3
        labels  = torch.randint(0, 5, (4,))
        loss, _ = loss_fn(exits, labels)
        assert loss.item() >= 0.0
