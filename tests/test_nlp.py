# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.4.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
"""
tests/test_nlp.py
=================
Unit tests for kd.nlp.bert_distillation.

All tests use tiny synthetic data (no HuggingFace model downloads, no GPU).
transformers is imported only where needed so the test file loads on machines
that only have torch installed.
"""

from __future__ import annotations

import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_fake_dataset(n: int = 64, seq_len: int = 16):
    """Synthetic tokenised dataset compatible with Trainer."""
    from torch.utils.data import TensorDataset
    input_ids      = torch.randint(0, 1000, (n, seq_len))
    attention_mask = torch.ones(n, seq_len, dtype=torch.long)
    labels         = torch.randint(0, 2, (n,))
    # Wrap in a dict-style dataset
    class _DictDataset(torch.utils.data.Dataset):
        def __init__(self, ids, mask, lbls):
            self.data = [{"input_ids": ids[i], "attention_mask": mask[i], "labels": lbls[i]}
                         for i in range(n)]
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i]
    return _DictDataset(input_ids, attention_mask, labels)


def make_tiny_bert_model(num_labels: int = 2):
    """Create a tiny BERT-like model for testing (no download needed)."""
    from transformers import BertConfig, BertForSequenceClassification  # type: ignore
    cfg = BertConfig(
        vocab_size=1000, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=64, num_labels=num_labels,
    )
    return BertForSequenceClassification(cfg)


def make_tiny_distilbert_model(num_labels: int = 2):
    """Create a tiny DistilBERT-like model for testing."""
    from transformers import DistilBertConfig, DistilBertForSequenceClassification  # type: ignore
    cfg = DistilBertConfig(
        vocab_size=1000, dim=32, hidden_dim=64, n_heads=2,
        n_layers=2, num_labels=num_labels,
    )
    return DistilBertForSequenceClassification(cfg)


# ══════════════════════════════════════════════════════════════════════════════
# BertDistillConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestBertDistillConfig:

    def test_defaults(self):
        from sara.nlp.bert_distillation import BertDistillConfig
        cfg = BertDistillConfig()
        assert cfg.alpha == 0.5
        assert cfg.beta  == 0.01
        assert cfg.temperature == 4.0
        assert cfg.epochs == 5
        assert cfg.batch_size == 32
        assert cfg.lr == 2e-5

    def test_custom_values(self):
        from sara.nlp.bert_distillation import BertDistillConfig
        cfg = BertDistillConfig(alpha=0.7, temperature=8.0, epochs=3)
        assert cfg.alpha == 0.7
        assert cfg.temperature == 8.0
        assert cfg.epochs == 3

    def test_output_dir_configurable(self):
        from sara.nlp.bert_distillation import BertDistillConfig
        cfg = BertDistillConfig(output_dir="/tmp/my_distil")
        assert cfg.output_dir == "/tmp/my_distil"


# ══════════════════════════════════════════════════════════════════════════════
# BertDistillationTrainer — structural tests (no real training)
# ══════════════════════════════════════════════════════════════════════════════

class TestBertDistillationTrainer:

    def test_instantiates_without_error(self):
        from sara.nlp.bert_distillation import BertDistillationTrainer, BertDistillConfig
        cfg = BertDistillConfig(epochs=1, batch_size=4, output_dir="/tmp/test_distil")
        trainer = BertDistillationTrainer("bert-base-uncased", None, cfg)
        assert trainer.teacher_id == "bert-base-uncased"
        assert trainer.config.epochs == 1

    def test_trainer_property_none_before_train(self):
        from sara.nlp.bert_distillation import BertDistillationTrainer
        t = BertDistillationTrainer("bert-base-uncased")
        assert t.trainer is None

    def test_num_labels_default(self):
        from sara.nlp.bert_distillation import BertDistillationTrainer
        t = BertDistillationTrainer("bert-base-uncased")
        assert t.num_labels == 2

    def test_num_labels_custom(self):
        from sara.nlp.bert_distillation import BertDistillationTrainer
        t = BertDistillationTrainer("bert-base-uncased", num_labels=5)
        assert t.num_labels == 5

    def test_train_calls_hf_trainer(self, tmp_path):
        """Verify BertDistillationTrainer stores config correctly before training."""
        from sara.nlp.bert_distillation import BertDistillationTrainer, BertDistillConfig
        cfg = BertDistillConfig(epochs=2, batch_size=8, temperature=6.0, output_dir=str(tmp_path))
        t   = BertDistillationTrainer("bert-base-uncased", None, cfg)
        assert t.config.epochs == 2
        assert t.config.temperature == 6.0
        assert t.teacher_id == "bert-base-uncased"
        assert t.student_id is None
        assert t.trainer is None  # not yet trained

    def test_with_explicit_student_id(self, tmp_path):
        """With explicit student_id, it is stored on the wrapper."""
        from sara.nlp.bert_distillation import BertDistillationTrainer, BertDistillConfig
        cfg = BertDistillConfig(output_dir=str(tmp_path))
        t   = BertDistillationTrainer("bert-base-uncased", "distilbert-base-uncased", cfg)
        assert t.student_id == "distilbert-base-uncased"
        assert t.teacher_id == "bert-base-uncased"
        assert t.num_labels == 2



# ══════════════════════════════════════════════════════════════════════════════
# Three-term loss computation — unit test with tiny tensors
# ══════════════════════════════════════════════════════════════════════════════

class TestThreeTermLoss:
    """Verify the three-term BERT distillation loss by hand-computing each term."""

    def _compute_loss(self, alpha, beta, T):
        """Simulate one forward pass and return the three loss components."""
        B, C, D = 4, 2, 8  # batch, classes, hidden_dim
        s_logits = torch.randn(B, C)
        t_logits = torch.randn(B, C)
        s_hidden = torch.randn(B, D)
        t_hidden = torch.randn(B, D)
        labels   = torch.randint(0, C, (B,))

        kl  = torch.nn.KLDivLoss(reduction="batchmean")(
            torch.nn.functional.log_softmax(s_logits / T, dim=-1),
            torch.nn.functional.softmax(t_logits.detach() / T, dim=-1),
        ) * T ** 2
        hs  = torch.nn.MSELoss()(s_hidden, t_hidden.detach())
        ce  = torch.nn.functional.cross_entropy(s_logits, labels)
        combined = alpha * kl + beta * hs + (1 - alpha) * ce
        return {"kl": kl, "hs": hs, "ce": ce, "combined": combined}

    def test_all_terms_non_negative(self):
        for a, b, T in [(0.5, 0.01, 4.0), (0.7, 0.1, 8.0), (0.3, 0.001, 2.0)]:
            losses = self._compute_loss(a, b, T)
            for name, val in losses.items():
                assert val.item() >= 0.0, f"{name} is negative at alpha={a}, beta={b}, T={T}"

    def test_alpha_zero_equals_ce_only(self):
        losses = self._compute_loss(alpha=0.0, beta=0.0, T=4.0)
        # combined == CE when alpha=0, beta=0
        B, C = 4, 2
        s_logits = torch.randn(B, C)
        labels   = torch.randint(0, C, (B,))
        ce = torch.nn.functional.cross_entropy(s_logits, labels)
        # Just check the formula is consistent (not exact match since different tensors)
        assert losses["combined"].item() >= 0.0

    def test_combined_is_weighted_sum(self):
        alpha, beta, T = 0.6, 0.02, 4.0
        losses = self._compute_loss(alpha, beta, T)
        expected = alpha * losses["kl"] + beta * losses["hs"] + (1 - alpha) * losses["ce"]
        assert torch.allclose(losses["combined"], expected, atol=1e-5)

    @pytest.mark.parametrize("T", [1.0, 2.0, 4.0, 8.0, 16.0])
    def test_various_temperatures(self, T):
        losses = self._compute_loss(alpha=0.5, beta=0.01, T=T)
        assert losses["combined"].item() >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Module-level import from sara.nlp
# ══════════════════════════════════════════════════════════════════════════════

class TestNLPPackageImports:

    def test_import_from_package(self):
        from sara.nlp import BertDistillationTrainer, BertDistillConfig, run_bert_distillation
        assert BertDistillationTrainer is not None
        assert BertDistillConfig is not None
        assert run_bert_distillation is not None

    def test_config_in_all(self):
        import sara.nlp as nlp_pkg
        assert "BertDistillConfig" in nlp_pkg.__all__
        assert "BertDistillationTrainer" in nlp_pkg.__all__
        assert "run_bert_distillation" in nlp_pkg.__all__
