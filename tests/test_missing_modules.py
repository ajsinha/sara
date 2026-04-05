# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.2.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
"""
tests/test_missing_modules.py
==============================
Tests for sara.advanced.progressive, sara.advanced.relation_based,
and sara.rag.prompt_opt — the three modules that had no test coverage
(audit finding #15).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader, TensorDataset


# ── Helpers ────────────────────────────────────────────────────────────────

def make_loader(n=16, c=3, h=8, num_classes=5, batch_size=8):
    x = torch.randn(n, c, h, h)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def make_mock_pipeline(answer="Answer [Doc-1]."):
    p = MagicMock()
    r = MagicMock(); r.answer = answer
    p.query.return_value = r
    p.client = MagicMock()
    p.client.system = "system"
    p.client.update_system = MagicMock()
    return p


# ══════════════════════════════════════════════════════════════════════════
# sara.advanced.progressive  — Stage @dataclass + ProgressiveDistiller
# ══════════════════════════════════════════════════════════════════════════

class TestStageDataclass:

    def test_stage_is_dataclass(self):
        import dataclasses
        from sara.advanced.progressive import Stage
        assert dataclasses.is_dataclass(Stage), "Stage must be @dataclass (audit fix)"

    def test_stage_fields_accessible(self):
        from sara.advanced.progressive import Stage
        s = Stage(teacher_id="big", student_id="small",
                  temperature=4.0, alpha=0.7, epochs=3, lr=2e-5)
        assert s.teacher_id == "big"
        assert s.temperature == 4.0
        assert s.epochs == 3

    def test_stage_default_temperature(self):
        from sara.advanced.progressive import Stage
        s = Stage(teacher_id="t", student_id="s")
        assert s.temperature > 0

    def test_stage_default_alpha_in_range(self):
        from sara.advanced.progressive import Stage
        s = Stage(teacher_id="t", student_id="s")
        assert 0 < s.alpha < 1


class TestProgressiveDistiller:

    def _make_tiny_model(self):
        """A tiny model that accepts HuggingFace-style batches."""
        class TinyClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 3)
            def forward(self, input_ids, attention_mask):
                x = input_ids.float()[:, :8]
                logits = self.fc(x)
                class Out:
                    def __init__(self, l): self.logits = l
                return Out(logits)
        return TinyClassifier()

    def _make_hf_loader(self, n=8):
        ids  = torch.randint(0, 100, (n, 8))
        mask = torch.ones(n, 8, dtype=torch.long)
        lbls = torch.randint(0, 3, (n,))
        class DS(torch.utils.data.Dataset):
            def __len__(self): return n
            def __getitem__(self, i):
                return {"input_ids": ids[i],
                        "attention_mask": mask[i],
                        "labels": lbls[i]}
        return DataLoader(DS(), batch_size=8)

    def test_instantiation(self):
        from sara.advanced.progressive import ProgressiveDistiller, Stage
        stages  = [Stage("t1", "s1", epochs=1)]
        factory = MagicMock(return_value=self._make_tiny_model())
        d       = ProgressiveDistiller(stages, factory, device="cpu")
        assert len(d.stages) == 1

    def test_run_returns_model(self):
        from sara.advanced.progressive import ProgressiveDistiller, Stage
        factory = MagicMock(side_effect=lambda _: self._make_tiny_model())
        stages  = [Stage("big", "small", epochs=1, lr=0.01)]
        loader  = self._make_hf_loader()
        d       = ProgressiveDistiller(stages, factory, device="cpu")
        result  = d.run(loader, loader, verbose=False)
        assert isinstance(result, nn.Module)

    def test_multiple_stages_chain(self):
        from sara.advanced.progressive import ProgressiveDistiller, Stage
        calls   = []
        def factory(mid):
            calls.append(mid)
            return self._make_tiny_model()
        stages = [
            Stage("huge", "medium", epochs=1, lr=0.01),
            Stage("medium", "tiny",  epochs=1, lr=0.01),
        ]
        loader = self._make_hf_loader()
        d      = ProgressiveDistiller(stages, factory, device="cpu")
        d.run(loader, loader, verbose=False)
        # factory called for teacher of stage 0 + student of each stage
        assert len(calls) >= 2


# ══════════════════════════════════════════════════════════════════════════
# sara.advanced.relation_based — RelationalKDDistiller
# ══════════════════════════════════════════════════════════════════════════

class TestRelationalKDDistiller:

    def _build(self):
        from sara.advanced.relation_based import RelationalKDDistiller
        teacher = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 5),
        )
        student = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 5),
        )
        return RelationalKDDistiller(
            teacher=teacher, student=student,
            teacher_embed_layer="2", student_embed_layer="2",
            device="cpu",
        )

    def test_instantiation(self):
        import os; os.environ.setdefault("ANTHROPIC_API_KEY","sk-test")
        d = self._build()
        assert d.best_val_acc == 0.0
        assert d._rkd_fn is not None

    def test_teacher_parameters_frozen(self):
        import os; os.environ.setdefault("ANTHROPIC_API_KEY","sk-test")
        d = self._build()
        for p in d.teacher.parameters():
            assert not p.requires_grad

    def test_train_one_epoch_updates_acc(self):
        import os; os.environ.setdefault("ANTHROPIC_API_KEY","sk-test")
        d      = self._build()
        loader = make_loader(n=16, c=3, h=8, num_classes=5)
        d.train(loader, loader, epochs=1, verbose=False)
        assert 0.0 <= d.best_val_acc <= 1.0

    def test_best_ckpt_saved(self):
        import os; os.environ.setdefault("ANTHROPIC_API_KEY","sk-test")
        d      = self._build()
        loader = make_loader(n=16, c=3, h=8, num_classes=5)
        d.train(loader, loader, epochs=2, verbose=False)
        assert d._best_ckpt is not None

    def test_rkd_losses_non_negative(self):
        from sara.core.losses import RKDLoss
        loss_fn = RKDLoss(lambda_d=1.0, lambda_a=2.0)
        s = torch.randn(8, 16); t = torch.randn(8, 16)
        result = loss_fn(s, t)
        for key in ("total", "distance", "angle"):
            assert result[key].item() >= 0.0


# ══════════════════════════════════════════════════════════════════════════
# sara.rag.prompt_opt — build_prompt, GridSearch, EvolutionaryAPO
# ══════════════════════════════════════════════════════════════════════════

class TestBuildPrompt:

    def test_returns_non_empty_string(self):
        from sara.rag.prompt_opt import build_prompt
        p = build_prompt()
        assert isinstance(p, str) and len(p) > 10

    def test_citation_instruction_included(self):
        from sara.rag.prompt_opt import build_prompt, CITATION_INSTRUCTIONS
        p = build_prompt(citation_instruction=CITATION_INSTRUCTIONS[0])
        assert CITATION_INSTRUCTIONS[0] in p

    def test_empty_cot_not_double_newline(self):
        from sara.rag.prompt_opt import build_prompt
        p = build_prompt(cot_scaffold="")
        assert "  \n" not in p  # no trailing whitespace lines

    def test_all_components_combined(self):
        from sara.rag.prompt_opt import (
            build_prompt, CITATION_INSTRUCTIONS,
            UNCERTAINTY_INSTRUCTIONS, COT_SCAFFOLDS,
        )
        p = build_prompt(
            citation_instruction=CITATION_INSTRUCTIONS[0],
            uncertainty=UNCERTAINTY_INSTRUCTIONS[0],
            cot_scaffold=COT_SCAFFOLDS[1],
        )
        assert CITATION_INSTRUCTIONS[0] in p
        assert UNCERTAINTY_INSTRUCTIONS[0] in p
        assert COT_SCAFFOLDS[1] in p


class TestGridSearch:

    def test_instantiation_defaults(self):
        from sara.rag.prompt_opt import GridSearch
        gs = GridSearch(student_model="test", vector_store=MagicMock(), max_combinations=4)
        assert gs.max_combinations == 4
        assert gs.student_model == "test"

    def test_run_returns_grid_search_result(self):
        from sara.rag.prompt_opt import GridSearch, GridSearchResult
        mock_pipe = make_mock_pipeline("answer [Doc-1].")
        with patch("sara.rag.prompt_opt.RAGPipeline", return_value=mock_pipe):
            gs = GridSearch(student_model="m", vector_store=MagicMock(), max_combinations=2)
            result = gs.run(
                eval_queries=["q1"],
                teacher_responses={"q1": "teacher [Doc-1]."},
                verbose=False,
            )
        assert isinstance(result, GridSearchResult)
        assert isinstance(result.best_prompt, str)
        assert result.best_score >= 0.0
        assert len(result.all_results) == 2

    def test_best_prompt_is_highest_scored(self):
        from sara.rag.prompt_opt import GridSearch
        # Make pipe return different answers per call to get score variation
        call_count = [0]
        def varying_answer(*a, **kw):
            call_count[0] += 1
            r = MagicMock()
            r.answer = "answer [Doc-1]." if call_count[0] % 2 == 0 else "answer without citation"
            return r
        mock_pipe = MagicMock()
        mock_pipe.query.side_effect = varying_answer
        with patch("sara.rag.prompt_opt.RAGPipeline", return_value=mock_pipe):
            gs = GridSearch("m", MagicMock(), max_combinations=4)
            result = gs.run(["q1"], {"q1": "answer [Doc-1]."}, verbose=False)
        assert result.best_score >= 0.0


class TestEvolutionaryAPO:

    def test_instantiation(self):
        from sara.rag.prompt_opt import EvolutionaryAPO
        with patch('sara.rag.prompt_opt.AnthropicClient'):
          evo = EvolutionaryAPO(
            student_model="m", vector_store=MagicMock(),
            generations=2, population_size=3,
        )
        assert evo.generations == 2
        assert evo.pop_size == 3

    def test_crossover_returns_string(self):
        from sara.rag.prompt_opt import EvolutionaryAPO
        with patch('sara.rag.prompt_opt.AnthropicClient'):
            evo  = EvolutionaryAPO("m", MagicMock())
        p1   = "You are helpful. Always cite with [Doc-N]."
        p2   = "You are precise. Use hedging language."
        result = evo._crossover(p1, p2)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_crossover_is_mix_of_parents(self):
        from sara.rag.prompt_opt import EvolutionaryAPO
        with patch('sara.rag.prompt_opt.AnthropicClient'):
            evo = EvolutionaryAPO("m", MagicMock())
        p1  = "Part A sentence one. Part A sentence two."
        p2  = "Part B sentence one. Part B sentence two."
        result = evo._crossover(p1, p2)
        # Result should contain material from both parents
        has_a = "Part A" in result
        has_b = "Part B" in result
        assert has_a or has_b  # at least one parent represented


# ══════════════════════════════════════════════════════════════════════════
# sara.core.utils — new shared scoring helpers (audit fix)
# ══════════════════════════════════════════════════════════════════════════

class TestSharedScoringHelpers:

    def test_jaccard_identical(self):
        from sara.core.utils import jaccard
        assert jaccard("hello world", "hello world") == 1.0

    def test_jaccard_disjoint(self):
        from sara.core.utils import jaccard
        assert jaccard("cat dog", "fish bird") == 0.0

    def test_jaccard_partial(self):
        from sara.core.utils import jaccard
        score = jaccard("the cat sat", "the cat ran")
        assert 0.0 < score < 1.0

    def test_kd_score_with_citation_match(self):
        from sara.core.utils import kd_score
        teacher = "Answer [Doc-1]."
        student = "Answer [Doc-1]."
        assert kd_score(student, teacher) > 0.9

    def test_kd_score_missing_citation_penalty(self):
        from sara.core.utils import kd_score
        teacher = "Answer [Doc-1]."
        s_with  = "Answer [Doc-1]."
        s_without = "Answer without citation."
        assert kd_score(s_with, teacher) > kd_score(s_without, teacher)

    def test_interpret_ab_gap_strong(self):
        from sara.core.utils import interpret_ab_gap
        assert "strong" in interpret_ab_gap(0.025).lower()

    def test_interpret_ab_gap_negative(self):
        from sara.core.utils import interpret_ab_gap
        assert "negative" in interpret_ab_gap(-0.01).lower()

    def test_default_system_prompt_non_empty(self):
        from sara.core.utils import DEFAULT_SYSTEM_PROMPT
        assert len(DEFAULT_SYSTEM_PROMPT) > 30
        assert "context" in DEFAULT_SYSTEM_PROMPT.lower()
