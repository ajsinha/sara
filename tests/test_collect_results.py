# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.6.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
"""
tests/test_collect_results.py
==============================
Tests for the experiment result collection and aggregation pipeline.
All tests use synthetic in-memory data — no real experiment files needed.
"""

from __future__ import annotations
import json, math, tempfile
from pathlib import Path
import pytest

# ── Synthetic result factory ───────────────────────────────────────────────

def make_run(config: str, seed: int, a: float, b: float, c: float, d: float) -> dict:
    """Create a synthetic ablation run dict."""
    conds = {
        "A": ("KD-SPAR (student self-proposed)", a),
        "B": ("Externally proposed (teacher)",   b),
        "C": ("Random instructions",              c),
        "D": ("No prompt tuning (baseline)",      d),
    }
    return {
        "timestamp": f"20250415_0{seed}0000",
        "config":    config,
        "teacher":   "llama3.1:8b",
        "student":   "llama3.2:3b",
        "seed":      seed,
        "conditions": [
            {
                "condition":   cond,
                "description": desc,
                "val_metrics": {
                    "mean_kd_score":     kd,
                    "citation_fidelity": kd * 0.9,
                    "hedge_match":       1.0,
                },
                "build_time_s": 120,
                "final_prompt": f"Prompt for {cond}",
            }
            for cond, (desc, kd) in conds.items()
        ],
    }


# ── Tests for helpers ──────────────────────────────────────────────────────

class TestHelpers:

    def test_mean_empty(self):
        from experiments.collect_results import _mean
        assert _mean([]) == 0.0

    def test_mean_values(self):
        from experiments.collect_results import _mean
        assert abs(_mean([1.0, 2.0, 3.0]) - 2.0) < 1e-9

    def test_std_single(self):
        from experiments.collect_results import _std
        assert _std([5.0]) == 0.0

    def test_std_two(self):
        from experiments.collect_results import _std
        # std of [0, 1] = 0.707...
        assert abs(_std([0.0, 1.0]) - math.sqrt(0.5)) < 1e-9

    def test_interpret_ab_gap_strong(self):
        from experiments.collect_results import interpret_ab_gap
        assert "strong" in interpret_ab_gap(0.025).lower()

    def test_interpret_ab_gap_moderate(self):
        from experiments.collect_results import interpret_ab_gap
        assert "moderate" in interpret_ab_gap(0.015).lower()

    def test_interpret_ab_gap_negative(self):
        from experiments.collect_results import interpret_ab_gap
        assert "negative" in interpret_ab_gap(-0.01).lower()


# ── Tests for aggregation ──────────────────────────────────────────────────

class TestAggregate:

    def _make_three_seeds(self, config="cfg1"):
        return [
            make_run(config, 42,  a=0.39, b=0.36, c=0.34, d=0.34),
            make_run(config, 123, a=0.38, b=0.36, c=0.34, d=0.34),
            make_run(config, 777, a=0.37, b=0.36, c=0.34, d=0.34),
        ]

    def test_aggregate_produces_correct_structure(self):
        from experiments.collect_results import aggregate
        raw = self._make_three_seeds()
        agg = aggregate(raw)
        assert agg.n_total_runs == 3
        assert len(agg.configs) == 1
        assert "A" in agg.configs[0].conditions
        assert "B" in agg.configs[0].conditions

    def test_ab_gap_positive(self):
        from experiments.collect_results import aggregate
        raw = self._make_three_seeds()
        agg = aggregate(raw)
        assert agg.overall_ab_gap > 0

    def test_ab_gap_correct_value(self):
        from experiments.collect_results import aggregate
        raw = [make_run("cfg", 1, a=0.40, b=0.35, c=0.33, d=0.33)]
        agg = aggregate(raw)
        assert abs(agg.overall_ab_gap - 0.05) < 0.001

    def test_hypothesis_supported_when_a_best(self):
        from experiments.collect_results import aggregate
        raw = self._make_three_seeds()
        agg = aggregate(raw)
        assert agg.configs[0].hypothesis_supported is True

    def test_hypothesis_not_supported_when_b_beats_a(self):
        from experiments.collect_results import aggregate
        raw = [make_run("cfg", 1, a=0.35, b=0.40, c=0.33, d=0.33)]
        agg = aggregate(raw)
        assert agg.configs[0].hypothesis_supported is False

    def test_multiple_configs_tracked_separately(self):
        from experiments.collect_results import aggregate
        raw = (
            self._make_three_seeds("llama8b-llama3b")
            + self._make_three_seeds("qwen7b-llama3b")
        )
        agg = aggregate(raw)
        assert len(agg.configs) == 2
        labels = [c.config_label for c in agg.configs]
        assert "llama8b-llama3b" in labels
        assert "qwen7b-llama3b" in labels

    def test_paper_table_rows_all_conditions(self):
        from experiments.collect_results import aggregate
        raw = self._make_three_seeds()
        agg = aggregate(raw)
        conds = [r["condition"] for r in agg.paper_table_rows]
        assert "A" in conds
        assert "B" in conds
        assert "C" in conds
        assert "D" in conds

    def test_paper_table_row_has_required_keys(self):
        from experiments.collect_results import aggregate
        raw = self._make_three_seeds()
        agg = aggregate(raw)
        row = agg.paper_table_rows[0]
        for key in ["condition", "kd_mean", "kd_std", "kd_str",
                    "delta_vs_d", "delta_str", "cit_mean", "cit_str"]:
            assert key in row, f"Missing key: {key}"

    def test_empty_input_returns_empty(self):
        from experiments.collect_results import aggregate
        agg = aggregate([])
        assert agg.n_total_runs == 0
        assert len(agg.configs) == 0

    def test_recommendation_strong_when_gap_large(self):
        from experiments.collect_results import aggregate
        raw = [make_run("cfg", 1, a=0.42, b=0.38, c=0.33, d=0.33)]
        agg = aggregate(raw)
        assert "STRONG" in agg.recommendation.upper() or "strong" in agg.recommendation.lower()


# ── Tests for load_all_results (file I/O) ─────────────────────────────────

class TestLoadResults:

    def test_load_from_tmpdir(self, tmp_path):
        from experiments.collect_results import load_all_results
        run = make_run("test_cfg", 42, a=0.38, b=0.36, c=0.34, d=0.34)
        fp  = tmp_path / "ablation_test_cfg_seed42_20250101.json"
        with open(fp, "w") as f:
            json.dump(run, f)
        loaded = load_all_results(results_dir=tmp_path)
        assert len(loaded) == 1
        assert loaded[0]["config"] == "test_cfg"

    def test_empty_dir_returns_empty(self, tmp_path):
        from experiments.collect_results import load_all_results
        loaded = load_all_results(results_dir=tmp_path)
        assert loaded == []

    def test_config_filter_works(self, tmp_path):
        from experiments.collect_results import load_all_results
        for name in ["ablation_llama8b_s1.json", "ablation_qwen7b_s1.json"]:
            run = make_run("cfg", 1, 0.38, 0.36, 0.34, 0.34)
            with open(tmp_path / name, "w") as f:
                json.dump(run, f)
        loaded = load_all_results(tmp_path, config_filter="llama8b")
        assert len(loaded) == 1

    def test_save_aggregated_creates_files(self, tmp_path):
        from experiments.collect_results import aggregate, save_aggregated
        raw = [make_run("cfg", 42, 0.38, 0.36, 0.34, 0.34)]
        agg = aggregate(raw)
        jp, tp = save_aggregated(agg, "test report", results_dir=tmp_path)
        assert jp.exists()
        assert tp.exists()
        with open(jp) as f:
            saved = json.load(f)
        assert saved["n_total_runs"] == 1


# ── Tests for ConditionStats ───────────────────────────────────────────────

class TestConditionStats:

    def test_kd_str_single_seed(self):
        from experiments.collect_results import ConditionStats
        s = ConditionStats("A","desc",1, 0.382,0.006, 0.9,0.01, 1.0,0.02)
        assert "0.382" in s.kd_str

    def test_n_runs_correct(self):
        from experiments.collect_results import aggregate
        raw = [make_run("c", i, 0.38, 0.36, 0.34, 0.34) for i in range(5)]
        agg = aggregate(raw)
        assert agg.configs[0].conditions["A"].n_runs == 5
