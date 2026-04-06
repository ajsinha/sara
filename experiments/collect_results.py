# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.6.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
"""
experiments/collect_results.py
================================
Aggregate all ablation JSON files in experiments/results/ into a clean
publication-ready summary with mean ± std across seeds.

Usage
-----
    # After running experiments (any mix of API and Ollama runs)
    python experiments/collect_results.py

    # Summary of a specific config only
    python experiments/collect_results.py --config llama8b

    # Show per-query detail for a specific condition
    python experiments/collect_results.py --verbose

Output
------
    experiments/results/aggregated_results.json  ← used by patch_paper.py
    experiments/results/aggregated_results.txt   ← human-readable summary
    Console printout of publication-ready tables

The aggregated JSON is the input to patch_paper.py which rewrites
Section 20 of the paper with your actual measured numbers.
"""

from __future__ import annotations

import glob
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

RESULTS_DIR  = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Dataclasses ───────────────────────────────────────────────────────────

@dataclass
class ConditionStats:
    """Statistics for one condition (A/B/C/D) across all seeds."""
    condition:          str
    description:        str
    n_runs:             int
    mean_kd:            float
    std_kd:             float
    mean_citation:      float
    std_citation:       float
    mean_hedge:         float
    std_hedge:          float
    all_kd_scores:      list[float] = field(default_factory=list)
    configs_seen:       list[str]   = field(default_factory=list)

    @property
    def kd_str(self) -> str:
        return f"{self.mean_kd:.3f} ± {self.std_kd:.3f}"

    @property
    def cit_str(self) -> str:
        return f"{self.mean_citation:.3f} ± {self.std_citation:.3f}"

    @property
    def hedge_str(self) -> str:
        return f"{self.mean_hedge:.3f} ± {self.std_hedge:.3f}"


@dataclass
class ConfigSummary:
    """Summary for one model config (e.g. llama8b→llama3b) across all seeds."""
    config_label:   str
    teacher:        str
    student:        str
    n_seeds:        int
    conditions:     dict[str, ConditionStats]
    ab_gap_mean:    float
    ab_gap_std:     float
    hypothesis_supported: bool   # A > B and A > C and A > D

    @property
    def ab_gap_str(self) -> str:
        return f"{self.ab_gap_mean:+.3f} ± {self.ab_gap_std:.3f}"


@dataclass
class AggregatedResults:
    """All configs combined — the final publication dataset."""
    generated_at:     str
    n_total_runs:     int
    configs:          list[ConfigSummary]
    overall_ab_gap:   float       # pooled mean across all configs
    all_ab_gaps:      list[float]
    recommendation:   str
    paper_table_rows: list[dict]  # ready to insert into the paper


# ── Helpers ───────────────────────────────────────────────────────────────

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def _std(xs: list[float]) -> float:
    if len(xs) < 2: return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _stderr(xs: list[float]) -> float:
    return _std(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def interpret_ab_gap(gap: float) -> str:
    """Return a human-readable interpretation of the A-B gap."""
    if gap > 0.02:  return "strong — gap > 0.02 provides compelling evidence"
    if gap > 0.01:  return "moderate — gap 0.01–0.02 supports the claim"
    if gap > 0.005: return "suggestive — gap in the 0.005–0.01 range; more seeds would strengthen"
    if gap > 0.0:   return "marginal — positive but near noise floor"
    return "negative — external proposer matched or exceeded self-proposed"


# ── Loading ────────────────────────────────────────────────────────────────

def load_all_results(
    results_dir: Path = RESULTS_DIR,
    config_filter: Optional[str] = None,
) -> list[dict]:
    """
    Load all ablation JSON files from results_dir.

    Parameters
    ----------
    results_dir   : Directory containing ablation_*.json files
    config_filter : If set, only load files whose name contains this string

    Returns
    -------
    List of raw result dicts
    """
    pattern = str(results_dir / "ablation_*.json")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"No ablation JSON files found in {results_dir}/")
        print("Run the experiment first:")
        print("  python experiments/kd_spar_ablation_ollama.py --config 1 --iterations 3")
        return []

    if config_filter:
        files = [f for f in files if config_filter.lower() in f.lower()]
        print(f"Filtered to {len(files)} files matching '{config_filter}'")

    loaded = []
    for fp in files:
        try:
            with open(fp) as fh:
                data = json.load(fh)
                data["_source_file"] = fp
                loaded.append(data)
        except Exception as exc:
            print(f"  Warning: could not load {fp}: {exc}")

    print(f"Loaded {len(loaded)} result file(s)")
    return loaded


# ── Aggregation ────────────────────────────────────────────────────────────

def aggregate(raw_results: list[dict]) -> AggregatedResults:
    """
    Aggregate raw result files into per-config statistics.

    Groups files by config_label, then computes mean ± std across seeds
    for each condition.
    """
    # Group by config
    by_config: dict[str, list[dict]] = defaultdict(list)
    for r in raw_results:
        config = r.get("config", r.get("timestamp", "unknown"))
        by_config[config].append(r)

    config_summaries: list[ConfigSummary] = []
    all_ab_gaps: list[float] = []

    for config_label, runs in sorted(by_config.items()):
        # Collect per-condition scores across seeds
        cond_scores: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"kd": [], "cit": [], "hedge": []}
        )
        cond_descs: dict[str, str] = {}
        teacher = student = ""

        for run in runs:
            if not teacher:
                teacher = run.get("teacher", "")
            if not student:
                student = run.get("student", "")
            # Fall back to config label if teacher/student not stored
            if not teacher and "config" in run:
                parts = run["config"].split("→")
                if len(parts) == 2:
                    teacher, student = parts[0].strip(), parts[1].strip()
                else:
                    teacher = run["config"]
            for cond_data in run.get("conditions", []):
                cond = cond_data["condition"]
                m    = cond_data.get("val_metrics", {})
                cond_scores[cond]["kd"].append(m.get("mean_kd_score", 0.0))
                cond_scores[cond]["cit"].append(m.get("citation_fidelity", 0.0))
                cond_scores[cond]["hedge"].append(m.get("hedge_match", 0.0))
                cond_descs[cond] = cond_data.get("description", cond)

        # Build ConditionStats
        conditions: dict[str, ConditionStats] = {}
        for cond, scores in cond_scores.items():
            conditions[cond] = ConditionStats(
                condition     = cond,
                description   = cond_descs.get(cond, ""),
                n_runs        = len(scores["kd"]),
                mean_kd       = round(_mean(scores["kd"]), 4),
                std_kd        = round(_std(scores["kd"]),  4),
                mean_citation = round(_mean(scores["cit"]), 4),
                std_citation  = round(_std(scores["cit"]),  4),
                mean_hedge    = round(_mean(scores["hedge"]), 4),
                std_hedge     = round(_std(scores["hedge"]),  4),
                all_kd_scores = [round(x, 4) for x in scores["kd"]],
                configs_seen  = [config_label],
            )

        # Compute A-B gap
        a_kd = [conditions["A"].mean_kd] if "A" in conditions else [0.0]
        b_kd = [conditions["B"].mean_kd] if "B" in conditions else [0.0]

        # Per-run A-B gaps
        per_run_gaps: list[float] = []
        for run in runs:
            conds_in_run = {c["condition"]: c.get("val_metrics",{}).get("mean_kd_score",0)
                            for c in run.get("conditions", [])}
            if "A" in conds_in_run and "B" in conds_in_run:
                per_run_gaps.append(conds_in_run["A"] - conds_in_run["B"])

        ab_mean = round(_mean(per_run_gaps), 4) if per_run_gaps else 0.0
        ab_std  = round(_std(per_run_gaps),  4) if per_run_gaps else 0.0
        all_ab_gaps.extend(per_run_gaps)

        a_kd_val = conditions.get("A", ConditionStats("A","",0,0,0,0,0,0,0)).mean_kd
        b_kd_val = conditions.get("B", ConditionStats("B","",0,0,0,0,0,0,0)).mean_kd
        c_kd_val = conditions.get("C", ConditionStats("C","",0,0,0,0,0,0,0)).mean_kd
        d_kd_val = conditions.get("D", ConditionStats("D","",0,0,0,0,0,0,0)).mean_kd
        hypothesis = (a_kd_val > b_kd_val and a_kd_val > c_kd_val and a_kd_val > d_kd_val)

        config_summaries.append(ConfigSummary(
            config_label         = config_label,
            teacher              = teacher,
            student              = student,
            n_seeds              = len(runs),
            conditions           = conditions,
            ab_gap_mean          = ab_mean,
            ab_gap_std           = ab_std,
            hypothesis_supported = hypothesis,
        ))

    # Overall A-B gap across all configs
    overall_gap = round(_mean(all_ab_gaps), 4)

    # Interpretation
    if overall_gap > 0.02:
        rec = "STRONG: A−B gap > 0.02. Compelling evidence for the self-knowledge claim. Submit."
    elif overall_gap > 0.01:
        rec = "MODERATE: A−B gap 0.01–0.02. Supports claim; add more seeds/iterations."
    elif overall_gap > 0.005:
        rec = "WEAK: A−B gap 0.005–0.01. Suggestive but in noise range."
    elif overall_gap > 0:
        rec = "MARGINAL: Positive gap but within noise. Try more iterations."
    else:
        rec = "NEGATIVE: External matches/beats self-proposed. Investigate."

    # Build paper table rows
    paper_rows = _build_paper_table(config_summaries)

    return AggregatedResults(
        generated_at     = datetime.now(timezone.utc).isoformat(),
        n_total_runs     = len(raw_results),
        configs          = config_summaries,
        overall_ab_gap   = overall_gap,
        all_ab_gaps      = [round(x, 4) for x in all_ab_gaps],
        recommendation   = rec,
        paper_table_rows = paper_rows,
    )


def _build_paper_table(configs: list[ConfigSummary]) -> list[dict]:
    """Build rows for the paper's Table 1 (pooled across all configs)."""
    pooled: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"kd":[], "cit":[], "hedge":[]})
    descs: dict[str, str] = {}

    for cfg in configs:
        for cond, stats in cfg.conditions.items():
            pooled[cond]["kd"].append(stats.mean_kd)
            pooled[cond]["cit"].append(stats.mean_citation)
            pooled[cond]["hedge"].append(stats.mean_hedge)
            descs[cond] = stats.description

    rows = []
    d_kd = _mean(pooled.get("D", {}).get("kd", [0.0]))
    for cond in ["A", "B", "C", "D", "E", "F"]:
        if cond not in pooled: continue
        m_kd  = _mean(pooled[cond]["kd"])
        s_kd  = _std(pooled[cond]["kd"])
        m_cit = _mean(pooled[cond]["cit"])
        s_cit = _std(pooled[cond]["cit"])
        delta = m_kd - d_kd
        rows.append({
            "condition":    cond,
            "description":  descs.get(cond, ""),
            "kd_mean":      round(m_kd,  4),
            "kd_std":       round(s_kd,  4),
            "kd_str":       f"{m_kd:.3f} ± {s_kd:.3f}" if s_kd > 0 else f"{m_kd:.3f}",
            "delta_vs_d":   round(delta, 4),
            "delta_str":    f"{delta:+.3f}",
            "cit_mean":     round(m_cit, 4),
            "cit_std":      round(s_cit, 4),
            "cit_str":      f"{m_cit:.3f} ± {s_cit:.3f}" if s_cit > 0 else f"{m_cit:.3f}",
        })
    return rows


# ── Reporting ──────────────────────────────────────────────────────────────

def print_full_report(agg: AggregatedResults, verbose: bool = False) -> str:
    lines = []
    lines.append("\n" + "="*70)
    lines.append("KD-SPAR ABLATION — AGGREGATED RESULTS")
    lines.append(f"Generated: {agg.generated_at}  |  Total runs: {agg.n_total_runs}")
    lines.append("="*70)

    for cfg in agg.configs:
        lines.append(f"\nConfig: {cfg.config_label}  ({cfg.n_seeds} seed(s))")
        lines.append(f"  Teacher : {cfg.teacher}")
        lines.append(f"  Student : {cfg.student}")
        lines.append(f"  A−B gap : {cfg.ab_gap_str}")
        lines.append(f"  H0 supported: {'✓' if cfg.hypothesis_supported else '✗'}")
        lines.append("")
        lines.append(f"  {'Cond':<5} {'KD Score':<18} {'Δ vs D':<10} "
                     f"{'Citation':<18} {'Hedge'}")
        lines.append("  " + "-"*62)
        d_kd = cfg.conditions.get("D", ConditionStats("D","",0,0,0,0,0,0,0)).mean_kd
        for cond in ["A", "B", "C", "D", "E", "F"]:
            if cond not in cfg.conditions: continue
            s     = cfg.conditions[cond]
            delta = s.mean_kd - d_kd
            lines.append(
                f"  {cond:<5} {s.kd_str:<18} {delta:+.3f}       "
                f"{s.cit_str:<18} {s.hedge_str}"
            )
        if verbose and "A" in cfg.conditions:
            lines.append(f"\n  All KD scores (Condition A): {cfg.conditions['A'].all_kd_scores}")

    lines.append("\n" + "="*70)
    lines.append("POOLED TABLE (all configs combined)")
    lines.append("="*70)
    lines.append(f"  {'Cond':<5} {'KD Score':<18} {'Δ vs D':<10} "
                 f"{'Citation':<18} Description")
    lines.append("  " + "-"*70)
    for row in agg.paper_table_rows:
        lines.append(
            f"  {row['condition']:<5} {row['kd_str']:<18} {row['delta_str']:<10} "
            f"{row['cit_str']:<18} {row['description'][:30]}"
        )

    lines.append(f"\nOverall A−B gap: {agg.overall_ab_gap:+.4f}")
    lines.append(f"All A−B gaps:   {agg.all_ab_gaps}")
    lines.append(f"\n{agg.recommendation}")

    report = "\n".join(lines)
    print(report)
    return report


# ── Save ───────────────────────────────────────────────────────────────────

def save_aggregated(agg: AggregatedResults, report: str, results_dir: Path = RESULTS_DIR) -> tuple[Path, Path]:
    """Save aggregated results JSON + text summary."""
    # Convert dataclasses to dicts
    def to_dict(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, list):
            return [to_dict(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        return obj

    jp = results_dir / "aggregated_results.json"
    tp = results_dir / "aggregated_results.txt"

    with open(jp, "w") as f:
        json.dump(to_dict(agg), f, indent=2)
    with open(tp, "w") as f:
        f.write(report)

    print(f"\nSaved:")
    print(f"  {jp}")
    print(f"  {tp}")
    print(f"\nNext step — patch the paper with real numbers:")
    print(f"  python experiments/patch_paper.py")
    return jp, tp


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Aggregate KD-SPAR ablation results")
    p.add_argument("--config", help="Filter to files containing this config string")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    raw = load_all_results(config_filter=args.config)
    if not raw:
        sys.exit(1)

    agg    = aggregate(raw)
    report = print_full_report(agg, verbose=args.verbose)
    save_aggregated(agg, report)
