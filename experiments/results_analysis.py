# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.4.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
"""
experiments/results_analysis.py
=================================
Load and compare one or more ablation run JSON files.

Usage:
    # Analyse the most recent run
    python experiments/results_analysis.py

    # Compare two runs (e.g. with different iteration counts)
    python experiments/results_analysis.py \\
        experiments/results/ablation_20250415_143200.json \\
        experiments/results/ablation_20250415_162300.json

Output:
    • Per-condition score table (console)
    • A vs B gap analysis (the key publication metric)
    • Per-query breakdown (optional --verbose)
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def load_latest() -> dict:
    files = sorted(glob.glob(str(Path(__file__).parent / "results" / "ablation_*.json")))
    if not files:
        print("No ablation results found. Run kd_spar_ablation.py first.")
        sys.exit(1)
    with open(files[-1]) as f:
        return json.load(f), files[-1]


def load_file(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def analyse(data: dict, label: str = "", verbose: bool = False) -> None:
    print(f"\n{'='*65}")
    if label:
        print(f"Run: {label}")
    print(f"Timestamp : {data.get('timestamp', 'unknown')}")
    teacher = data.get('teacher', '')
    student = data.get('student', '')
    config  = data.get('config', '')
    if teacher and student:
        print(f"Teacher   : {teacher}")
        print(f"Student   : {student}")
    elif config:
        print(f"Config    : {config}")
    print(f"{'='*65}")

    conditions = {c["condition"]: c for c in data["conditions"]}

    # Table header
    print(f"\n{'Cond':<6} {'KD Score':<12} {'Δ vs D':<10} {'Cit Fid':<10} "
          f"{'Hedge':<10} {'Time(s)':<10} Description")
    print("-"*65)

    d_kd = conditions.get("D", {}).get("val_metrics", {}).get("mean_kd_score", 0.0)
    for cond in sorted(conditions.keys()):
        c   = conditions[cond]
        m   = c["val_metrics"]
        kd  = m.get("mean_kd_score", 0)
        cit = m.get("citation_fidelity", 0)
        hed = m.get("hedge_match", 0)
        t   = c.get("build_time_sec", c.get("build_time_s", 0))
        d   = kd - d_kd
        print(f"  {cond}      {kd:.4f}       {d:+.4f}    {cit:.3f}       "
              f"{hed:.3f}       {t:>5.0f}s    {c['description'][:30]}")

    # Key metric: A vs B gap
    if "A" in conditions and "B" in conditions:
        a_kd = conditions["A"]["val_metrics"].get("mean_kd_score", 0)
        b_kd = conditions["B"]["val_metrics"].get("mean_kd_score", 0)
        gap  = a_kd - b_kd
        print(f"\nKEY METRIC — A vs B gap (self-authorship value):")
        print(f"  A (KD-SPAR)  = {a_kd:.4f}")
        print(f"  B (External) = {b_kd:.4f}")
        print(f"  Gap A − B    = {gap:+.4f}")
        if gap > 0.02:
            print("  → STRONG: gap >0.02 is a compelling claim for reviewers")
        elif gap > 0.01:
            print("  → MODERATE: gap >0.01 supports the self-knowledge claim")
        elif gap > 0.005:
            print("  → WEAK: gap >0.005 is suggestive but reviewers will want more data")
        elif gap > 0:
            print("  → MARGINAL: positive gap but likely within noise — need more iterations")
        else:
            print("  → NEGATIVE: external beats self-proposed — investigate failure modes")

    # Per-query breakdown
    if verbose and "A" in conditions:
        print(f"\nPer-query breakdown (Condition A):")
        for pq in conditions["A"]["val_metrics"].get("per_query", []):
            print(f"  [{pq['kd']:.3f} / cit={pq['cit']:.0f}]  {pq['q']}")

    # Final prompts diff
    if "A" in conditions and "D" in conditions:
        d_prompt = conditions["D"]["final_prompt"]
        a_prompt = conditions["A"]["final_prompt"]
        added = a_prompt[len(d_prompt):].strip()
        if added:
            print(f"\nInstructions added by KD-SPAR (Condition A beyond D):")
            for line in added.split("\n"):
                line = line.strip()
                if line:
                    print(f"  {line}")


def compare(data_list: list[tuple[dict, str]]) -> None:
    """Compare KD scores across multiple runs."""
    print(f"\n{'='*65}")
    print("CROSS-RUN COMPARISON")
    print(f"{'='*65}")
    print(f"{'Run':<30} {'A (KD-SPAR)':<14} {'B (External)':<14} {'A-B Gap'}")
    print("-"*65)
    for data, path in data_list:
        conds = {c["condition"]: c["val_metrics"]["mean_kd_score"]
                 for c in data["conditions"]}
        a = conds.get("A", 0); b = conds.get("B", 0)
        run_label = Path(path).stem[:28]
        print(f"  {run_label:<28} {a:.4f}         {b:.4f}         {a-b:+.4f}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*", help="Path(s) to ablation JSON files")
    p.add_argument("--verbose", "-v", action="store_true", help="Per-query breakdown")
    args = p.parse_args()

    if not args.files:
        data, path = load_latest()
        analyse(data, label=Path(path).name, verbose=args.verbose)
    elif len(args.files) == 1:
        data = load_file(args.files[0])
        analyse(data, label=Path(args.files[0]).name, verbose=args.verbose)
    else:
        data_list = [(load_file(f), f) for f in args.files]
        for data, path in data_list:
            analyse(data, label=Path(path).name, verbose=args.verbose)
        compare(data_list)
