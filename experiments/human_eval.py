# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.8.3
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ajsinha/sara
"""
experiments/human_eval.py
==========================
Human evaluation toolkit for KD-SPAR ablation.

Generates blind evaluation sheets from ablation results and computes
inter-rater agreement (Cohen's κ and Fleiss' κ) from collected ratings.

Usage
-----
    # Generate evaluation sheets (after running ablation):
    python experiments/human_eval.py generate --n-queries 20

    # Score collected ratings:
    python experiments/human_eval.py score --ratings ratings.csv

    # Generate a summary report:
    python experiments/human_eval.py report --ratings ratings.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
EVAL_DIR    = RESULTS_DIR / "human_eval"

DIMENSIONS = [
    ("correctness",  "Is the response factually correct given the context? (1=wrong, 5=perfect)"),
    ("completeness", "Does the response cover all key points from the teacher? (1=missing most, 5=comprehensive)"),
    ("citation",     "Are citations properly placed and formatted? (1=none/wrong, 5=perfect [Doc-N])"),
    ("tone",         "Does the response tone match the teacher's? (1=very different, 5=identical tone)"),
]


def _load_ablation_results() -> list[dict]:
    """Load all ablation result JSON files."""
    results = []
    for jp in sorted(RESULTS_DIR.glob("ablation_ollama_*.json")):
        if "chroma" in jp.name:
            continue
        try:
            data = json.loads(jp.read_text())
            results.append(data)
        except Exception:
            pass
    return results


def _extract_eval_items(results: list[dict], n_per_condition: int = 5,
                        conditions: str = "ABEF") -> list[dict]:
    """Extract query-response pairs for blind evaluation."""
    items_by_cond: dict[str, list] = defaultdict(list)

    for run in results:
        for cond_data in run.get("conditions", []):
            cond = cond_data.get("condition", "")
            if cond not in conditions:
                continue
            prompt = cond_data.get("final_prompt", "")
            for pq in cond_data.get("val_metrics", {}).get("per_query", []):
                items_by_cond[cond].append({
                    "query": pq.get("q", ""),
                    "condition": cond,
                    "kd_score": pq.get("kd", 0),
                })

    # Select top items per condition (most informative = mid-range scores)
    eval_items = []
    for cond in conditions:
        pool = items_by_cond.get(cond, [])
        if not pool:
            continue
        pool.sort(key=lambda x: abs(x["kd_score"] - 0.5))  # closest to mid-range
        selected = pool[:n_per_condition]
        eval_items.extend(selected)

    return eval_items


def generate_sheets(n_queries: int = 20, conditions: str = "ABEF",
                    seed: int = 42) -> Path:
    """Generate blind evaluation CSV sheets."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    results = _load_ablation_results()
    if not results:
        print("No ablation results found. Run the ablation first.")
        return EVAL_DIR

    n_per = max(1, n_queries // len(conditions))
    items = _extract_eval_items(results, n_per, conditions)

    # Shuffle for blind evaluation
    random.seed(seed)
    random.shuffle(items)

    # Assign blind IDs
    for i, item in enumerate(items, 1):
        item["eval_id"] = f"EVAL-{i:03d}"
        item["blind_condition"] = ""  # hidden from raters

    # Write rater template
    template_path = EVAL_DIR / "rater_template.csv"
    with open(template_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["eval_id", "rater_name", "query"]
        header.extend([d[0] for d in DIMENSIONS])
        header.append("free_text_comments")
        writer.writerow(header)

        for item in items:
            row = [item["eval_id"], "", item["query"]]
            row.extend([""] * len(DIMENSIONS))  # blank scores for rater
            row.append("")
            writer.writerow(row)

    # Write answer key (condition mapping — do NOT share with raters)
    key_path = EVAL_DIR / "answer_key.csv"
    with open(key_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eval_id", "condition", "kd_score", "query"])
        for item in items:
            writer.writerow([item["eval_id"], item["condition"],
                           item["kd_score"], item["query"]])

    # Write instructions
    instr_path = EVAL_DIR / "INSTRUCTIONS.md"
    instr_path.write_text(f"""# Human Evaluation Instructions

## Setup
- You will evaluate {len(items)} query-response pairs
- Each pair shows a query and a model's response
- You do NOT know which condition produced the response (blind evaluation)

## Rating Dimensions
Rate each response on a 1-5 scale for:

{chr(10).join(f'- **{d[0].title()}** — {d[1]}' for d in DIMENSIONS)}

## Process
1. Open `rater_template.csv`
2. Enter your name in the `rater_name` column
3. For each row, read the query and response
4. Rate each dimension 1-5
5. Add optional free-text comments
6. Save as `rater_<your_name>.csv`

## Important
- Rate each response independently
- Do not discuss ratings with other raters until all are complete
- If a response is clearly wrong, rate correctness low even if well-written
- Target: ~3 minutes per evaluation = ~{len(items) * 3} minutes total
""")

    print(f"Generated {len(items)} evaluation items:")
    print(f"  Template:     {template_path}")
    print(f"  Answer key:   {key_path}")
    print(f"  Instructions: {instr_path}")
    return EVAL_DIR


def score_ratings(ratings_paths: list[Path]) -> dict:
    """Compute inter-rater agreement from collected rating CSVs."""
    all_ratings: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )  # eval_id -> dimension -> rater -> score

    raters = set()
    for rpath in ratings_paths:
        with open(rpath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rater = row.get("rater_name", rpath.stem)
                raters.add(rater)
                eid = row["eval_id"]
                for dim, _ in DIMENSIONS:
                    try:
                        score = int(row.get(dim, 0))
                        if 1 <= score <= 5:
                            all_ratings[eid][dim][rater] = score
                    except (ValueError, TypeError):
                        pass

    rater_list = sorted(raters)
    n_raters = len(rater_list)
    print(f"\nRaters: {rater_list}")
    print(f"Items:  {len(all_ratings)}")

    # Compute per-dimension agreement
    results = {}
    for dim, desc in DIMENSIONS:
        pairs = []
        for eid, dim_ratings in all_ratings.items():
            scores = dim_ratings.get(dim, {})
            vals = [scores.get(r) for r in rater_list if r in scores]
            if len(vals) >= 2:
                pairs.append(vals)

        if not pairs:
            continue

        # Cohen's κ for 2 raters, Fleiss' κ for 3+
        if n_raters == 2:
            kappa = _cohens_kappa(pairs, rater_list)
        else:
            kappa = _fleiss_kappa(pairs, n_raters)

        # Mean scores per condition (using answer key)
        results[dim] = {"kappa": round(kappa, 3), "n_pairs": len(pairs)}
        print(f"  {dim:15s}  κ = {kappa:.3f}  (n={len(pairs)})")

    # Load answer key for per-condition means
    key_path = EVAL_DIR / "answer_key.csv"
    if key_path.exists():
        cond_scores: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        key_map = {}
        with open(key_path) as f:
            for row in csv.DictReader(f):
                key_map[row["eval_id"]] = row["condition"]

        for eid, dim_ratings in all_ratings.items():
            cond = key_map.get(eid, "?")
            for dim, _ in DIMENSIONS:
                scores = list(dim_ratings.get(dim, {}).values())
                if scores:
                    cond_scores[cond][dim].extend(scores)

        print("\nPer-condition mean human scores:")
        print(f"  {'Cond':<6}" + "".join(f"{d[0]:>14}" for d in DIMENSIONS))
        for cond in sorted(cond_scores):
            line = f"  {cond:<6}"
            for dim, _ in DIMENSIONS:
                vals = cond_scores[cond].get(dim, [])
                m = sum(vals) / len(vals) if vals else 0
                line += f"{m:>14.2f}"
            print(line)

    return results


def _cohens_kappa(pairs, raters):
    """Cohen's κ for two raters."""
    if len(raters) < 2:
        return 0.0
    r1, r2 = raters[0], raters[1]
    agree, total = 0, 0
    freq = defaultdict(int)
    freq1 = defaultdict(int)
    freq2 = defaultdict(int)
    for vals in pairs:
        if len(vals) >= 2:
            a, b = vals[0], vals[1]
            if a == b:
                agree += 1
            freq1[a] += 1
            freq2[b] += 1
            total += 1
    if total == 0:
        return 0.0
    po = agree / total
    pe = sum(freq1.get(k, 0) * freq2.get(k, 0) for k in range(1, 6)) / (total ** 2)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def _fleiss_kappa(pairs, n_raters):
    """Fleiss' κ for n raters."""
    n_items = len(pairs)
    categories = range(1, 6)  # 1-5 scale

    # Count ratings per category per item
    n_ij = []
    for vals in pairs:
        counts = {k: 0 for k in categories}
        for v in vals:
            if v in counts:
                counts[v] += 1
        n_ij.append(counts)

    n = n_raters
    N = n_items

    # P_i for each item
    P_i = []
    for counts in n_ij:
        s = sum(c * c for c in counts.values()) - n
        P_i.append(s / (n * (n - 1)) if n > 1 else 0)

    P_bar = sum(P_i) / N if N > 0 else 0

    # P_j for each category
    p_j = {}
    for j in categories:
        total_j = sum(counts[j] for counts in n_ij)
        p_j[j] = total_j / (N * n) if (N * n) > 0 else 0

    P_e = sum(p ** 2 for p in p_j.values())

    if P_e >= 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Human evaluation for KD-SPAR")
    sub = p.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate blind evaluation sheets")
    gen.add_argument("--n-queries", type=int, default=20)
    gen.add_argument("--conditions", type=str, default="ABEF")
    gen.add_argument("--seed", type=int, default=42)

    sc = sub.add_parser("score", help="Score collected ratings")
    sc.add_argument("--ratings", nargs="+", type=Path, required=True)

    args = p.parse_args()

    if args.command == "generate":
        generate_sheets(args.n_queries, args.conditions, args.seed)
    elif args.command == "score":
        score_ratings(args.ratings)
    else:
        p.print_help()
