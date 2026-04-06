# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.6.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
from __future__ import annotations
"""
sara.rag.evaluation
=================
Behavioural equivalence test suite for RAG model migration.

Run :func:`run_equivalence_suite` on a list of RAGTrace objects after both
teacher and student responses have been populated to get a pass/fail report
across five dimensions:

    1. Citation fidelity      — student cites passages when teacher did
    2. Semantic similarity    — BERTScore F1 (or Jaccard fallback)
    3. Format preservation    — structured responses remain parseable
    4. Calibration ratio      — hedging-phrase frequency is preserved
    5. Hallucination proxy    — citation-when-cited rate
"""


import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sara.rag.migration import RAGTrace

# ── Signals ───────────────────────────────────────────────────────────────────
CITATION_RE   = re.compile(r"\[Doc-\d+\]")
HEDGE_PHRASES = [
    "may", "might", "could", "possibly", "perhaps",
    "it appears", "it seems", "i'm not certain",
    "i cannot confirm", "unclear from the provided",
]


# ── Report dataclass ──────────────────────────────────────────────────────────

@dataclass
class EquivalenceReport:
    """
    Full behavioural equivalence test results.

    Attributes
    ----------
    citation_fidelity   : Fraction of responses that cite [Doc-N] when teacher did
    mean_kd_score       : Mean semantic similarity (BERTScore F1 or Jaccard)
    format_pass_rate    : Fraction of structured responses that parse as JSON
    calibration_ratio   : student_hedge_rate / teacher_hedge_rate  (1.0 = identical)
    hallucination_proxy : 1 − citation-when-cited rate  (lower = better)
    pass_all            : True when all checks pass their thresholds
    details             : Supplementary numeric details
    """
    citation_fidelity:   float
    mean_kd_score:       float
    format_pass_rate:    float
    calibration_ratio:   float
    hallucination_proxy: float
    pass_all:            bool
    details:             dict = field(default_factory=dict)

    # Default thresholds
    THRESHOLDS = {
        "citation_fidelity":   0.90,
        "mean_kd_score":       0.85,
        "format_pass_rate":    1.00,
        "calibration_lo":      0.80,
        "calibration_hi":      1.20,
        "hallucination_proxy": 0.12,
    }

    def print(self) -> None:
        """Print a formatted pass/fail report."""
        th = self.THRESHOLDS
        ok = lambda v, lo, hi=None: (
            "✓" if (hi is None and v >= lo) else
            "✓" if (hi is not None and lo <= v <= hi) else "✗"
        )
        print("\n" + "="*60)
        print("EQUIVALENCE TEST REPORT")
        print("="*60)
        print(f"  Citation fidelity   {self.citation_fidelity:.3f}   "
              f"{ok(self.citation_fidelity, th['citation_fidelity'])}  "
              f"(≥{th['citation_fidelity']})")
        print(f"  Mean KD score       {self.mean_kd_score:.3f}   "
              f"{ok(self.mean_kd_score, th['mean_kd_score'])}  "
              f"(≥{th['mean_kd_score']})")
        print(f"  Format pass rate    {self.format_pass_rate:.3f}   "
              f"{ok(self.format_pass_rate, th['format_pass_rate'])}  "
              f"(={th['format_pass_rate']:.0f})")
        print(f"  Calibration ratio   {self.calibration_ratio:.3f}   "
              f"{ok(self.calibration_ratio, th['calibration_lo'], th['calibration_hi'])}  "
              f"({th['calibration_lo']}–{th['calibration_hi']})")
        print(f"  Hallucination proxy {self.hallucination_proxy:.3f}   "
              f"{ok(1 - self.hallucination_proxy, 1 - th['hallucination_proxy'])}  "
              f"(≤{th['hallucination_proxy']})")
        print("="*60)
        status = "PROMOTE TO PRODUCTION  ✓" if self.pass_all else "NEEDS REVIEW  ✗"
        print(f"  OVERALL: {status}")
        print("="*60)

    def to_dict(self) -> dict:
        return {
            "citation_fidelity":   self.citation_fidelity,
            "mean_kd_score":       self.mean_kd_score,
            "format_pass_rate":    self.format_pass_rate,
            "calibration_ratio":   self.calibration_ratio,
            "hallucination_proxy": self.hallucination_proxy,
            "pass_all":            self.pass_all,
            **self.details,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_citation(text: str) -> bool:
    return bool(CITATION_RE.search(text))


def _hedge_count(text: str) -> int:
    return sum(1 for p in HEDGE_PHRASES if p in text.lower())


def _is_json(text: str) -> bool:
    text = text.strip()
    if not (text.startswith("{") or text.startswith("[")):
        return False
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(len(sa | sb), 1)


# ── Main suite ────────────────────────────────────────────────────────────────

def run_equivalence_suite(
    traces:     list["RAGTrace"],
    thresholds: dict | None = None,
) -> EquivalenceReport:
    """
    Run the full five-check behavioural equivalence suite.

    Parameters
    ----------
    traces     : RAGTrace list with both teacher_response and student_response set
    thresholds : Override default pass/fail thresholds (merged with defaults)

    Returns
    -------
    EquivalenceReport

    Raises
    ------
    ValueError if no valid traces (both responses populated) are found
    """
    th = {**EquivalenceReport.THRESHOLDS, **(thresholds or {})}

    valid = [t for t in traces
             if t.teacher_response and t.student_response]
    if not valid:
        raise ValueError("No traces with both teacher and student responses found.")

    n = len(valid)

    # 1. Citation fidelity
    teacher_cited = [t for t in valid if _has_citation(t.teacher_response)]
    if teacher_cited:
        s_cited_when_t = sum(
            1 for t in teacher_cited if _has_citation(t.student_response)
        )
        citation_fidelity = s_cited_when_t / len(teacher_cited)
    else:
        citation_fidelity = 1.0   # teacher never cited; trivially satisfied

    # 2. Semantic similarity  (BERTScore → Jaccard fallback)
    try:
        from bert_score import score as bscore  # type: ignore
        _, _, F = bscore(
            [t.student_response for t in valid],
            [t.teacher_response  for t in valid],
            lang="en", verbose=False,
        )
        mean_kd = float(F.mean())
    except ImportError:
        mean_kd = sum(
            _jaccard(t.student_response, t.teacher_response) for t in valid
        ) / n

    # 3. Format preservation
    structured = [t for t in valid if _is_json(t.teacher_response)]
    if structured:
        parse_ok  = sum(1 for t in structured if _is_json(t.student_response))
        fmt_pass  = parse_ok / len(structured)
    else:
        fmt_pass  = 1.0

    # 4. Calibration ratio
    t_hedge = sum(_hedge_count(t.teacher_response) > 0 for t in valid) / n
    s_hedge = sum(_hedge_count(t.student_response) > 0 for t in valid) / n
    calib   = s_hedge / max(t_hedge, 0.01)

    # 5. Hallucination proxy  (same computation as citation_fidelity, kept separate)
    hallu = 1.0 - citation_fidelity

    pass_all = (
        citation_fidelity >= th["citation_fidelity"]
        and mean_kd        >= th["mean_kd_score"]
        and fmt_pass       >= th["format_pass_rate"]
        and th["calibration_lo"] <= calib <= th["calibration_hi"]
        and hallu          <= th["hallucination_proxy"]
    )

    return EquivalenceReport(
        citation_fidelity   = round(citation_fidelity, 4),
        mean_kd_score       = round(mean_kd, 4),
        format_pass_rate    = round(fmt_pass, 4),
        calibration_ratio   = round(calib, 4),
        hallucination_proxy = round(hallu, 4),
        pass_all            = pass_all,
        details             = {
            "n_valid":     n,
            "t_hedge_rate": round(t_hedge, 4),
            "s_hedge_rate": round(s_hedge, 4),
        },
    )
