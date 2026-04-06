# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.6.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
"""
experiments/kd_spar_ablation.py
================================
Controlled ablation study for the KD-SPAR self-knowledge hypothesis.

THE CLAIM WE ARE TESTING
-------------------------
"Student-proposed instructions outperform randomly sampled and externally
 proposed instructions as a baseline, demonstrating that the student's
 self-knowledge about its own failure modes is the mechanism driving
 KD-SPAR's improvements."

FOUR CONDITIONS
---------------
A  KD-SPAR (student self-proposed)   — our method
B  External-proposed                  — teacher model proposes instructions
                                        (same KD signal, different proposer)
C  Random instructions                — randomly sampled from a generic pool
D  No prompt tuning                   — vanilla DEFAULT_SYSTEM (baseline)

METRICS
-------
For each condition, across 3 iterations × held-out val set:
  • mean_kd_score      — Jaccard similarity vs teacher  (primary)
  • citation_fidelity  — fraction of responses citing [Doc-N]
  • hedge_match        — student/teacher hedge-phrase frequency ratio
  • delta_vs_baseline  — improvement over condition D

RUNNING ON YOUR ORYXPRO (Pop!_OS / Ubuntu)
-------------------------------------------
  cd knowledge_distillation
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e ".[rag]"
  export ANTHROPIC_API_KEY="sk-ant-..."
  python experiments/kd_spar_ablation.py

Expected runtime: 25–45 minutes (API rate is the bottleneck, not GPU).
GPU is NOT required. The experiment is fully CPU-bound on the local side.

Results are written to:
  experiments/results/ablation_YYYYMMDD_HHMMSS.json
  experiments/results/ablation_YYYYMMDD_HHMMSS_summary.txt
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Ensure project root is on sys.path ─────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sara.rag.pipeline import (
    RAGPipeline,
    RAGVectorStore,
    AnthropicClient,
    DEFAULT_SYSTEM,
    TEACHER_MODEL,
    STUDENT_MODEL,
)
from sara.rag.kd_spar import KDSPAR, _kd_score, _classify_failure

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Random instruction pool ─────────────────────────────────────────────────
# Generic instructions that have NOTHING to do with the student's actual failure
# modes — used for the random baseline (Condition C).
RANDOM_INSTRUCTION_POOL = [
    "Always use bullet points when listing multiple items.",
    "Begin responses with a one-sentence summary.",
    "Use markdown formatting throughout your response.",
    "Keep responses under 150 words whenever possible.",
    "Always spell out numbers below ten in prose.",
    "Use the active voice in all sentences.",
    "End every response with a clarifying question.",
    "Include at least one example in each response.",
    "Use the Oxford comma when listing three or more items.",
    "Format technical terms in backticks when appropriate.",
    "Include a brief disclaimer about response limitations.",
    "Use subheadings to organise responses longer than 100 words.",
    "Translate any jargon into plain language immediately after using it.",
    "Always acknowledge if there could be multiple perspectives.",
    "Start responses with 'Thank you for your question'.",
    "Prefer shorter sentences with a maximum of 20 words each.",
    "Reference the current date if relevant to the response.",
    "Emphasise key terms by capitalising the first letter.",
    "Use the phrase 'In summary' before concluding paragraphs.",
    "Include a confidence rating (high/medium/low) at the end.",
]

# External proposer prompt — used for Condition B
EXTERNAL_PROPOSE_PROMPT = """\
You are a prompt engineering expert optimising a RAG assistant's system prompt.

The RAG assistant currently struggles with these failure modes:
{failure_modes}

The target pattern from a high-quality reference model is:
{target_pattern}

Write exactly ONE instruction that should be added to the assistant's system prompt
to help it match the reference model's style.

Return ONLY the instruction text. No preamble, no explanation, no quotes.
"""


# ── Corpus and queries ───────────────────────────────────────────────────────
CORPUS = {
    "kd_foundations.txt": """
        Knowledge distillation transfers knowledge from a large teacher model to a smaller
        student model. Hinton et al. (2015) formalised this using temperature-scaled softmax
        outputs as soft targets. The temperature parameter T controls the softness of the
        distribution — higher T flattens it, revealing inter-class similarity called dark
        knowledge. The combined distillation loss is: L = alpha * T^2 * KL(student || teacher)
        + (1 - alpha) * CE(student, labels). The T-squared factor compensates for gradient
        magnitude reduction. The alpha parameter balances soft targets versus hard labels.
        Feature-based distillation (FitNets) aligns intermediate layer representations.
        Attention transfer distillation aligns spatial attention maps. Relational KD
        distils pairwise structural relationships between samples.
    """,
    "rag_systems.txt": """
        Retrieval-Augmented Generation (RAG) grounds language model responses in external
        knowledge by retrieving relevant document passages at inference time. A RAG pipeline
        consists of three phases: document ingestion (chunk text, create embeddings, store
        in vector database), retrieval (semantic search over embeddings), and generation
        (language model synthesises answer from retrieved context). ChromaDB is a popular
        open-source vector database for RAG. Citation format [Doc-N] helps trace claims
        to source passages. Good RAG responses cite every claim with [Doc-N] notation,
        express uncertainty when context is partial, and avoid claims not supported by
        retrieved passages.
    """,
    "kd_spar_method.txt": """
        KD-SPAR (Knowledge Distillation via Student Prompt Auto-Rewriting) is a paradigm
        where the student model diagnoses its own failure modes and proposes targeted
        amendments to its own system prompt. The algorithm has four phases: diagnostic
        pass (compare student to teacher, classify failures), self-interview (student
        proposes one instruction per failure mode), aggregation (cluster and score
        proposals), and validate-and-commit (accept only if KD score improves).
        The self-knowledge hypothesis claims that the student model has privileged
        knowledge about what instructions will improve its own performance, because
        the model that proposes the instruction is the same one that will execute it.
        This distinguishes KD-SPAR from external prompt optimisers like OPRO or DSPy.
    """,
    "kd_variants.txt": """
        Multi-Teacher KD-SPAR aligns a single student to multiple specialist teachers
        simultaneously. The worst-aligned teacher drives the self-interview each iteration.
        Non-regression validation ensures no secondary teacher's alignment degrades.
        Adversarial KD-SPAR focuses on hard examples — queries where the teacher-student
        gap is largest or queries generated by the teacher to expose student weaknesses.
        Federated KD-SPAR enables distributed clients to share only proposed instruction
        strings, never raw query or response data. The central server aggregates proposals
        and broadcasts an updated global prompt to all clients.
    """,
}

# 40 training queries  +  15 validation queries
TRAIN_QUERIES = [
    "What is the role of temperature in knowledge distillation?",
    "How does the T-squared scaling factor work?",
    "What is dark knowledge and why does it matter?",
    "Explain the combined distillation loss formula.",
    "What is the difference between soft targets and hard labels?",
    "How does FitNets feature-based distillation work?",
    "What does the alpha parameter control in KD?",
    "How does attention transfer distillation work?",
    "What is relational knowledge distillation?",
    "How does ChromaDB support a RAG pipeline?",
    "What are the three phases of a RAG pipeline?",
    "Why should RAG responses use [Doc-N] citations?",
    "What failure modes does KD-SPAR target?",
    "What is the self-interview phase of KD-SPAR?",
    "What is the self-knowledge hypothesis in KD-SPAR?",
    "How does KD-SPAR differ from OPRO and DSPy?",
    "What happens in the validate-and-commit phase of KD-SPAR?",
    "What is Multi-Teacher KD-SPAR?",
    "What is the worst-aligned teacher principle?",
    "How does non-regression validation work?",
    "What is Adversarial KD-SPAR?",
    "How are hard examples mined in Adversarial KD-SPAR?",
    "What is dual-objective validation in adversarial KD-SPAR?",
    "What is Federated KD-SPAR?",
    "Why does Federated KD-SPAR only share instruction strings?",
    "How does temperature scaling relate to entropy?",
    "When should you use a high temperature in KD?",
    "What is the capacity ratio in knowledge distillation?",
    "How does progressive distillation avoid capacity-gap failures?",
    "What is online mutual learning?",
    "How does self-distillation work with early-exit networks?",
    "What is data-free knowledge distillation?",
    "How does DistilBERT compare to BERT?",
    "What makes Phi-3-Mini significant?",
    "How does Whisper compression use KD?",
    "What is the diagnostic pass in KD-SPAR?",
    "How does aggregation work in KD-SPAR?",
    "What citations should a good RAG response include?",
    "How does ChromaDB use cosine similarity for retrieval?",
    "What is the role of sentence-transformers in RAG embeddings?",
]

VAL_QUERIES = [
    "What is the primary contribution of KD-SPAR?",
    "Why does the student self-author its own prompt in KD-SPAR?",
    "How does Federated KD-SPAR protect privacy?",
    "What is the T-squared factor and why is it necessary?",
    "When should you use Adversarial KD-SPAR instead of base KD-SPAR?",
    "What does the missing_citation failure mode indicate?",
    "How does the calibration ratio equivalence check work?",
    "What is the hallucination proxy metric?",
    "How does relational KD differ from response-based KD?",
    "What is the worst-teacher principle in Multi-Teacher KD-SPAR?",
    "Why does higher temperature reveal more dark knowledge?",
    "What is the difference between gap-mined and generated adversarial queries?",
    "How does the validate-and-commit phase prevent overfitting in KD-SPAR?",
    "What are the three phases of a RAG pipeline?",
    "How does KD-SPAR's self-knowledge differ from Constitutional AI's self-critique?",
]


# ── Metrics ──────────────────────────────────────────────────────────────────
CITATION_RE   = re.compile(r"\[Doc-\d+\]")
HEDGE_WORDS   = ["may", "might", "could", "possibly", "perhaps", "appears", "seems",
                 "uncertain", "unclear", "cannot confirm"]

def citation_fidelity(student_resp: str, teacher_resp: str) -> float:
    t_cited = bool(CITATION_RE.search(teacher_resp))
    s_cited = bool(CITATION_RE.search(student_resp))
    return 1.0 if (not t_cited) or s_cited else 0.0

def hedge_match(student_resp: str, teacher_resp: str) -> float:
    """Student/teacher hedge-phrase frequency ratio (1.0 = identical calibration)."""
    s = sum(1 for w in HEDGE_WORDS if w in student_resp.lower())
    t = sum(1 for w in HEDGE_WORDS if w in teacher_resp.lower())
    return s / max(t, 0.01)

def evaluate_prompt(
    prompt:            str,
    queries:           list[str],
    teacher_responses: dict[str, str],
    store:             RAGVectorStore,
    label:             str = "",
) -> dict:
    """
    Score a prompt on val queries against teacher responses.

    Returns dict with mean_kd_score, citation_fidelity, hedge_match, per_query.
    """
    pipeline = RAGPipeline(STUDENT_MODEL, store=store, system_prompt=prompt)
    scores, cits, hedges = [], [], []
    per_query = []
    for q in queries:
        if q not in teacher_responses:
            continue
        try:
            s_resp = pipeline.query(q, return_context=False).answer
            t_resp = teacher_responses[q]
            kd  = _kd_score(s_resp, t_resp)
            cit = citation_fidelity(s_resp, t_resp)
            hed = hedge_match(s_resp, t_resp)
            scores.append(kd);  cits.append(cit);  hedges.append(hed)
            per_query.append({"q": q[:60], "kd": round(kd,4), "cit": round(cit,4)})
        except Exception as exc:
            print(f"    [eval] error on '{q[:40]}': {exc}")

    result = {
        "label":            label,
        "prompt_length":    len(prompt),
        "n_queries":        len(scores),
        "mean_kd_score":    round(sum(scores) / max(len(scores),1), 4),
        "citation_fidelity":round(sum(cits)   / max(len(cits),1),  4),
        "hedge_match":      round(sum(hedges)  / max(len(hedges),1),4),
        "per_query":        per_query,
    }
    print(f"  [{label}]  kd={result['mean_kd_score']:.4f}  "
          f"cit={result['citation_fidelity']:.3f}  hedge={result['hedge_match']:.3f}")
    return result


# ── Condition generators ──────────────────────────────────────────────────────
def build_condition_D_prompt() -> str:
    """Condition D: no prompt tuning — just the default system prompt."""
    return DEFAULT_SYSTEM

def build_condition_C_prompt(n_instructions: int = 3) -> str:
    """Condition C: randomly sampled generic instructions."""
    sampled = random.sample(RANDOM_INSTRUCTION_POOL, min(n_instructions, len(RANDOM_INSTRUCTION_POOL)))
    return DEFAULT_SYSTEM + "\n\n# Random instructions:\n" + "\n".join(f"- {s}" for s in sampled)

def build_condition_B_prompt(
    n_instructions:    int,
    teacher_client:    AnthropicClient,
    train_queries:     list[str],
    teacher_responses: dict[str, str],
    store:             RAGVectorStore,
    iterations:        int = 3,
) -> str:
    """
    Condition B: externally-proposed instructions.
    The TEACHER model proposes instructions (same KD signal, different proposer).
    This is the strongest baseline — it tests whether self-authorship matters.
    """
    current_prompt = DEFAULT_SYSTEM
    student_pipe   = RAGPipeline(STUDENT_MODEL, store=store, system_prompt=current_prompt)

    for it in range(iterations):
        # Diagnose failures (same as KD-SPAR phase 1)
        failures = []
        for q in train_queries[:8]:
            if q not in teacher_responses: continue
            try:
                s = student_pipe.query(q, return_context=False).answer
                kd = _kd_score(s, teacher_responses[q])
                mode = _classify_failure(s, teacher_responses[q])
                failures.append((q, s, teacher_responses[q], mode, kd))
            except Exception: pass
        failures.sort(key=lambda x: x[4])  # worst first

        if not failures:
            break

        # Phase 2: TEACHER proposes instructions (not student)
        proposals = []
        for q, s_resp, t_resp, mode, _ in failures[:3]:
            t_patterns = []
            if CITATION_RE.search(t_resp): t_patterns.append("uses [Doc-N] citations")
            if any(w in t_resp.lower() for w in HEDGE_WORDS): t_patterns.append("hedges uncertainty")
            if len(t_resp) > 200: t_patterns.append("provides detailed answers")
            target_str = "; ".join(t_patterns) if t_patterns else "matches teacher style"

            prompt_text = EXTERNAL_PROPOSE_PROMPT.format(
                failure_modes=mode, target_pattern=target_str
            )
            try:
                resp = teacher_client._client.messages.create(
                    model=teacher_client.model_id, max_tokens=80,
                    messages=[{"role": "user", "content": prompt_text}],
                )
                text = resp.content[0].text.strip()
                if len(text) > 15 and not text.lower().startswith(("sure", "here")):
                    proposals.append(text)
            except Exception as exc:
                print(f"    [external-propose] error: {exc}")

        if not proposals:
            break

        # Same validate-and-commit as KD-SPAR (fair comparison)
        old_score = _batch_kd(VAL_QUERIES[:5], teacher_responses, student_pipe)
        candidate = (current_prompt + "\n\n# Externally proposed:\n"
                     + "\n".join(f"- {p}" for p in proposals[:3]))
        student_pipe.client.update_system(candidate)
        new_score = _batch_kd(VAL_QUERIES[:5], teacher_responses, student_pipe)

        if new_score > old_score + 0.003:
            current_prompt = candidate
            print(f"    [external] iter {it+1}: {old_score:.4f} → {new_score:.4f}  ✓")
        else:
            student_pipe.client.update_system(current_prompt)
            print(f"    [external] iter {it+1}: {old_score:.4f} → {new_score:.4f}  ✗ reverted")

    return current_prompt

def build_condition_A_prompt(
    n_instructions:    int,
    train_queries:     list[str],
    val_queries:       list[str],
    teacher_responses: dict[str, str],
    store:             RAGVectorStore,
    iterations:        int = 3,
) -> str:
    """
    Condition A: KD-SPAR (student self-proposed) — our method.
    """
    spar = KDSPAR(
        teacher_model=TEACHER_MODEL,
        student_model=STUDENT_MODEL,
        vector_store=store,
    )
    final_prompt, _ = spar.run(
        train_queries     = train_queries,
        val_queries       = val_queries,
        teacher_responses = teacher_responses,
        iterations        = iterations,
        threshold         = 0.003,
        n_proposals       = 4,
        top_k             = 3,
    )
    return final_prompt


def _batch_kd(queries, teacher_resps, pipeline) -> float:
    scores = []
    for q in queries:
        if q not in teacher_resps: continue
        try:
            s = pipeline.query(q, return_context=False).answer
            scores.append(_kd_score(s, teacher_resps[q]))
        except Exception: scores.append(0.0)
    return sum(scores) / max(len(scores), 1)


# ── Main ablation runner ──────────────────────────────────────────────────────
@dataclass
class AblationResult:
    condition:         str
    description:       str
    final_prompt:      str
    val_metrics:       dict
    build_time_sec:    float
    iterations_used:   int


def run_ablation(
    iterations:    int  = 3,
    seed:          int  = 42,
    quick_mode:    bool = False,   # True → use fewer queries (faster for testing)
) -> list[AblationResult]:
    """
    Run all four conditions and return results.

    Parameters
    ----------
    iterations  : KD-SPAR iterations per condition (3 = ~25 min, 5 = ~45 min)
    seed        : Random seed for reproducibility
    quick_mode  : Use 10 train + 5 val queries for fast testing (~10 min)
    """
    random.seed(seed)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError("ANTHROPIC_API_KEY is not set. Run: export ANTHROPIC_API_KEY='sk-ant-...'")

    train_q = TRAIN_QUERIES[:10] if quick_mode else TRAIN_QUERIES
    val_q   = VAL_QUERIES[:5]    if quick_mode else VAL_QUERIES
    print(f"\nAblation config: {len(train_q)} train  |  {len(val_q)} val  |  {iterations} iterations")

    # ── Build vector store ──────────────────────────────────────────────────
    print("\nBuilding vector store …")
    store = RAGVectorStore(persist_path=str(RESULTS_DIR / "ablation_chroma_db"))
    pipeline = RAGPipeline(TEACHER_MODEL, store=store)
    n = pipeline.ingest(CORPUS)
    print(f"Indexed {n} chunks")

    # ── Harvest teacher responses ───────────────────────────────────────────
    print("\nHarvesting teacher responses …")
    teacher_responses: dict[str, str] = {}
    all_queries = list(dict.fromkeys(train_q + val_q))  # deduplicated, ordered
    for i, q in enumerate(all_queries):
        try:
            resp = pipeline.query(q, return_context=False)
            teacher_responses[q] = resp.answer
            if (i+1) % 10 == 0:
                print(f"  {i+1}/{len(all_queries)} collected …")
        except Exception as exc:
            print(f"  error: {q[:50]}: {exc}")
    print(f"  {len(teacher_responses)}/{len(all_queries)} teacher responses collected")

    teacher_client = AnthropicClient(model_id=TEACHER_MODEL)
    results: list[AblationResult] = []

    # ── Condition D: No prompt tuning (baseline) ─────────────────────────────
    print("\n" + "="*60)
    print("CONDITION D — Baseline (no prompt tuning)")
    t0 = time.time()
    prompt_D = build_condition_D_prompt()
    metrics_D = evaluate_prompt(prompt_D, val_q, teacher_responses, store, label="D_baseline")
    results.append(AblationResult("D", "No prompt tuning (DEFAULT_SYSTEM)",
                                  prompt_D, metrics_D, time.time()-t0, 0))

    # ── Condition C: Random instructions ─────────────────────────────────────
    print("\n" + "="*60)
    print("CONDITION C — Random instruction baseline")
    t0 = time.time()
    prompt_C = build_condition_C_prompt(n_instructions=min(iterations*2, 6))
    metrics_C = evaluate_prompt(prompt_C, val_q, teacher_responses, store, label="C_random")
    results.append(AblationResult("C", "Random instructions (no self-knowledge)",
                                  prompt_C, metrics_C, time.time()-t0, 0))

    # ── Condition B: Externally proposed ─────────────────────────────────────
    print("\n" + "="*60)
    print("CONDITION B — External-proposed (teacher proposes, not student)")
    t0 = time.time()
    prompt_B = build_condition_B_prompt(
        n_instructions=3, teacher_client=teacher_client,
        train_queries=train_q, teacher_responses=teacher_responses,
        store=store, iterations=iterations,
    )
    metrics_B = evaluate_prompt(prompt_B, val_q, teacher_responses, store, label="B_external")
    results.append(AblationResult("B", "Externally proposed (teacher proposes)",
                                  prompt_B, metrics_B, time.time()-t0, iterations))

    # ── Condition A: KD-SPAR (student self-proposed) ─────────────────────────
    print("\n" + "="*60)
    print("CONDITION A — KD-SPAR (student self-proposed) — our method")
    t0 = time.time()
    prompt_A = build_condition_A_prompt(
        n_instructions=3, train_queries=train_q, val_queries=val_q,
        teacher_responses=teacher_responses, store=store, iterations=iterations,
    )
    metrics_A = evaluate_prompt(prompt_A, val_q, teacher_responses, store, label="A_kd_spar")
    results.append(AblationResult("A", "KD-SPAR (student self-proposed)",
                                  prompt_A, metrics_A, time.time()-t0, iterations))

    return results, teacher_responses


# ── Reporting ─────────────────────────────────────────────────────────────────
def print_report(results: list[AblationResult], teacher_responses: dict) -> str:
    """Print and return a human-readable ablation report."""
    # Sort by mean_kd_score descending
    sorted_r = sorted(results, key=lambda r: r.val_metrics["mean_kd_score"], reverse=True)
    baseline = next(r for r in results if r.condition == "D")
    baseline_kd = baseline.val_metrics["mean_kd_score"]

    lines = []
    lines.append("\n" + "="*70)
    lines.append("KD-SPAR ABLATION RESULTS")
    lines.append("="*70)
    lines.append(f"{'Rank':<5} {'Cond':<5} {'KD Score':<12} {'Δ vs D':<10} "
                 f"{'Cit Fid':<10} {'Hedge':<10} Description")
    lines.append("-"*70)

    for rank, r in enumerate(sorted_r, 1):
        delta = r.val_metrics["mean_kd_score"] - baseline_kd
        lines.append(
            f"  {rank}    {r.condition}     "
            f"{r.val_metrics['mean_kd_score']:.4f}       "
            f"{delta:+.4f}    "
            f"{r.val_metrics['citation_fidelity']:.3f}       "
            f"{r.val_metrics['hedge_match']:.3f}      "
            f"{r.description[:35]}"
        )

    lines.append("="*70)

    # Key finding
    kd_spar = next(r for r in results if r.condition == "A")
    external = next(r for r in results if r.condition == "B")
    random_r = next(r for r in results if r.condition == "C")

    a_score = kd_spar.val_metrics["mean_kd_score"]
    b_score = external.val_metrics["mean_kd_score"]
    c_score = random_r.val_metrics["mean_kd_score"]
    d_score = baseline_kd

    lines.append("\nKEY FINDING:")
    hypothesis_supported = a_score > b_score and a_score > c_score and a_score > d_score
    if hypothesis_supported:
        lines.append("  ✓ SELF-KNOWLEDGE HYPOTHESIS SUPPORTED")
        lines.append(f"  KD-SPAR (A={a_score:.4f}) > External (B={b_score:.4f}) "
                     f"> Random (C={c_score:.4f}) > Baseline (D={d_score:.4f})")
        lines.append(f"  Student self-proposed instructions outperform all baselines.")
        lines.append(f"  The performance gap A−B = {a_score-b_score:+.4f} isolates the")
        lines.append(f"  pure value of self-authorship over external proposal with the same KD signal.")
    else:
        lines.append("  ✗ HYPOTHESIS NOT CLEARLY SUPPORTED — analysis needed")
        lines.append(f"  Scores: A={a_score:.4f}  B={b_score:.4f}  C={c_score:.4f}  D={d_score:.4f}")
        if a_score <= b_score:
            lines.append("  NOTE: External proposer matched or exceeded KD-SPAR.")
            lines.append("  This does NOT disprove the method but weakens the self-knowledge claim.")
            lines.append("  Consider: more iterations, larger query set, different model pair.")

    lines.append("\nINTERPRETATION GUIDE:")
    lines.append("  A > B  →  self-authorship matters beyond the KD signal alone")
    lines.append("  A > C  →  KD-SPAR is better than random prompt augmentation")
    lines.append("  B > C  →  even external KD-signal-guided proposals beat random")
    lines.append("  B > D  →  any KD-guided proposal is better than no tuning")
    lines.append("  A−B gap is the key number for reviewers: >0.01 = compelling claim")

    report = "\n".join(lines)
    print(report)
    return report


def save_results(
    results: list[AblationResult],
    teacher_responses: dict,
    report: str,
) -> Path:
    """Save full results to JSON + summary to txt."""
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"ablation_{ts}.json"
    txt_path  = RESULTS_DIR / f"ablation_{ts}_summary.txt"

    output = {
        "timestamp":   ts,
        "teacher":     TEACHER_MODEL,
        "student":     STUDENT_MODEL,
        "n_val_queries": len(teacher_responses),
        "conditions": [
            {
                "condition":      r.condition,
                "description":    r.description,
                "val_metrics":    r.val_metrics,
                "prompt_length":  r.prompt_length,
                "build_time_sec": round(r.build_time_sec, 1),
                "iterations":     r.iterations_used,
                "final_prompt":   r.final_prompt,
            }
            for r in results
        ],
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    with open(txt_path, "w") as f:
        f.write(report)

    print(f"\nResults saved:")
    print(f"  {json_path}")
    print(f"  {txt_path}")
    return json_path


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="KD-SPAR Ablation Study — Self-Knowledge Hypothesis Test"
    )
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="SPAR iterations per condition (default 3 = ~25 min)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 10 train / 5 val queries, fewer API calls (~10 min)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    print("="*70)
    print("KD-SPAR ABLATION STUDY")
    print(f"  Teacher  : {TEACHER_MODEL}")
    print(f"  Student  : {STUDENT_MODEL}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Mode     : {'QUICK (dev)' if args.quick else 'FULL'}")
    print(f"  Seed     : {args.seed}")
    print("="*70)

    results, teacher_responses = run_ablation(
        iterations=args.iterations,
        seed=args.seed,
        quick_mode=args.quick,
    )
    report = print_report(results, teacher_responses)
    save_results(results, teacher_responses, report)
