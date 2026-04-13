# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.8.3
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ajsinha/sara
"""
Example 09 — MetaKDSPAR: Metaprompting-Enhanced Prompt Distillation
===================================================================

Demonstrates the conductor + specialist architecture where four
specialist perspectives (citation, calibration, completeness, format)
independently diagnose student failures and propose domain-specific fixes.

Requires: Ollama running with llama3.1:8b (teacher) and llama3.2:3b (student).
"""

from sara.rag.pipeline import RAGVectorStore
from sara.rag.ollama_pipeline import OllamaRAGPipeline
from sara.rag.kd_spar_meta import MetaKDSPAR, SPECIALISTS

# ── Configuration ──────────────────────────────────────────────────────────
TEACHER_MODEL = "llama3.1:8b"
STUDENT_MODEL = "llama3.2:3b"

CORPUS = {
    "kd_overview.txt": (
        "Knowledge distillation (KD) transfers knowledge from a large teacher model "
        "to a smaller student model. Hinton et al. (2015) formalised this using "
        "temperature-scaled softmax. The combined loss is: "
        "L = alpha * T^2 * KL(student || teacher) + (1-alpha) * CE(student, labels)."
    ),
    "kd_spar.txt": (
        "KD-SPAR (Knowledge Distillation via Student Prompt Auto-Rewriting) lets the "
        "student model diagnose its own failure modes and rewrite its system prompt. "
        "The four-phase algorithm: diagnose, self-interview, aggregate, validate."
    ),
    "metaprompting.txt": (
        "Metaprompting (Suzgun & Kalai, 2024) uses a conductor + specialist architecture. "
        "A conductor prompt decomposes tasks, delegates to expert personas, and synthesises "
        "outputs. MetaKDSPAR integrates this into KD-SPAR's diagnostic loop."
    ),
}

TRAIN_QUERIES = [
    "What is knowledge distillation and how does it work?",
    "Explain the KD-SPAR algorithm and its four phases.",
    "How does metaprompting improve KD-SPAR?",
    "What is the temperature parameter in knowledge distillation?",
]

VAL_QUERIES = [
    "Compare KD-SPAR with traditional prompt optimisation methods.",
    "What is the self-knowledge hypothesis?",
    "How does the validate-and-commit gate work?",
]


def main():
    print("=" * 60)
    print("  MetaKDSPAR Example — Conductor + Specialist Architecture")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────────────────────────
    store = RAGVectorStore()
    teacher_pipe = OllamaRAGPipeline(
        TEACHER_MODEL, store=store, auto_pull=True, temperature=0.0,
    )
    teacher_pipe.ingest(CORPUS)

    # Harvest teacher responses
    print("\nHarvesting teacher responses...")
    teacher_responses = {}
    all_queries = list(dict.fromkeys(TRAIN_QUERIES + VAL_QUERIES))
    for q in all_queries:
        try:
            teacher_responses[q] = teacher_pipe.query(q, return_context=False).answer
        except Exception as exc:
            print(f"  Warning: {exc}")

    print(f"  {len(teacher_responses)} teacher responses collected.")

    # ── Show specialists ───────────────────────────────────────────────────
    print(f"\nSpecialists ({len(SPECIALISTS)}):")
    for s in SPECIALISTS:
        print(f"  • {s.name}: {s.focus}")

    # ── Run MetaKDSPAR ─────────────────────────────────────────────────────
    print("\nRunning MetaKDSPAR...")
    meta = MetaKDSPAR(
        student_model=STUDENT_MODEL,
        vector_store=store,
        temperature=0.3,
    )
    final_prompt, history = meta.run(
        train_queries=TRAIN_QUERIES,
        val_queries=VAL_QUERIES,
        teacher_responses=teacher_responses,
        iterations=2,
        top_k_diag=3,
        top_k_instr=3,
    )

    # ── Results ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    accepted = sum(1 for s in history if s.accepted)
    print(f"  Iterations: {len(history)}   Accepted: {accepted}")
    if history:
        print(f"  Score: {history[0].score_before:.4f} → {history[-1].score_after:.4f}")
        for it in history:
            print(f"    it={it.iteration}  Δ={it.delta:+.4f}  "
                  f"{'✓' if it.accepted else '✗'}  "
                  f"instructions: {len(it.selected)}")

    print("\nFinal prompt:")
    print(f"  {final_prompt[:200]}...")
    print("\nDone.")


if __name__ == "__main__":
    main()
