# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.8.3
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ajsinha/sara
"""
Example 10 — Enhanced KD-SPAR: Seven Algorithmic Improvements
=============================================================

Demonstrates all seven enhancements over base KD-SPAR:
1. Hybrid proposer (teacher diagnoses, student proposes)
2. BERTScore semantic scoring (replaces Jaccard)
3. Contrastive good/bad pair interview
4. Warm-start from external proposals
5. Increased iterations (5)
6. Soft probabilistic commit gate
7. Teacher-guided interview with actual teacher text

Requires: Ollama with llama3.1:8b and llama3.2:3b.
"""

from sara.rag.pipeline import RAGVectorStore
from sara.rag.ollama_pipeline import OllamaRAGPipeline
from sara.rag.kd_spar_enhanced import EnhancedKDSPAR, EnhancedConfig

TEACHER = "llama3.1:8b"
STUDENT = "llama3.2:3b"

CORPUS = {
    "kd.txt": (
        "Knowledge distillation (KD) transfers knowledge from a large teacher "
        "to a smaller student. Hinton et al. (2015) used temperature-scaled "
        "softmax. Loss = alpha * T^2 * KL(student || teacher) + (1-alpha) * CE."
    ),
    "spar.txt": (
        "KD-SPAR lets the student diagnose failures and rewrite its own prompt. "
        "Four phases: diagnose, self-interview, aggregate, validate-and-commit."
    ),
    "enhanced.txt": (
        "Enhanced KD-SPAR adds: BERTScore scoring, hybrid teacher-diagnosis, "
        "contrastive interviews, warm-start, soft commit gate, and more iterations."
    ),
}

TRAIN = [
    "What is knowledge distillation?",
    "Explain the KD-SPAR algorithm.",
    "What are the seven enhancements in Enhanced KD-SPAR?",
    "Why does the student self-author its own prompt?",
]

VAL = [
    "Compare KD-SPAR with traditional prompt optimisation.",
    "What is BERTScore and why does it help?",
    "How does the warm-start work?",
]


def main():
    print("=" * 60)
    print("  Enhanced KD-SPAR — All Seven Improvements")
    print("=" * 60)

    store = RAGVectorStore()
    teacher_pipe = OllamaRAGPipeline(
        TEACHER, store=store, auto_pull=True, temperature=0.0,
    )
    teacher_pipe.ingest(CORPUS)

    print("\nHarvesting teacher responses...")
    t_resps = {}
    for q in list(dict.fromkeys(TRAIN + VAL)):
        try:
            t_resps[q] = teacher_pipe.query(q, return_context=False).answer
        except Exception as e:
            print(f"  Warning: {e}")
    print(f"  {len(t_resps)} responses collected.")

    cfg = EnhancedConfig(
        use_bert_score=True,
        use_hybrid_proposer=True,
        use_contrastive=True,
        warm_start_from_b=True,
        iterations=3,        # shorter for demo
        soft_gate=True,
        teacher_guided=True,
    )

    print(f"\nConfig: {cfg}")
    print("\nRunning Enhanced KD-SPAR...")

    enhanced = EnhancedKDSPAR(
        teacher_model=TEACHER, student_model=STUDENT,
        vector_store=store, config=cfg,
    )
    final_prompt, history = enhanced.run(
        train_queries=TRAIN, val_queries=VAL,
        teacher_responses=t_resps,
    )

    print("\n" + "=" * 60)
    accepted = sum(1 for s in history if s.accepted)
    print(f"  Iterations: {len(history)}   Accepted: {accepted}")
    if history:
        print(f"  Score: {history[0].score_before:.4f} → {history[-1].score_after:.4f}")
    print(f"\nFinal prompt ({len(final_prompt)} chars):")
    print(f"  {final_prompt[:200]}...")


if __name__ == "__main__":
    main()
