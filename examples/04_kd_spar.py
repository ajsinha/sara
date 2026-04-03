"""
examples/04_kd_spar.py
=======================
Run the KD-SPAR student prompt auto-rewriting loop.

Teacher : claude-3-5-sonnet-20241022
Student : claude-sonnet-4-5-20250929

Run:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python examples/04_kd_spar.py

Requirements:
    pip install anthropic chromadb sentence-transformers
"""

import json
import os
from sara.rag.pipeline import RAGPipeline, RAGVectorStore, TEACHER_MODEL, STUDENT_MODEL
from sara.rag.kd_spar import KDSPAR

CORPUS = {
    "knowledge_distillation.txt": """
        Knowledge distillation (KD) transfers knowledge from a large teacher model
        to a smaller student model. The teacher's output probability distributions,
        produced with temperature scaling, serve as rich training targets for the
        student. The student minimises KL divergence from these soft targets.
        Key hyperparameters: temperature T (controls distribution softness) and
        alpha (balances KD loss vs. cross-entropy). Higher T reveals more
        inter-class similarity (dark knowledge). T-squared scaling compensates
        for gradient magnitude reduction.
    """,
    "kd_rag.txt": """
        In RAG systems, knowledge distillation can transfer the teacher model's
        citation style, hedging behaviour, and reasoning patterns to a cheaper
        student model. The KD-SPAR algorithm lets the student diagnose its own
        failure modes and propose prompt amendments. This avoids expensive
        fine-tuning while achieving teacher-aligned behaviour.
        ChromaDB stores document embeddings for semantic retrieval.
        Citation format [Doc-N] helps trace answers to source passages.
    """,
}


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY before running.")
        return

    print("Setting up vector store …")
    store         = RAGVectorStore(persist_path="./spar_demo_db")
    teacher_pipe  = RAGPipeline(TEACHER_MODEL, store=store)
    teacher_pipe.ingest(CORPUS)

    TRAIN_QUERIES = [
        "What is temperature scaling in knowledge distillation?",
        "How does KD transfer citation behaviour in RAG?",
        "What is the T-squared scaling factor for?",
        "What failure modes does KD-SPAR address?",
        "How does ChromaDB support retrieval?",
        "What is alpha in the distillation loss?",
    ]
    VAL_QUERIES = [
        "What is dark knowledge in KD?",
        "How does KD-SPAR improve student prompts?",
    ]

    print("\nCollecting teacher reference responses …")
    teacher_responses: dict[str, str] = {}
    for q in TRAIN_QUERIES + VAL_QUERIES:
        resp = teacher_pipe.query(q, return_context=False)
        teacher_responses[q] = resp.answer
        print(f"  ✓  {q[:55]}…")

    print(f"\nStarting KD-SPAR  (teacher={TEACHER_MODEL[:30]}…)")
    spar = KDSPAR(
        teacher_model=TEACHER_MODEL,
        student_model=STUDENT_MODEL,
        vector_store=store,
    )

    final_prompt, history = spar.run(
        train_queries     = TRAIN_QUERIES,
        val_queries       = VAL_QUERIES,
        teacher_responses = teacher_responses,
        iterations        = 5,
        threshold         = 0.002,
        n_proposals       = 3,
        top_k             = 2,
        log_path          = "./spar_log.jsonl",
    )

    accepted = sum(1 for s in history if s.accepted)
    print(f"\nKD-SPAR complete  |  {accepted}/{len(history)} iterations accepted")
    print(f"\nFinal optimised prompt:\n{'-'*50}")
    print(final_prompt)
    print(f"{'-'*50}")

    # Save
    output = {
        "final_prompt": final_prompt,
        "history": [
            {"it": s.iteration, "delta": s.delta, "accepted": s.accepted,
             "selected": s.selected}
            for s in history
        ],
    }
    with open("./spar_result.json", "w") as fh:
        json.dump(output, fh, indent=2)
    print("\nResults saved to ./spar_result.json")


if __name__ == "__main__":
    main()
