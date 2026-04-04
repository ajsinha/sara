# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.1.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
"""
examples/04_kd_spar.py
=======================
Base KD-SPAR — student rewrites its own system prompt.

Backend is read from configs/backend.yaml (default: Ollama/FOSS).
Override:  export SARA_BACKEND=anthropic  ANTHROPIC_API_KEY=sk-ant-...

Run:
    python examples/04_kd_spar.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sara.rag.pipeline import RAGVectorStore
from sara.rag.backend import get_pipeline, get_spar, cfg, describe
from sara.core.progress import SaraLogger

CORPUS = {
    "kd_basics.txt": """
        Knowledge distillation transfers knowledge from a large teacher model to a
        smaller student model using soft probability targets. The temperature parameter
        controls how soft the teacher's output distribution is. Higher temperature reveals
        inter-class similarity called dark knowledge. The combined loss weights KL
        divergence from soft targets against cross-entropy from hard labels.
    """,
    "rag_basics.txt": """
        Retrieval-Augmented Generation grounds responses in retrieved passages.
        A good RAG assistant cites every claim with [Doc-N] inline notation.
        ChromaDB stores document embeddings for semantic retrieval.
        Uncertainty should be expressed explicitly when context is partial.
    """,
}

TRAIN_QUERIES = [
    "What is temperature scaling in knowledge distillation?",
    "How does the T-squared factor work?",
    "What is dark knowledge?",
    "Why should RAG responses include [Doc-N] citations?",
    "How does ChromaDB support retrieval?",
    "What is the alpha parameter in KD?",
    "What are soft targets vs hard labels?",
    "How does the KD loss formula work?",
]

VAL_QUERIES = [
    "What is dark knowledge and why does it matter?",
    "How should a RAG assistant handle uncertainty?",
    "What makes soft targets more informative than hard labels?",
]


def main():
    log = SaraLogger("KD-SPAR")
    log.banner(
        "Sara — Base KD-SPAR Example",
        f"Teacher : {cfg['teacher_model']}",
        f"Student : {cfg['student_model']}",
        f"Backend : {cfg['backend']}",
    )
    log.info(describe())

    log.section("Vector store + corpus ingestion")
    store        = RAGVectorStore(persist_path="./example_04_db")
    teacher_pipe = get_pipeline("teacher", store=store)
    log.step("Ingesting corpus", total=len(CORPUS))
    n = teacher_pipe.ingest(CORPUS)
    log.tick(len(CORPUS))
    log.done(f"Indexed {n} chunks")

    log.section("Harvesting teacher responses")
    all_q = TRAIN_QUERIES + VAL_QUERIES
    log.step(f"Querying teacher for {len(all_q)} responses", total=len(all_q))
    log.start_heartbeat(interval=25, message="Waiting for model response…")
    teacher_responses = {}
    for i, q in enumerate(all_q, 1):
        try:
            teacher_responses[q] = teacher_pipe.query(q, return_context=False).answer
        except Exception as e:
            log.warn(f"  q{i} failed: {e}")
        log.tick(i)
    log.stop_heartbeat()
    log.done(f"{len(teacher_responses)}/{len(all_q)} responses collected")

    log.section("KD-SPAR loop")
    # OllamaKDSPAR.run() / KDSPAR.run() has full phase-level logging built in
    spar = get_spar(store=store)
    final_prompt, history = spar.run(
        train_queries=TRAIN_QUERIES,
        val_queries=VAL_QUERIES,
        teacher_responses=teacher_responses,
        iterations=4,
        threshold=0.002,
        n_proposals=3,
        top_k=2,
    )

    accepted = sum(1 for s in history if s.accepted)
    log.section("Results")
    log.metric("Iterations",  str(len(history)))
    log.metric("Accepted",    f"{accepted}/{len(history)}")
    if history:
        log.metric("Score",
                   f"{history[0].score_before:.4f} → {history[-1].score_after:.4f}",
                   f"Δ={history[-1].score_after - history[0].score_before:+.4f}")

    log.info("Optimised student prompt:")
    log.info("-" * 55)
    for line in final_prompt.split("\n")[:12]:
        log.info(f"  {line}")
    if final_prompt.count("\n") > 12:
        log.info("  … (truncated)")
    log.summary()


if __name__ == "__main__":
    main()
