# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.1.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
"""
examples/08_ollama_kd_spar.py
==============================
KD-SPAR using local Ollama models — no API key, no cost, works offline.

Teacher : llama3.1:8b   (or any larger model you have pulled)
Student : llama3.2:3b   (or any smaller model)

Run:
    # Full KD-SPAR run (4 iterations, ~15 min)
    python examples/08_ollama_kd_spar.py

    # Quick sanity check only (~1 min, tests Ollama connectivity)
    python examples/08_ollama_kd_spar.py --sanity-only

    # Custom models
    python examples/08_ollama_kd_spar.py --teacher qwen2.5:7b --student llama3.2:3b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sara.rag.pipeline import RAGVectorStore
from sara.rag.ollama_client import (
    OLLAMA_TEACHER_MODEL,
    OLLAMA_STUDENT_MODEL,
    check_ollama_running,
    ensure_model,
)
from sara.rag.ollama_pipeline import OllamaRAGPipeline
from sara.rag.ollama_kd_spar import OllamaKDSPAR
from sara.core.progress import SaraLogger

CORPUS = {
    "kd_basics.txt": """
        Knowledge distillation compresses large teacher models into smaller students.
        The teacher produces soft probability distributions over all classes using
        a temperature-scaled softmax. These soft targets reveal inter-class similarity
        called dark knowledge. The combined KD loss weights KL divergence from soft
        targets against cross-entropy from hard labels, controlled by alpha.
        The T-squared factor compensates for gradient magnitude reduction at high temperatures.
    """,
    "rag_basics.txt": """
        Retrieval-Augmented Generation (RAG) grounds responses in retrieved passages.
        A good RAG assistant cites every claim with [Doc-N] inline notation.
        ChromaDB stores document embeddings for semantic retrieval.
        Uncertainty should be expressed explicitly when context is partial.
        Never make claims that go beyond what the retrieved passages support.
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


def sanity_check(teacher_model: str, student_model: str) -> bool:
    """
    Quick connectivity check — one corpus ingest, one teacher query,
    one student query. Takes ~1 min. Used by setup_and_run.sh Step 4.
    """
    log = SaraLogger("Sanity")
    log.banner(
        "Sara — Ollama Sanity Check",
        f"Teacher : {teacher_model}",
        f"Student : {student_model}",
        "Single query — tests Ollama connectivity only",
    )

    log.section("Checking Ollama server")
    if not check_ollama_running():
        log.error("Ollama server not running. Start with: ollama serve")
        return False
    log.done("Ollama server reachable")

    log.section("Ensuring models are available")
    for model in [teacher_model, student_model]:
        log.info(f"  Checking {model} …")
        ensure_model(model)
        log.done(f"{model} ready")

    log.section("Building vector store")
    store = RAGVectorStore(persist_path="./sanity_check_db")
    teacher_pipe = OllamaRAGPipeline(teacher_model, store=store, auto_pull=False)
    n = teacher_pipe.ingest(CORPUS)
    log.done(f"Indexed {n} chunks")

    log.section("Teacher query")
    log.step("Running one teacher query", total=1)
    q = "What is dark knowledge in knowledge distillation?"
    try:
        t_resp = teacher_pipe.query(q, return_context=False).answer
        log.tick(1)
        log.done(f"Teacher responded ({len(t_resp)} chars)")
        log.info(f"  Q: {q}")
        log.info(f"  A: {t_resp[:120]}{'…' if len(t_resp) > 120 else ''}")
    except Exception as e:
        log.error(f"Teacher query failed: {e}")
        return False

    log.section("Student query")
    log.step("Running one student query", total=1)
    student_pipe = OllamaRAGPipeline(student_model, store=store, auto_pull=False)
    try:
        s_resp = student_pipe.query(q, return_context=False).answer
        log.tick(1)
        log.done(f"Student responded ({len(s_resp)} chars)")
        log.info(f"  A: {s_resp[:120]}{'…' if len(s_resp) > 120 else ''}")
    except Exception as e:
        log.error(f"Student query failed: {e}")
        return False

    log.summary()
    log.done("Sanity check PASSED — Ollama pipeline is working correctly")
    return True


def main(teacher_model: str, student_model: str):
    log = SaraLogger("KD-SPAR Example")
    log.banner(
        "Sara (सार) — Ollama KD-SPAR Example",
        f"Teacher : {teacher_model}",
        f"Student : {student_model}",
        "Full 4-iteration run",
    )

    if not check_ollama_running():
        log.error("Ollama server not running. Start with: ollama serve")
        return

    log.section("Ensuring models")
    for model in [teacher_model, student_model]:
        log.info(f"  Checking {model} …")
        ensure_model(model)
        log.done(f"{model} ready")

    log.section("Building vector store")
    store = RAGVectorStore(persist_path="./ollama_demo_db")
    teacher_pipe = OllamaRAGPipeline(teacher_model, store=store, auto_pull=False)
    n = teacher_pipe.ingest(CORPUS)
    log.done(f"Indexed {n} chunks")

    log.section("Harvesting teacher responses")
    spar = OllamaKDSPAR(teacher_model, student_model, vector_store=store, auto_pull=False)
    all_queries = TRAIN_QUERIES + VAL_QUERIES
    log.step(f"Querying teacher for {len(all_queries)} responses", total=len(all_queries))
    teacher_responses: dict = {}
    for i, q in enumerate(all_queries, 1):
        try:
            teacher_responses[q] = teacher_pipe.query(q, return_context=False).answer
        except Exception as e:
            log.warn(f"  q{i} failed: {e}")
        log.tick(i)
    log.done(f"{len(teacher_responses)}/{len(all_queries)} responses collected")

    log.section("Running KD-SPAR loop")
    # run() now has full per-phase progress built in (SaraLogger in OllamaKDSPAR.run)
    final_prompt, history = spar.run(
        train_queries     = TRAIN_QUERIES,
        val_queries       = VAL_QUERIES,
        teacher_responses = teacher_responses,
        iterations        = 4,
        threshold         = 0.002,
        n_proposals       = 3,
        top_k             = 2,
    )

    accepted = sum(1 for s in history if s.accepted)
    log.section("Results")
    log.metric("Iterations completed", str(len(history)))
    log.metric("Accepted",             f"{accepted}/{len(history)}")
    if history:
        log.metric("Score",
                   f"{history[0].score_before:.4f} → {history[-1].score_after:.4f}",
                   f"Δ={history[-1].score_after - history[0].score_before:+.4f}")

    log.info("Optimised student prompt:")
    log.info("-" * 55)
    for line in final_prompt.split("\n")[:10]:
        log.info(f"  {line}")
    if final_prompt.count("\n") > 10:
        log.info("  … (truncated)")
    log.info("-" * 55)

    log.section("Student sample responses (optimised prompt)")
    student_pipe = OllamaRAGPipeline(
        student_model, store=store, system_prompt=final_prompt, auto_pull=False
    )
    for q in VAL_QUERIES[:2]:
        try:
            resp = student_pipe.query(q, return_context=False).answer
            log.info(f"\n  Q: {q}")
            log.info(f"  A: {resp[:300]}{'…' if len(resp) > 300 else ''}")
        except Exception as e:
            log.warn(f"  Sample failed: {e}")

    log.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sara KD-SPAR — Ollama Example")
    parser.add_argument("--teacher",     default=OLLAMA_TEACHER_MODEL,
                        help="Teacher model (default: llama3.1:8b)")
    parser.add_argument("--student",     default=OLLAMA_STUDENT_MODEL,
                        help="Student model (default: llama3.2:3b)")
    parser.add_argument("--sanity-only", action="store_true",
                        help="Quick connectivity check only (~1 min)")
    args = parser.parse_args()

    if args.sanity_only:
        ok = sanity_check(args.teacher, args.student)
        sys.exit(0 if ok else 1)
    else:
        main(args.teacher, args.student)
