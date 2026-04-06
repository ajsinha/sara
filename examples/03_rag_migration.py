# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.6.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
"""
examples/03_rag_migration.py
=============================
Full 5-phase RAG KD migration pipeline.

Backend is read from configs/backend.yaml (default: Ollama/FOSS).
Override:  export SARA_BACKEND=anthropic  ANTHROPIC_API_KEY=sk-ant-...

Run:
    python examples/03_rag_migration.py

Requirements:
    pip install -e ".[rag]"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sara.rag.pipeline import RAGVectorStore
from sara.rag.backend import get_pipeline, cfg, describe
from sara.rag.migration import RAGMigration
from sara.core.progress import SaraLogger

CORPUS = {
    "kd_intro.txt": """
        Knowledge distillation is a model compression technique where a smaller
        student model learns to mimic a larger teacher model. The teacher produces
        soft probability distributions (soft targets) using a temperature-scaled
        softmax. These soft targets contain richer information than one-hot labels,
        revealing inter-class similarity structure called 'dark knowledge'.
        The student minimises a weighted KL divergence from the teacher's soft
        targets plus standard cross-entropy on the hard labels.
    """,
    "rag_overview.txt": """
        Retrieval-Augmented Generation (RAG) grounds LLM responses in external
        knowledge by retrieving relevant documents at inference time. A typical
        RAG pipeline has three phases: document ingestion (chunk, embed, store),
        retrieval (semantic search), and generation (LLM synthesis).
        ChromaDB is a popular vector database for RAG, supporting cosine
        similarity search with sentence-transformer embeddings.
    """,
    "kd_spar.txt": """
        KD-SPAR (Knowledge Distillation via Student Prompt Auto-Rewriting) is a
        paradigm where the student model diagnoses its own failure modes and
        proposes targeted amendments to its system prompt. This self-calibrating
        loop does not require model weight updates and works with API-only access.
    """,
}

QUERIES = [
    "What is the role of temperature in knowledge distillation?",
    "How does ChromaDB support RAG pipelines?",
    "What is KD-SPAR and why is it self-calibrating?",
    "What are soft targets and how do they encode dark knowledge?",
    "Explain the three phases of a RAG pipeline.",
    "How does response-based KD differ from feature-based KD?",
    "What failure modes does KD-SPAR target?",
    "How is ChromaDB's cosine search used in retrieval?",
]


def main():
    log = SaraLogger("RAG Migration")
    log.banner(
        "Sara — RAG KD Migration Pipeline",
        f"Teacher : {cfg['teacher_model']}",
        f"Student : {cfg['student_model']}",
        f"Backend : {cfg['backend']}",
    )
    log.info(describe())

    log.section("Vector store + corpus ingestion")
    store    = RAGVectorStore(persist_path="./demo_chroma_db")
    pipeline = get_pipeline("teacher", store=store)
    log.step("Ingesting corpus", total=len(CORPUS))
    n = pipeline.ingest(CORPUS)
    log.tick(len(CORPUS))
    log.done(f"Indexed {n} chunks from {len(CORPUS)} documents")

    log.section("Running migration pipeline")
    log.info(f"  {len(QUERIES)} queries  |  all 5 phases")
    log.start_heartbeat(interval=25, message="Waiting for model response…")

    migration = RAGMigration(
        teacher_model=cfg["teacher_model"],
        student_model=cfg["student_model"],
        vector_store=store,
    )
    result = migration.run(query_log=QUERIES, n_harvest=len(QUERIES), verbose=True)
    log.stop_heartbeat()

    log.section("Migration results")
    log.metric("Mean KD score", f"{result.mean_kd:.4f}")
    log.metric("Promote?", "YES ✓" if result.report.pass_all else "NO  — review needed")

    if hasattr(result.report, "checks"):
        for check, passed in result.report.checks.items():
            log.info(f"  {'✓' if passed else '✗'}  {check}")

    log.summary()


if __name__ == "__main__":
    main()
