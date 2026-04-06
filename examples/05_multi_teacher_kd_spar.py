# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.4.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
"""
examples/05_multi_teacher_kd_spar.py
======================================
Multi-Teacher KD-SPAR: align one student to a committee of specialist teachers.

Backend is read from configs/backend.yaml (default: Ollama/FOSS).
Override:  export SARA_BACKEND=anthropic  ANTHROPIC_API_KEY=sk-ant-...

Run:
    python examples/05_multi_teacher_kd_spar.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sara.rag.pipeline import RAGVectorStore
from sara.rag.backend import get_pipeline, cfg, describe
from sara.rag.kd_spar_multi_teacher import MultiTeacherKDSPAR, TeacherSpec
from sara.core.progress import SaraLogger

CORPUS = {
    "kd_basics.txt": """
        Knowledge distillation compresses large teacher models into smaller student models.
        The teacher's output probability distributions serve as soft targets for the student.
        Temperature scaling controls how soft the teacher's distribution is.
        The KD loss combines KL divergence from soft targets with cross-entropy on hard labels.
    """,
    "rag_pipeline.txt": """
        Retrieval-Augmented Generation combines document retrieval with language model generation.
        Documents are chunked, embedded, and stored in a vector database like ChromaDB.
        Citation formatting using [Doc-N] notation helps trace claims to their sources.
    """,
    "kd_spar.txt": """
        KD-SPAR enables the student model to rewrite its own system prompt.
        The multi-teacher variant aligns the student to multiple specialist teachers simultaneously.
        The worst-aligned teacher drives the self-interview each iteration.
        Non-regression validation ensures no secondary teacher degrades by more than the tolerance.
    """,
}

TEACHER_SYSTEM_PROMPTS = {
    "citation_expert": (
        "You are a RAG assistant. You MUST cite every single claim with [Doc-N] notation. "
        "Never make a claim without citing a specific passage. Always include at least 3 citations."
    ),
    "reasoning_expert": (
        "You are a RAG assistant. Always reason step by step before answering. "
        "Show your reasoning chain explicitly. Connect evidence from multiple passages."
    ),
    "calibration_expert": (
        "You are a RAG assistant. Always express confidence explicitly. Use 'may', 'might', "
        "or 'appears to' when evidence is partial. Never overstate certainty."
    ),
}

TRAIN_QUERIES = [
    "What is temperature scaling in knowledge distillation?",
    "How does ChromaDB store embeddings for retrieval?",
    "What failure modes does KD-SPAR target?",
    "How does the student learn from soft targets?",
    "What is the role of citation format in RAG?",
]
VAL_QUERIES = [
    "How does KD-SPAR use self-knowledge?",
    "What is the T-squared scaling factor?",
]


def main():
    log = SaraLogger("Multi-Teacher KD-SPAR")
    log.banner(
        "Sara — Multi-Teacher KD-SPAR",
        f"Teacher : {cfg['teacher_model']}  (3 specialist prompts)",
        f"Student : {cfg['student_model']}",
        f"Backend : {cfg['backend']}",
        "Primary: citation_expert  |  regression_tol: 2%",
    )
    log.info(describe())

    log.section("Vector store + corpus ingestion")
    store     = RAGVectorStore(persist_path="./mt_spar_demo_db")
    base_pipe = get_pipeline("teacher", store=store)
    log.step("Ingesting corpus", total=len(CORPUS))
    base_pipe.ingest(CORPUS)
    log.tick(len(CORPUS))
    log.done(f"Indexed {store.count} chunks")

    log.section("Building teacher specs (3 specialists)")
    specs = [
        TeacherSpec(name="citation_expert",   model_id=cfg["teacher_model"],
                    weight=2.0, is_primary=True),
        TeacherSpec(name="reasoning_expert",  model_id=cfg["teacher_model"], weight=1.5),
        TeacherSpec(name="calibration_expert",model_id=cfg["teacher_model"], weight=1.0),
    ]
    for s in specs:
        log.info(f"  {s.name}  weight={s.weight}  {'(primary)' if s.is_primary else ''}")

    spar = MultiTeacherKDSPAR(
        student_model=cfg["student_model"],
        teachers=specs,
        vector_store=store,
        regression_tol=0.02,
    )

    # Inject specialist system prompts
    from sara.rag.pipeline import RAGPipeline
    for spec in specs:
        spar._teacher_pipes[spec.name] = RAGPipeline(
            model_id=spec.model_id, store=store,
            system_prompt=TEACHER_SYSTEM_PROMPTS[spec.name],
        )

    log.section("Harvesting teacher responses (3 × all queries)")
    all_q = TRAIN_QUERIES + VAL_QUERIES
    total  = len(specs) * len(all_q)
    log.step(f"Querying {len(specs)} teachers × {len(all_q)} queries = {total} total",
             total=total)
    log.start_heartbeat(interval=25, message="Waiting for model response…")
    teacher_response_sets = spar.harvest_teacher_responses(all_q)
    log.stop_heartbeat()
    for t_name, resps in teacher_response_sets.items():
        log.done(f"  {t_name}: {len(resps)} responses")

    log.section("Multi-Teacher KD-SPAR loop")
    final_prompt, history = spar.run(
        train_queries=TRAIN_QUERIES,
        val_queries=VAL_QUERIES,
        teacher_response_sets=teacher_response_sets,
        iterations=5,
        threshold=0.002,
        n_proposals=3,
        top_k=2,
        log_path="./mt_spar_log.jsonl",
    )

    accepted = sum(1 for s in history if s.accepted)
    log.section("Results")
    log.metric("Iterations", str(len(history)))
    log.metric("Accepted",   f"{accepted}/{len(history)}")
    if history:
        log.metric("Score",
                   f"{history[0].score_before:.4f} → {history[-1].score_after:.4f}")

    log.info("Optimised prompt (first 10 lines):")
    log.info("-" * 55)
    for line in final_prompt.split("\n")[:10]:
        log.info(f"  {line}")
    log.summary()


if __name__ == "__main__":
    main()
