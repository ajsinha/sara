# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.6.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
"""
examples/07_federated_kd_spar.py
==================================
Federated KD-SPAR: multiple sites jointly optimise a global prompt
without sharing any raw query or response data.

Backend is read from configs/backend.yaml (default: Ollama/FOSS).
Override:  export SARA_BACKEND=anthropic  ANTHROPIC_API_KEY=sk-ant-...

Run:
    python examples/07_federated_kd_spar.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sara.rag.pipeline import RAGVectorStore
from sara.rag.backend import get_pipeline, cfg, describe
from sara.rag.kd_spar_federated import (
    FederatedSimulation,
    FederatedKDSPARClient,
    FederatedKDSPARServer,
    FederatedClientConfig,
)
from sara.core.progress import SaraLogger

CORPUS = {
    "clinical_kd.txt": """
        Knowledge distillation helps deploy AI models in clinical settings.
        Patient privacy regulations require that no raw data leaves individual hospital sites.
        Federated learning enables collaborative model improvement without data sharing.
        ChromaDB can store medical document embeddings for semantic retrieval.
        Citation format [Doc-N] helps clinicians trace AI responses to source documents.
        Uncertainty hedging is critical in clinical settings to avoid overconfident claims.
        Federated KD-SPAR shares only instruction strings, preserving patient data privacy.
    """,
    "kd_methods.txt": """
        Response-based distillation uses teacher output logits as training targets.
        Feature-based distillation aligns intermediate layer representations.
        Temperature scaling controls the softness of teacher output distributions.
        Progressive distillation chains multiple stages to avoid capacity-gap failures.
        KD-SPAR diagnoses failure modes and proposes targeted system prompt amendments.
    """,
}

# Simulated private traces per site (in production: real RAG logs, never shared)
SITE_TRACES = {
    "site_a": [
        ("What is knowledge distillation?",
         "Knowledge distillation [Doc-1] transfers knowledge from a large teacher "
         "to a smaller student [Doc-2]. The teacher's soft targets [Doc-1] contain "
         "rich inter-class similarity called dark knowledge [Doc-2]."),
        ("How does temperature scaling work?",
         "Temperature T [Doc-1] controls the softness of the teacher's distribution. "
         "Higher T [Doc-1] flattens it, revealing inter-class similarity [Doc-2]. "
         "The T-squared factor [Doc-1] compensates for gradient magnitude reduction."),
        ("What is feature-based distillation?",
         "Feature-based distillation [Doc-1] aligns intermediate representations. "
         "FitNets [Doc-2] pioneered this via hint layers between networks."),
    ],
    "site_b": [
        ("What is knowledge distillation?",
         "Step by step: First, a large teacher model is trained on hard labels. "
         "Then, the teacher produces soft probability distributions. "
         "The student learns from these soft targets to compress teacher knowledge. "
         "Summary: teacher → soft targets → student training → compressed model."),
        ("How does temperature scaling work?",
         "Step 1: Take teacher raw logits. Step 2: Divide by temperature T. "
         "Step 3: Apply softmax. At T=4+ we get flatter distributions. "
         "Multiply KL loss by T-squared to compensate for gradient reduction."),
        ("What is Federated KD-SPAR?",
         "Step 1: each site diagnoses failures locally. "
         "Step 2: sites send instruction strings (no data). "
         "Step 3: server aggregates and validates. "
         "Step 4: server broadcasts improved global prompt. "
         "Key: only text crosses boundaries, preserving patient data privacy."),
    ],
    "site_c": [
        ("What is knowledge distillation?",
         "Knowledge distillation appears to be a compression technique where "
         "a smaller student may learn from a larger teacher. Results suggest "
         "students can possibly recover much of the teacher's accuracy."),
        ("What is RAG?",
         "Retrieval-Augmented Generation might combine document retrieval "
         "with language model generation. Quality appears to depend on "
         "retrieval accuracy, though this may vary by implementation."),
        ("What is KD-SPAR?",
         "KD-SPAR might be a paradigm where the student appears to diagnose "
         "its own failures and may propose prompt amendments. The federated "
         "variant possibly addresses privacy by sharing only instruction strings."),
    ],
}


def main():
    log = SaraLogger("Federated KD-SPAR")
    log.banner(
        "Sara — Federated KD-SPAR",
        f"Student : {cfg['student_model']}",
        f"Backend : {cfg['backend']}",
        "3 simulated sites — no raw data crosses boundaries",
        "Only instruction strings shared",
    )
    log.info(describe())

    log.section("Vector store + corpus ingestion")
    store     = RAGVectorStore(persist_path="./fed_spar_demo_db")
    base_pipe = get_pipeline("teacher", store=store)
    log.step("Ingesting shared corpus", total=len(CORPUS))
    base_pipe.ingest(CORPUS)
    log.tick(len(CORPUS))
    log.done(f"Indexed {store.count} chunks")

    log.section("Federation setup")
    all_traces = [(q, r) for traces in SITE_TRACES.values() for q, r in traces]
    for site, traces in SITE_TRACES.items():
        log.info(f"  {site}: {len(traces)} private traces (never leave this site)")
    log.info(f"  Total traces: {len(all_traces)}")

    log.step("Building federated simulation (3 clients + server)")
    sim = FederatedSimulation(
        n_clients=3, all_traces=all_traces,
        val_fraction=0.2, student_model=cfg["student_model"],
        vector_store=store,
    )
    server = sim.build_server(threshold=0.002, regression_tol=0.02)
    log.done(f"Server ready  |  val queries: {len(server.val_queries)}")

    for client in server.clients:
        log.info(f"  {client.config.client_id}: "
                 f"{len(client.local_traces)} traces  (private, not shared)")

    log.section("Privacy guarantees")
    log.info("  ✓  No query text leaves any client site")
    log.info("  ✓  No model responses leave any client site")
    log.info("  ✓  Only instruction strings cross site boundaries")
    log.info("  ✓  Server validates on its own data, not client data")

    log.section("Federated optimisation rounds")
    log.start_heartbeat(interval=25, message="Waiting for federated round…")
    final_prompt, history = server.run(
        rounds=5, min_clients=2,
        log_path="./fed_spar_log.jsonl",
    )
    log.stop_heartbeat()

    accepted = sum(1 for r in history if r.accepted)
    total_props = sum(r.total_proposals for r in history)

    log.section("Results")
    log.metric("Rounds completed",   str(len(history)))
    log.metric("Rounds accepted",    f"{accepted}/{len(history)}")
    log.metric("Total proposals",    str(total_props))
    if history:
        log.metric("Score trajectory",
                   f"{history[0].score_before:.4f} → {history[-1].score_after:.4f}")

    log.info("\n  Per-round summary:")
    for r in history:
        status = "✓" if r.accepted else "✗"
        log.info(f"    Round {r.round_number}: {status}  "
                 f"Δ={r.delta:+.4f}  "
                 f"{r.total_proposals} proposals  "
                 f"{len(r.clients_participated)} sites")
        for ins in (r.selected_instrs or []):
            log.info(f"      → {ins[:70]}")

    log.info(f"\n  Final global prompt ({len(final_prompt)} chars):")
    log.info("-" * 55)
    for line in final_prompt.split("\n")[:10]:
        log.info(f"  {line}")
    log.summary()


if __name__ == "__main__":
    main()
