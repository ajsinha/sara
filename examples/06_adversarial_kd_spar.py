# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.6.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
"""
examples/06_adversarial_kd_spar.py
====================================
Adversarial KD-SPAR: build prompt robustness by targeting hard examples.

Backend is read from configs/backend.yaml (default: Ollama/FOSS).
Override:  export SARA_BACKEND=anthropic  ANTHROPIC_API_KEY=sk-ant-...

Run:
    python examples/06_adversarial_kd_spar.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sara.rag.pipeline import RAGVectorStore
from sara.rag.backend import get_pipeline, cfg, describe
from sara.rag.kd_spar_adversarial import AdversarialKDSPAR
from sara.core.progress import SaraLogger

CORPUS = {
    "kd_theory.txt": """
        Knowledge distillation was formalised by Hinton et al. in 2015.
        The teacher produces soft probability distributions using temperature scaling.
        Temperature T controls how soft the distribution is — higher T flattens it.
        The student minimises KL divergence from teacher soft targets plus cross-entropy.
        The T-squared factor compensates for gradient magnitude reduction at high temperatures.
        Feature-based distillation guides intermediate layer representations.
        Attention transfer distillation aligns spatial attention maps between models.
        Relational KD distils pairwise structural relationships between samples.
        Progressive distillation chains stages from large to small reducing capacity gradually.
    """,
    "kd_applications.txt": """
        DistilBERT retains 97% of BERT's performance at 40% smaller size.
        TinyBERT achieves 7.5x speedup using attention-head distillation.
        Whisper-large was distilled to Whisper-tiny fitting in 150MB.
        YOLOv8-Nano is distilled from YOLOv8-X with 30x size reduction.
        Phi-3-Mini was distilled from GPT-4 outputs and outperforms Llama-2-7B.
        KD-SPAR lets the student model write amendments to its own system prompt.
    """,
}

STANDARD_QUERIES = [
    "What is knowledge distillation?",
    "How does temperature scaling work?",
    "What is DistilBERT?",
    "How does Whisper compression work?",
    "What is soft prompt distillation?",
]

SEED_HARD_QUERIES = [
    "Explain why T-squared scaling is necessary and what would happen without it.",
    "How does relational KD differ from feature-based KD in terms of what is transferred?",
    "Under what conditions does progressive distillation outperform single-stage distillation?",
    "What is the theoretical relationship between temperature T and entropy of soft targets?",
]


def main():
    log = SaraLogger("Adversarial KD-SPAR")
    log.banner(
        "Sara — Adversarial KD-SPAR",
        f"Teacher : {cfg['teacher_model']}",
        f"Student : {cfg['student_model']}",
        f"Backend : {cfg['backend']}",
        "Hard query mining + dual-objective validation",
    )
    log.info(describe())

    log.section("Vector store + corpus ingestion")
    store        = RAGVectorStore(persist_path="./adv_spar_demo_db")
    teacher_pipe = get_pipeline("teacher", store=store)
    log.step("Ingesting corpus", total=len(CORPUS))
    teacher_pipe.ingest(CORPUS)
    log.tick(len(CORPUS))
    log.done(f"Indexed {store.count} chunks")

    log.section("Harvesting teacher responses")
    all_initial = STANDARD_QUERIES + SEED_HARD_QUERIES
    log.step(f"Querying teacher for {len(all_initial)} responses",
             total=len(all_initial))
    log.start_heartbeat(interval=25, message="Waiting for model response…")
    teacher_responses: dict = {}
    for i, q in enumerate(all_initial, 1):
        try:
            teacher_responses[q] = teacher_pipe.query(q, return_context=False).answer
        except Exception as e:
            log.warn(f"  q{i} failed: {e}")
        log.tick(i)
    log.done(f"{len(teacher_responses)}/{len(all_initial)} responses collected")

    log.section("Initialising AdversarialKDSPAR")
    spar = AdversarialKDSPAR(
        teacher_model=cfg["teacher_model"],
        student_model=cfg["student_model"],
        vector_store=store,
        adversarial_topics=["knowledge distillation theory", "KD model compression"],
        n_generated_per_topic=5,
        hardness_percentile=0.5,
        dual_threshold=0.003,
        standard_regression=0.02,
    )
    log.info("  hardness_percentile=0.5  dual_threshold=0.003  std_regression=0.02")

    log.section("Phase 1A — Gap-mining hard queries")
    log.step(f"Scoring {len(SEED_HARD_QUERIES)} seed queries",
             total=len(SEED_HARD_QUERIES))
    student_pipe = get_pipeline("student", store=store)
    mined = spar.mine_hard_queries(SEED_HARD_QUERIES, teacher_responses,
                                   student_pipeline=student_pipe)
    log.tick(len(SEED_HARD_QUERIES))
    log.done(f"Mined {len(mined)} hard queries from seed set")

    log.section("Phase 1B — Teacher generates adversarial queries")
    log.step("Generating adversarial queries from topics")
    generated = spar.generate_adversarial_queries(teacher_pipe)
    log.tick(len(generated))
    log.done(f"Generated {len(generated)} adversarial queries")

    # Collect teacher responses for any new generated queries
    new_q = [aq.query for aq in generated if aq.query not in teacher_responses]
    if new_q:
        log.step(f"Collecting teacher responses for {len(new_q)} new queries",
                 total=len(new_q))
        for i, q in enumerate(new_q, 1):
            try:
                teacher_responses[q] = teacher_pipe.query(q, return_context=False).answer
            except Exception as e:
                log.warn(f"  failed: {e}")
            log.tick(i)
        log.done(f"{len(new_q)} new responses collected")

    all_hard = mined + generated
    log.info(f"Hard query set: {len(all_hard)} total")
    for aq in all_hard[:5]:
        log.info(f"  [{aq.difficulty_score:.2f}] {aq.query[:65]}")
    if len(all_hard) > 5:
        log.info(f"  … +{len(all_hard)-5} more")

    log.stop_heartbeat()

    log.section("Phase 2 — Adversarial KD-SPAR loop")
    final_prompt, history = spar.run_adversarial(
        adversarial_queries=all_hard,
        standard_queries=STANDARD_QUERIES,
        teacher_responses=teacher_responses,
        iterations=5,
        n_proposals=3,
        top_k=2,
        log_path="./adv_spar_log.jsonl",
    )

    accepted = sum(1 for s in history if s.accepted)
    log.section("Results")
    log.metric("Iterations", str(len(history)))
    log.metric("Accepted",   f"{accepted}/{len(history)}")
    if history:
        deltas = [s.delta for s in history]
        log.metric("Delta range",
                   f"{min(deltas):+.4f} to {max(deltas):+.4f}")

    log.info("Final prompt (first 10 lines):")
    log.info("-" * 55)
    for line in final_prompt.split("\n")[:10]:
        log.info(f"  {line}")
    log.summary()


if __name__ == "__main__":
    main()
