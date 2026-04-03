"""
examples/06_adversarial_kd_spar.py
====================================
Adversarial KD-SPAR: build prompt robustness by targeting hard examples.

Demonstrates:
  1. Gap-mining: finding production queries where student already struggles
  2. Adversarial query generation: teacher generates hard questions about the topic
  3. Dual-objective validation: improves on hard queries without regressing on standard

Run:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python examples/06_adversarial_kd_spar.py

Requirements:
    pip install anthropic chromadb sentence-transformers
"""

import os
from sara.rag.pipeline import RAGPipeline, RAGVectorStore, TEACHER_MODEL, STUDENT_MODEL
from sara.rag.kd_spar_adversarial import AdversarialKDSPAR

CORPUS = {
    "kd_theory.txt": """
        Knowledge distillation (KD) was formalised by Hinton et al. in 2015.
        The teacher model produces soft probability distributions using temperature scaling.
        Temperature T controls how soft the distribution is — higher T flattens it.
        The student minimises KL divergence from teacher soft targets plus cross-entropy.
        The T-squared factor compensates for gradient magnitude reduction at high temperatures.
        Feature-based distillation (FitNets) guides intermediate layer representations.
        Attention transfer distillation aligns spatial attention maps between models.
        Relational KD distils pairwise structural relationships between samples.
        Data-free KD uses a generator to create pseudo-samples when no training data is available.
        Progressive distillation chains stages from large to small reducing capacity gradually.
    """,
    "kd_applications.txt": """
        DistilBERT retains 97% of BERT's performance at 40% smaller size.
        TinyBERT achieves 7.5x speedup using attention-head distillation.
        Whisper-large was distilled to Whisper-tiny fitting in 150MB.
        YOLOv8-Nano is distilled from YOLOv8-X with 30x size reduction.
        Phi-3-Mini was distilled from GPT-4 outputs and outperforms Llama-2-7B.
        Medical imaging uses data-free KD to comply with HIPAA patient data rules.
        Edge deployment uses KD to run models on devices with 512MB RAM constraints.
        Soft prompt distillation optimises prefix tokens without changing model weights.
        KD-SPAR lets the student model write amendments to its own system prompt.
    """,
}

# Standard production queries (easy, common queries)
STANDARD_QUERIES = [
    "What is knowledge distillation?",
    "How does temperature scaling work?",
    "What is DistilBERT?",
    "How does Whisper compression work?",
    "What is soft prompt distillation?",
]

# These will be supplemented by gap-mined and generated hard queries
SEED_HARD_QUERIES = [
    "Explain why T-squared scaling is necessary and what would happen without it.",
    "How does relational KD differ from feature-based KD in terms of what is transferred?",
    "Under what conditions does progressive distillation outperform single-stage distillation?",
    "What is the theoretical relationship between temperature T and the entropy of soft targets?",
]


def collect_teacher_responses(teacher_pipe, all_queries):
    """Collect teacher reference responses for all queries."""
    responses = {}
    print("\nCollecting teacher responses …")
    for q in all_queries:
        try:
            resp = teacher_pipe.query(q, return_context=False)
            responses[q] = resp.answer
            print(f"  ✓ {q[:60]}")
        except Exception as exc:
            print(f"  ✗ {q[:40]}: {exc}")
    return responses


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY before running.")
        return

    print("Setting up vector store and ingesting corpus …")
    store         = RAGVectorStore(persist_path="./adv_spar_demo_db")
    teacher_pipe  = RAGPipeline(TEACHER_MODEL, store=store)
    teacher_pipe.ingest(CORPUS)
    print(f"Indexed {store.count} chunks")

    # Collect teacher responses
    all_initial_queries = STANDARD_QUERIES + SEED_HARD_QUERIES
    teacher_responses   = collect_teacher_responses(teacher_pipe, all_initial_queries)

    print("\nInitialising AdversarialKDSPAR …")
    spar = AdversarialKDSPAR(
        teacher_model         = TEACHER_MODEL,
        student_model         = STUDENT_MODEL,
        vector_store          = store,
        adversarial_topics    = ["knowledge distillation theory", "KD model compression"],
        n_generated_per_topic = 5,       # generate 5 hard queries per topic
        hardness_percentile   = 0.5,     # bottom 50% of initial KD scores are "hard"
        dual_threshold        = 0.003,   # must improve adversarial queries by this much
        standard_regression   = 0.02,    # standard queries may not drop by more than 2%
    )

    # Build the hard query set
    print("\n[Phase 1] Building hard query set …")
    print("  Step A: Gap-mining from seed hard queries …")
    mined_queries = spar.mine_hard_queries(
        SEED_HARD_QUERIES, teacher_responses,
        student_pipeline=RAGPipeline(STUDENT_MODEL, store=store),
    )
    print(f"  Mined {len(mined_queries)} hard queries from seed set")

    print("  Step B: Teacher generates adversarial queries …")
    generated_queries = spar.generate_adversarial_queries(teacher_pipe)
    print(f"  Generated {len(generated_queries)} adversarial queries")

    # Collect teacher responses for any new generated queries
    new_queries = [aq.query for aq in generated_queries
                   if aq.query not in teacher_responses]
    if new_queries:
        new_responses = collect_teacher_responses(teacher_pipe, new_queries)
        teacher_responses.update(new_responses)

    all_hard_queries = mined_queries + generated_queries

    print(f"\n  Hard query set: {len(all_hard_queries)} total")
    print(f"    {'Query':<60}  Difficulty  Type")
    print(f"    {'-'*60}  ----------  ----")
    for aq in all_hard_queries[:8]:
        print(f"    {aq.query[:58]:<58}  {aq.difficulty_score:.2f}       {aq.adversarial_type}")

    print(f"\n[Phase 2] Running Adversarial KD-SPAR loop …")
    final_prompt, history = spar.run_adversarial(
        adversarial_queries = all_hard_queries,
        standard_queries    = STANDARD_QUERIES,
        teacher_responses   = teacher_responses,
        iterations          = 5,
        n_proposals         = 3,
        top_k               = 2,
        log_path            = "./adv_spar_log.jsonl",
    )

    accepted = sum(1 for s in history if s.accepted)
    print(f"\n{'='*55}")
    print(f"Adversarial KD-SPAR complete")
    print(f"  Total iterations : {len(history)}")
    print(f"  Accepted         : {accepted}")
    if history:
        deltas = [s.delta for s in history]
        print(f"  Delta range      : {min(deltas):+.4f} to {max(deltas):+.4f}")

    print(f"\nFinal prompt:\n{'-'*55}")
    print(final_prompt[:800] + ("…" if len(final_prompt) > 800 else ""))
    print(f"{'-'*55}")

    # Compare standard queries before and after
    print(f"\nSanity check: running student on standard queries with new prompt …")
    student_pipe = RAGPipeline(STUDENT_MODEL, store=store, system_prompt=final_prompt)
    for q in STANDARD_QUERIES[:3]:
        if q in teacher_responses:
            resp  = student_pipe.query(q, return_context=False)
            score = spar._batch_kd([q], teacher_responses, student_pipe)
            print(f"  [{score:.3f}] {q[:55]}")


if __name__ == "__main__":
    main()
