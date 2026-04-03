"""
examples/05_multi_teacher_kd_spar.py
======================================
Multi-Teacher KD-SPAR: align a single student to a committee of specialist teachers.

In this demo we simulate three teachers with different specialties using the
same underlying model (since we only have one API key), but with different
system prompts that enforce distinct styles:
  - teacher_citation:   strict [Doc-N] citation requirement
  - teacher_reasoning:  step-by-step chain-of-thought requirement
  - teacher_calibration: explicit uncertainty hedging requirement

Run:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python examples/05_multi_teacher_kd_spar.py

Requirements:
    pip install anthropic chromadb sentence-transformers
"""

import os
from sara.rag.pipeline import RAGPipeline, RAGVectorStore, TEACHER_MODEL, STUDENT_MODEL
from sara.rag.kd_spar_multi_teacher import MultiTeacherKDSPAR, TeacherSpec

CORPUS = {
    "kd_basics.txt": """
        Knowledge distillation compresses large teacher models into smaller student models.
        The teacher's output probability distributions serve as soft targets for the student.
        Temperature scaling controls how soft the teacher's distribution is.
        Higher temperature produces flatter distributions that reveal inter-class similarity.
        The KD loss combines KL divergence from soft targets with cross-entropy on hard labels.
    """,
    "rag_pipeline.txt": """
        Retrieval-Augmented Generation (RAG) combines document retrieval with language model generation.
        Documents are chunked, embedded, and stored in a vector database like ChromaDB.
        At query time, semantically similar passages are retrieved and provided as context.
        The language model generates an answer grounded in the retrieved passages.
        Citation formatting using [Doc-N] notation helps trace claims to their sources.
    """,
    "kd_spar.txt": """
        KD-SPAR enables the student model to rewrite its own system prompt.
        The algorithm diagnoses failure modes by comparing student to teacher responses.
        The student proposes prompt amendments during a self-interview phase.
        Proposals are aggregated and validated before being committed to the prompt.
        The multi-teacher variant aligns the student to multiple specialist teachers simultaneously.
    """,
}

# Simulate three teachers with distinct style emphases via different system prompts
TEACHER_SYSTEM_PROMPTS = {
    "citation_expert": (
        "You are a RAG assistant. You MUST cite every single claim with [Doc-N] notation. "
        "Never make a claim without citing a specific passage. Always include at least 3 citations."
    ),
    "reasoning_expert": (
        "You are a RAG assistant. Always reason step by step before answering. "
        "Show your reasoning chain explicitly. Connect evidence from multiple passages "
        "before drawing conclusions."
    ),
    "calibration_expert": (
        "You are a RAG assistant. Always express confidence explicitly. Use 'may', 'might', "
        "or 'appears to' when evidence is partial. Say 'The context confirms that' only when "
        "direct evidence exists. Never overstate certainty."
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
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY before running.")
        return

    print("Setting up shared vector store …")
    store = RAGVectorStore(persist_path="./mt_spar_demo_db")
    base_pipe = RAGPipeline(TEACHER_MODEL, store=store)
    base_pipe.ingest(CORPUS)
    print(f"Indexed {store.count} chunks")

    print("\nBuilding teacher specs …")
    specs = [
        TeacherSpec(
            name       = "citation_expert",
            model_id   = TEACHER_MODEL,
            weight     = 2.0,
            is_primary = True,
        ),
        TeacherSpec(
            name   = "reasoning_expert",
            model_id = TEACHER_MODEL,
            weight = 1.5,
        ),
        TeacherSpec(
            name   = "calibration_expert",
            model_id = TEACHER_MODEL,
            weight = 1.0,
        ),
    ]

    spar = MultiTeacherKDSPAR(
        student_model  = STUDENT_MODEL,
        teachers       = specs,
        vector_store   = store,
        regression_tol = 0.02,
    )

    # Override teacher pipelines with specialist system prompts
    for spec in specs:
        spar._teacher_pipes[spec.name] = RAGPipeline(
            model_id      = spec.model_id,
            store         = store,
            system_prompt = TEACHER_SYSTEM_PROMPTS[spec.name],
        )

    print("\nHarvesting teacher responses from all three specialist teachers …")
    teacher_response_sets = spar.harvest_teacher_responses(TRAIN_QUERIES + VAL_QUERIES)
    for t_name, resps in teacher_response_sets.items():
        print(f"  {t_name}: {len(resps)} responses collected")

    print(f"\nRunning Multi-Teacher KD-SPAR …")
    print(f"  Primary teacher  : citation_expert (weight=2.0)")
    print(f"  Secondary teachers: reasoning_expert, calibration_expert")
    print(f"  Regression tol   : 2%  (no secondary teacher may degrade by more)")

    final_prompt, history = spar.run(
        train_queries         = TRAIN_QUERIES,
        val_queries           = VAL_QUERIES,
        teacher_response_sets = teacher_response_sets,
        iterations            = 5,
        threshold             = 0.002,
        n_proposals           = 3,
        top_k                 = 2,
        log_path              = "./mt_spar_log.jsonl",
    )

    accepted = sum(1 for s in history if s.accepted)
    print(f"\n{'='*55}")
    print(f"Multi-Teacher KD-SPAR complete")
    print(f"  Iterations : {len(history)}  |  Accepted: {accepted}")
    if history:
        print(f"  Score trajectory: {history[0].score_before:.4f} → {history[-1].score_after:.4f}")

    print(f"\nFinal optimised prompt:\n{'-'*55}")
    print(final_prompt)
    print(f"{'-'*55}")


if __name__ == "__main__":
    main()
