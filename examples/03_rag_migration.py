"""
examples/03_rag_migration.py
=============================
Migrate from claude-3-5-sonnet-20241022 (teacher)
        to claude-sonnet-4-5-20250929    (student)
using the full 5-phase RAG KD migration pipeline.

Run:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python examples/03_rag_migration.py

Requirements:
    pip install anthropic chromadb sentence-transformers
"""

import os
from sara.rag.pipeline import RAGPipeline, RAGVectorStore, TEACHER_MODEL, STUDENT_MODEL
from sara.rag.migration import RAGMigration

# ── Demo corpus ───────────────────────────────────────────────────────────────
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
        The student acts as its own prompt author, leveraging self-knowledge about
        what instructions elicit its best performance.
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
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY before running this example.")
        return

    print("Building vector store and ingesting corpus …")
    store    = RAGVectorStore(persist_path="./demo_chroma_db")
    pipeline = RAGPipeline(model_id=TEACHER_MODEL, store=store)
    n        = pipeline.ingest(CORPUS)
    print(f"Indexed {n} chunks")

    print(f"\nStarting migration:")
    print(f"  Teacher : {TEACHER_MODEL}")
    print(f"  Student : {STUDENT_MODEL}")

    migration = RAGMigration(
        teacher_model=TEACHER_MODEL,
        student_model=STUDENT_MODEL,
        vector_store=store,
    )

    result = migration.run(
        query_log  = QUERIES,
        n_harvest  = len(QUERIES),
        verbose    = True,
    )

    print(f"\n{'='*50}")
    print(f"Migration complete")
    print(f"  Mean KD score : {result.mean_kd:.4f}")
    print(f"  Promote?      : {'YES ✓' if result.report.pass_all else 'NO  — review needed'}")


if __name__ == "__main__":
    main()
