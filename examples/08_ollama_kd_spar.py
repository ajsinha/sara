"""
examples/08_ollama_kd_spar.py
==============================
KD-SPAR using local Ollama models — no API key, no cost, works offline.

Teacher : llama3.1:8b   (or any larger model you have pulled)
Student : llama3.2:3b   (or any smaller model)

Run:
    # 1. Install Ollama + pull models (one-time)
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3.1:8b
    ollama pull llama3.2:3b

    # 2. Start the Ollama server
    ollama serve &

    # 3. Run this example
    python examples/08_ollama_kd_spar.py

Swap to Qwen teachers:
    python examples/08_ollama_kd_spar.py --teacher qwen2.5:7b --student llama3.2:3b

Requirements:
    pip install chromadb sentence-transformers requests
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sara.rag.pipeline import RAGVectorStore
from sara.rag.ollama_client import (
    OLLAMA_TEACHER_MODEL,
    OLLAMA_STUDENT_MODEL,
    OLLAMA_ALT_TEACHER,
    check_ollama_running,
    ensure_model,
)
from sara.rag.ollama_pipeline import OllamaRAGPipeline
from sara.rag.ollama_kd_spar import OllamaKDSPAR

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
    "What is soft targets vs hard labels?",
    "How does the KD loss formula work?",
]

VAL_QUERIES = [
    "What is dark knowledge and why does it matter?",
    "How should a RAG assistant handle uncertainty?",
    "What makes soft targets more informative than hard labels?",
]


def main(teacher_model: str, student_model: str):
    print(f"\n{'='*55}")
    print(f"Ollama KD-SPAR Example")
    print(f"  Teacher : {teacher_model}")
    print(f"  Student : {student_model}")
    print(f"{'='*55}")

    if not check_ollama_running():
        print("\nERROR: Ollama server is not running.")
        print("Start it with:  ollama serve")
        print("Install from:   https://ollama.com/install.sh")
        return

    ensure_model(teacher_model)
    ensure_model(student_model)

    # Build shared vector store
    print("\nBuilding vector store …")
    store = RAGVectorStore(persist_path="./ollama_demo_db")
    teacher_pipe = OllamaRAGPipeline(teacher_model, store=store, auto_pull=False)
    n = teacher_pipe.ingest(CORPUS)
    print(f"Indexed {n} chunks")

    # Harvest teacher responses
    spar = OllamaKDSPAR(teacher_model, student_model, vector_store=store, auto_pull=False)
    teacher_responses = spar.harvest_teacher_responses(TRAIN_QUERIES + VAL_QUERIES)

    # Run KD-SPAR
    print(f"\nRunning KD-SPAR ({teacher_model} → {student_model}) …")
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
    print(f"\n{'='*55}")
    print(f"KD-SPAR complete")
    print(f"  Iterations : {len(history)}  |  Accepted : {accepted}")
    if history:
        print(f"  Score : {history[0].score_before:.4f} → {history[-1].score_after:.4f}")

    print(f"\nOptimised student prompt:\n{'-'*55}")
    print(final_prompt)
    print(f"{'-'*55}")

    # Test the student with the optimised prompt
    print("\nStudent response sample with optimised prompt:")
    student_pipe = OllamaRAGPipeline(
        student_model, store=store, system_prompt=final_prompt, auto_pull=False
    )
    for q in VAL_QUERIES[:2]:
        resp = student_pipe.query(q, return_context=False)
        print(f"\n  Q: {q}")
        print(f"  A: {resp.answer[:300]}…" if len(resp.answer) > 300 else f"  A: {resp.answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", default=OLLAMA_TEACHER_MODEL)
    parser.add_argument("--student", default=OLLAMA_STUDENT_MODEL)
    args = parser.parse_args()
    main(args.teacher, args.student)
