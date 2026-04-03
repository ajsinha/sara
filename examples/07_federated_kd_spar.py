"""
examples/07_federated_kd_spar.py
==================================
Federated KD-SPAR: multiple sites jointly optimise a global prompt
without sharing any raw query or response data.

This demo simulates three "hospital sites" (Site A, B, C) each with
private RAG trace data. They collaborate to optimise a shared student
prompt via the aggregation server. Only instruction strings — never
query text or model responses — cross site boundaries.

Run:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python examples/07_federated_kd_spar.py

Requirements:
    pip install anthropic chromadb sentence-transformers
"""

import os
from sara.rag.pipeline import RAGPipeline, RAGVectorStore, TEACHER_MODEL, STUDENT_MODEL
from sara.rag.kd_spar_federated import (
    FederatedSimulation,
    FederatedKDSPARClient,
    FederatedKDSPARServer,
    FederatedClientConfig,
)

# Shared corpus — represents a shared knowledge base all sites can query
CORPUS = {
    "clinical_kd.txt": """
        Knowledge distillation helps deploy AI models in clinical settings.
        Patient privacy regulations require that no raw data leaves individual hospital sites.
        Federated learning enables collaborative model improvement without data sharing.
        ChromaDB can store medical document embeddings for semantic retrieval.
        The RAG pipeline grounds model responses in retrieved clinical passages.
        Soft targets from teacher models provide richer training signal than hard labels.
        KD-SPAR allows prompt improvement without accessing model weights directly.
        Citation format [Doc-N] helps clinicians trace AI responses to source documents.
        Uncertainty hedging is critical in clinical settings to avoid overconfident claims.
        Federated KD-SPAR shares only instruction strings, preserving patient data privacy.
    """,
    "kd_methods.txt": """
        Response-based distillation uses teacher output logits as training targets.
        Feature-based distillation aligns intermediate layer representations.
        Temperature scaling controls the softness of teacher output distributions.
        Alpha parameter balances distillation loss versus cross-entropy loss.
        The T-squared scaling factor compensates for gradient magnitude reduction.
        Progressive distillation chains multiple stages to avoid capacity-gap failures.
        Online mutual learning trains two models simultaneously using peer supervision.
        Self-distillation uses multi-exit networks where deeper exits supervise shallower ones.
        Data-free distillation uses generated synthetic data when original data is unavailable.
        KD-SPAR diagnoses failure modes and proposes targeted system prompt amendments.
    """,
}

# Simulated private traces for each "hospital site"
# In production these would be real (query, teacher_response) pairs from local RAG logs
# Here we generate them from different topic areas to simulate site specialisation
SITE_TRACES = {
    "site_a": [  # Cardiology site — specialises in citation-heavy answers
        ("What is knowledge distillation?",
         "Knowledge distillation [Doc-1] transfers knowledge from a large teacher to "
         "a smaller student model [Doc-2]. The teacher's soft targets [Doc-1] contain "
         "rich inter-class similarity information called dark knowledge [Doc-2]."),
        ("How does temperature scaling work?",
         "Temperature T [Doc-1] controls the softness of the teacher's output distribution. "
         "Higher T [Doc-1] flattens the distribution, revealing inter-class similarity [Doc-2]. "
         "The T-squared factor [Doc-1] compensates for gradient magnitude reduction."),
        ("What is feature-based distillation?",
         "Feature-based distillation [Doc-1] aligns intermediate layer representations. "
         "FitNets [Doc-2] pioneered this approach using hint layers between networks. "
         "A learnable adapter [Doc-1] projects student features to teacher channel space."),
    ],
    "site_b": [  # Radiology site — specialises in step-by-step reasoning
        ("What is knowledge distillation?",
         "Let me reason through this step by step. First, we have a large teacher model "
         "trained on hard labels. Then, the teacher produces soft probability distributions "
         "over all classes. These soft targets contain richer information than hard labels. "
         "The student then learns from these soft targets to compress the teacher's knowledge. "
         "In summary: teacher → soft targets → student training → compressed model."),
        ("How does temperature scaling work?",
         "Step 1: Take the teacher's raw logits. Step 2: Divide by temperature T. "
         "Step 3: Apply softmax. At T=1 we get standard softmax (sharp). "
         "At T=4+ we get a flatter distribution that reveals how similar classes are. "
         "Multiply KL loss by T-squared to compensate for gradient reduction."),
        ("What is Federated KD-SPAR?",
         "Following the logic step by step: First, each site diagnoses its own failures "
         "locally. Second, sites propose instruction strings (no data shared). Third, "
         "the server aggregates proposals semantically. Fourth, the server validates and "
         "broadcasts the improved global prompt. The key insight is that only text "
         "instructions cross boundaries, ensuring patient data privacy."),
    ],
    "site_c": [  # Oncology site — specialises in uncertainty-calibrated answers
        ("What is knowledge distillation?",
         "Knowledge distillation appears to be a model compression technique where "
         "a smaller student model may learn from a larger teacher model. The teacher "
         "might use temperature-scaled outputs, though the optimal temperature could "
         "vary by task. Results suggest students can possibly recover much of the "
         "teacher's accuracy, though this may depend on capacity ratio."),
        ("What is RAG?",
         "Retrieval-Augmented Generation might be defined as a technique that appears "
         "to combine document retrieval with language model generation. ChromaDB could "
         "serve as the vector database, though other options may exist. The quality "
         "of answers appears to depend heavily on retrieval accuracy."),
        ("What is KD-SPAR?",
         "KD-SPAR might be a novel paradigm where the student model appears to diagnose "
         "its own failures and may propose prompt amendments. Whether this self-knowledge "
         "is reliable remains uncertain, though early results could be promising. "
         "The federated variant possibly addresses privacy concerns by sharing only "
         "instruction strings, though the privacy guarantees may vary by implementation."),
    ],
}


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY before running.")
        return

    print("="*60)
    print("Federated KD-SPAR Demo")
    print("Simulating 3 hospital sites with private RAG traces")
    print("="*60)

    # Shared vector store (simulates shared knowledge base)
    print("\nSetting up shared vector store …")
    store = RAGVectorStore(persist_path="./fed_spar_demo_db")
    base_pipe = RAGPipeline(TEACHER_MODEL, store=store)
    base_pipe.ingest(CORPUS)
    print(f"Indexed {store.count} chunks")

    print("\nCreating 3 federated client sites …")
    all_traces = []
    for site_name, traces in SITE_TRACES.items():
        for q, r in traces:
            all_traces.append((q, r))
        print(f"  {site_name}: {len(traces)} private traces")

    # Build the federation using the simulation harness
    sim = FederatedSimulation(
        n_clients     = 3,
        all_traces    = all_traces,
        val_fraction  = 0.2,
        student_model = STUDENT_MODEL,
        vector_store  = store,
    )
    server = sim.build_server(threshold=0.002, regression_tol=0.02)

    print(f"\nFederation structure:")
    print(f"  Clients           : {len(server.clients)}")
    print(f"  Server val queries: {len(server.val_queries)}")
    for client in server.clients:
        print(f"  {client.config.client_id}: {len(client.local_traces)} traces (private)")

    print(f"\nPrivacy guarantee:")
    print(f"  ✓ No query text leaves any client site")
    print(f"  ✓ No model responses leave any client site")
    print(f"  ✓ Only instruction strings cross boundaries")
    print(f"  ✓ Server validates on its own data, not client data")

    print(f"\nStarting federated optimisation rounds …")
    final_prompt, history = server.run(
        rounds      = 5,
        min_clients = 2,    # need at least 2 sites to participate
        log_path    = "./fed_spar_log.jsonl",
    )

    accepted = sum(1 for r in history if r.accepted)
    print(f"\n{'='*60}")
    print(f"Federated KD-SPAR complete")
    print(f"  Total rounds     : {len(history)}")
    print(f"  Accepted rounds  : {accepted}")
    print(f"  Total proposals  : {sum(r.total_proposals for r in history)}")
    if history:
        print(f"  Score trajectory : {history[0].score_before:.4f} → {history[-1].score_after:.4f}")

    print(f"\nPer-round summary:")
    for r in history:
        status = "✓" if r.accepted else "✗"
        print(f"  Round {r.round_number}: {status} Δ={r.delta:+.4f}  "
              f"({r.total_proposals} proposals from {len(r.clients_participated)} sites)")
        if r.selected_instrs:
            for ins in r.selected_instrs:
                print(f"    → {ins[:70]}")

    print(f"\nFinal global prompt ({len(final_prompt)} chars):\n{'-'*60}")
    print(final_prompt)
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
