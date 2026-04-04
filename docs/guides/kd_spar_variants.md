# KD-SPAR Variants Guide
**Sara v1.1.0** · Copyright (C) 2025 Ashutosh Sinha · AGPL-3.0

---

## Overview

KD-SPAR (Knowledge Distillation via Student Prompt Auto-Rewriting) has four variants.
All share the same four-phase loop — diagnose → self-interview → aggregate → validate.
They differ in who proposes, what queries are used, and how validation works.

| Variant | When to use | Key constraint |
|---------|-------------|----------------|
| Base | Single teacher, standard queries | None |
| Multi-Teacher | Committee of specialist models | Non-regression across all teachers |
| Adversarial | Long-tail robustness needed | Dual-objective: hard queries + no standard regression |
| Federated | Multi-site, private data | No raw data leaves any site |

---

## Base KD-SPAR

```python
from sara.rag.backend import get_spar
from sara.rag.pipeline import RAGVectorStore

store = RAGVectorStore()
spar  = get_spar(store=store)   # reads teacher/student from configs/backend.yaml

final_prompt, history = spar.run(
    train_queries     = train_q,
    val_queries       = val_q,
    teacher_responses = teacher_resps,   # {query: teacher_answer}
    iterations        = 5,
    threshold         = 0.003,
    n_proposals       = 4,
    top_k             = 3,
)
```

---

## Multi-Teacher KD-SPAR

Align one student to N specialist teachers simultaneously.
The worst-aligned teacher drives the self-interview each iteration.
Validation accepts only when the primary teacher improves AND no secondary
teacher regresses beyond `regression_tol`.

```python
from sara.rag.kd_spar_multi_teacher import MultiTeacherKDSPAR, TeacherSpec
from sara.rag.ollama_kd_spar import OllamaMultiTeacherKDSPAR, OllamaTeacherSpec

# Ollama version (FOSS)
specs = [
    OllamaTeacherSpec(
        name="citation_expert", model_id="llama3.1:8b",
        system_prompt="Always cite every claim with [Doc-N].",
        weight=2.0, is_primary=True,
    ),
    OllamaTeacherSpec(
        name="reasoning_expert", model_id="llama3.1:8b",
        system_prompt="Reason step by step before answering.",
        weight=1.0,
    ),
]
spar = OllamaMultiTeacherKDSPAR(
    student_model="llama3.2:3b",
    teacher_specs=specs,
    vector_store=store,
    regression_tol=0.02,
)
teacher_response_sets = spar.harvest_all_teacher_responses(queries)
final_prompt, history = spar.run_multi(
    train_queries=train_q,
    val_queries=val_q,
    teacher_response_sets=teacher_response_sets,
    iterations=8,
)
```

---

## Adversarial KD-SPAR

Focus the optimisation loop on hard examples to build robustness.
Hard examples come from two sources:
- **Gap-mined**: bottom decile of KD scores in your query log
- **Generated**: teacher produces adversarial questions about your topic

```python
from sara.rag.kd_spar_adversarial import AdversarialKDSPAR

spar = AdversarialKDSPAR(
    teacher_model="llama3.1:8b",   # or read from cfg
    student_model="llama3.2:3b",
    vector_store=store,
    adversarial_topics=["knowledge distillation", "RAG retrieval"],
    n_generated_per_topic=10,
    hardness_percentile=0.25,   # bottom 25% = hard
    dual_threshold=0.005,
    standard_regression=0.02,
)

# Build hard query set
hard_queries = spar.build_hard_query_set(production_queries, teacher_responses)

# Run adversarial loop
final_prompt, history = spar.run_adversarial(
    adversarial_queries=hard_queries,
    standard_queries=production_queries,
    teacher_responses=teacher_responses,
    iterations=8,
)
```

---

## Federated KD-SPAR

Multiple sites each hold private RAG data. They jointly optimise a shared
global prompt. Only instruction strings — never query text or responses —
cross site boundaries.

```python
from sara.rag.kd_spar_federated import (
    FederatedSimulation,
    FederatedKDSPARClient, FederatedKDSPARServer, FederatedClientConfig,
)

# Simulate 3 sites from a shared trace pool (for development)
sim = FederatedSimulation(
    n_clients=3,
    all_traces=[(q, teacher_response) for q, teacher_response in trace_pairs],
    student_model="llama3.2:3b",
    vector_store=store,
)
server = sim.build_server(threshold=0.003)
final_prompt, history = server.run(rounds=10)

# Production: use real distributed clients
cfg_a = FederatedClientConfig(client_id="site_hospital_a")
client_a = FederatedKDSPARClient(cfg_a, site_a_traces, site_a_store)
# ... more clients
server = FederatedKDSPARServer(
    clients=[client_a, client_b],
    server_val_queries=val_q,
    server_val_responses=val_resps,
    student_model="llama3.2:3b",
)
final_prompt, history = server.run(rounds=10)
```

---

## Interpreting results

```python
accepted = sum(1 for s in history if s.accepted)
print(f"Iterations: {len(history)}  Accepted: {accepted}")
print(f"Score: {history[0].score_before:.4f} → {history[-1].score_after:.4f}")

# Each SPARIteration has:
for it in history:
    print(f"  it={it.iteration}  Δ={it.delta:+.4f}  accepted={it.accepted}")
    print(f"  selected instructions: {it.selected}")
```

---

*Sara v1.1.0 · Ashutosh Sinha · ajsinha@gmail.com · AGPL-3.0*
