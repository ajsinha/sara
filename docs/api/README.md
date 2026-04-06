# Sara API Reference
**Sara v1.6.0** · Copyright (C) 2025 Ashutosh Sinha · AGPL-3.0

---

## Module map

```
sara/
├── core/
│   ├── losses.py        DistillationLoss, FeatureDistillationLoss, RKDLoss, ...
│   ├── utils.py         jaccard, kd_score, batch_kd_score, interpret_ab_gap,
│   │                    DEFAULT_SYSTEM_PROMPT, CITATION_RE, HEDGE_WORDS
│   └── progress.py      SaraLogger, Heartbeat, ProgressBar, phase
├── vision/
│   ├── response_based.py   ResponseBasedDistiller, VisionDistillConfig
│   ├── feature_based.py    FeatureBasedDistiller, attach_feature_hooks
│   └── attention_transfer.py  AttentionTransferDistiller
├── nlp/
│   └── bert_distillation.py  BertDistillationTrainer, BertDistillConfig, run_bert_distillation
├── advanced/
│   ├── progressive.py    ProgressiveDistiller, Stage
│   ├── mutual.py         MutualDistiller
│   ├── self_distill.py   MultiExitResNet, SelfDistillTrainer
│   └── relation_based.py RelationalKDDistiller
└── rag/
    ├── backend.py        get_pipeline, get_client, get_spar, cfg, describe  ← START HERE
    ├── pipeline.py       RAGPipeline, RAGVectorStore, AnthropicClient
    ├── ollama_client.py  OllamaClient, check_ollama_running, ensure_model
    ├── ollama_pipeline.py OllamaRAGPipeline
    ├── ollama_kd_spar.py OllamaKDSPAR, OllamaMultiTeacherKDSPAR, OllamaTeacherSpec
    ├── kd_spar.py        KDSPAR (base)
    ├── kd_spar_multi_teacher.py  MultiTeacherKDSPAR, TeacherSpec
    ├── kd_spar_adversarial.py    AdversarialKDSPAR, AdversarialQuery
    ├── kd_spar_federated.py      FederatedKDSPARServer/Client, FederatedSimulation
    ├── kd_spar_meta.py           MetaKDSPAR, Specialist, SPECIALISTS
    ├── kd_spar_enhanced.py       EnhancedKDSPAR, EnhancedConfig
    ├── evaluation.py     EquivalenceReport, run_equivalence_suite
    ├── migration.py      RAGMigration, RAGTrace
    └── prompt_opt.py     GridSearch, EvolutionaryAPO
```

---

## Key entry points

### `sara.rag.backend` — start here for most use cases

```python
from sara.rag.backend import get_pipeline, get_client, get_spar, cfg, describe

describe()           # → "Backend: ollama | Teacher: llama3.1:8b | ..."
get_pipeline("teacher", store=store)   # → OllamaRAGPipeline or RAGPipeline
get_client("student")                  # → OllamaClient or AnthropicClient
get_spar(store=store)                  # → OllamaKDSPAR or KDSPAR
cfg["teacher_model"]                   # → "llama3.1:8b"
```

### `sara.core.losses` — distillation loss functions

```python
from sara.core.losses import DistillationLoss, RKDLoss

criterion = DistillationLoss(alpha=0.5, temperature=4.0)
loss = criterion(student_logits, teacher_logits, labels)
```

### `sara.core.utils` — shared scoring helpers

```python
from sara.core.utils import kd_score, jaccard, batch_kd_score, interpret_ab_gap

kd_score("answer [Doc-1].", "answer [Doc-1].")   # → 1.0
interpret_ab_gap(0.025)   # → "strong — gap > 0.02 provides compelling evidence"
```

### `sara.core.progress` — experiment logging

```python
from sara.core.progress import SaraLogger, phase

log = SaraLogger("Ablation")
log.section("Condition A — KD-SPAR")
log.step("Harvesting teacher responses", total=40)
for i in range(40):
    log.tick(i + 1)
log.done("40 responses collected")
log.metric("KD Score", "0.386")

with phase("SPAR iteration 1/3"):
    # long computation — heartbeat prints if silent too long
    pass
```

---

## Common patterns

### Harvest teacher responses
```python
store   = RAGVectorStore()
teacher = get_pipeline("teacher", store=store)
teacher.ingest({"doc.txt": "Your document text here..."})

teacher_responses = {}
for q in queries:
    teacher_responses[q] = teacher.query(q, return_context=False).answer
```

### Run KD-SPAR
```python
spar = get_spar(store=store)
final_prompt, history = spar.run(
    train_queries=train_q,
    val_queries=val_q,
    teacher_responses=teacher_responses,
    iterations=5,
)
```

### Query with optimised prompt
```python
student = get_pipeline("student", store=store, system_prompt=final_prompt)
resp = student.query("What is knowledge distillation?")
print(resp.answer)
print(resp.citations)   # ["[Doc-1]", "[Doc-2]"]
```

---

*Sara v1.6.0 · Ashutosh Sinha · ajsinha@gmail.com · AGPL-3.0*
