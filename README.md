# Sara (सार)

**sara** — from Sanskrit सार (sāra), meaning *essence*, *quintessence*, *the best and most
refined part of something*. In Vedic philosophy it denotes the core substance distilled from
a larger whole: Ayurveda speaks of *rasa-sāra* (essence of nutrition); poetry theory of
*kāvya-sāra* (the distilled meaning of verse); and Vedanta of *param-sāra* (the ultimate
essence of reality).

This project is named Sara because **knowledge distillation is, literally, the extraction of
sāra** — the refined essence of a large teacher model, compressed into a smaller student without
losing the core meaning. KD-SPAR takes this further: the student discovers the *sāra* of its own
failures and uses that self-knowledge to improve. The Federated variant shares only the *sāra*
of insights (instruction strings) across sites, never the raw data.

---

**Author:** Ashutosh Sinha · ajsinha@gmail.com  
**Python:** 3.10+  **License:** MIT

---

## What Sara does

| Module | Techniques |
|--------|-----------|
| `sara.core` | `DistillationLoss`, `FeatureDistillationLoss`, `RKDLoss`, profiler, hyperparams, shared scoring |
| `sara.vision` | Response-based, FitNets feature-based, attention transfer |
| `sara.nlp` | BERT → DistilBERT (3-term loss, HuggingFace Trainer) |
| `sara.advanced` | Progressive, mutual learning, self-distillation, relation-based |
| `sara.rag` | RAG pipeline (Anthropic + Ollama), model migration, equivalence suite |
| `sara.rag.kd_spar` | **SARA** — base self-rewriting loop |
| `sara.rag.kd_spar_multi_teacher` | Multi-teacher variant |
| `sara.rag.kd_spar_adversarial` | Adversarial variant |
| `sara.rag.kd_spar_federated` | Federated variant (privacy-preserving) |

---

## Install

```bash
cd sara
pip install -e .           # vision + core only
pip install -e ".[nlp]"    # add NLP
pip install -e ".[rag]"    # add RAG (Anthropic API)
pip install -e ".[all]"    # everything
```

For local Ollama backend (free, offline):
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b && ollama pull llama3.2:3b
pip install -e ".[rag]"   # requests is included
```

---

## Quick start

```python
# Vision distillation
from sara.vision.response_based import ResponseBasedDistiller, VisionDistillConfig
distiller = ResponseBasedDistiller(teacher, student, config=VisionDistillConfig())
distiller.train(train_loader, val_loader)

# RAG KD-SPAR (Anthropic)
from sara.rag.kd_spar import KDSPAR
spar = KDSPAR(teacher_model="claude-3-5-sonnet-20241022",
              student_model="claude-sonnet-4-5-20250929", vector_store=store)
final_prompt, history = spar.run(train_q, val_q, teacher_responses)

# RAG KD-SPAR (local Ollama — zero cost)
from sara.rag.ollama_kd_spar import OllamaKDSPAR
spar = OllamaKDSPAR("llama3.1:8b", "llama3.2:3b", vector_store=store)
final_prompt, history = spar.run(train_q, val_q, teacher_responses)
```

---

## Running tests

```bash
pytest                    # 216 tests, all offline
pytest tests/test_losses.py tests/test_rag.py   # core + RAG only
```

---

## Ablation experiment (OryxPro)

See `experiments/ORYXPRO_SETUP.md` for full instructions. Short version:

```bash
# Run 2 configs × 5 seeds (recommended)
for cfg in 1 2; do
  for seed in 42 123 456 789 101; do
    python experiments/kd_spar_ablation_ollama.py --config $cfg --iterations 3 --seed $seed
  done
done

# Aggregate and patch paper
python experiments/collect_results.py
python experiments/patch_paper.py
```

The **A−B gap** (SARA score minus externally-proposed score) is the key publication metric.
A gap > 0.02 across both llama and qwen configurations constitutes strong evidence for
the self-knowledge hypothesis.

---

## Project structure

```
sara/
├── sara/              Python package (import as 'from sara...')
│   ├── core/          Losses, utils, shared scoring helpers
│   ├── vision/        Vision distillation trainers
│   ├── nlp/           BERT-family distillation
│   ├── advanced/      Progressive, mutual, self-distil, RKD
│   └── rag/           RAG pipeline + all 4 KD-SPAR variants + Ollama backend
├── paper/             PDF builder (sara_helpers.py + sara_story.py)
├── experiments/       Ablation scripts + OryxPro guide
├── tests/             216 tests, all offline
├── examples/          8 runnable end-to-end examples
└── configs/           YAML hyperparameter configs
```
