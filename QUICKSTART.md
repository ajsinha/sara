# Quickstart Guide

Get up and running in 5 minutes.

---

## Step 1 — Install

```bash
cd knowledge_distillation

# Core vision + NLP
pip install -e ".[nlp]"

# Add RAG support
pip install -e ".[rag]"

# Set your Anthropic key (RAG only)
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Step 2 — Run your first distillation

### Option A: Vision (no API key needed)

Distil a ResNet-50 teacher into a MobileNetV2 student on CIFAR-10:

```bash
python examples/01_response_based_kd.py
```

Expected output:
```
Teacher accuracy: 0.9340
Training response-based distillation ...
Epoch 001/30  loss=1.2341  val_acc=0.8123  best=0.8123
...
Epoch 030/30  loss=0.4521  val_acc=0.9178  best=0.9178
Speedup: 3.8x   Compression: 4.1x
```

### Option B: NLP (no API key needed)

Distil BERT-base into DistilBERT on SST-2:

```bash
python examples/02_bert_distillation.py
```

### Option C: RAG migration (API key required)

Migrate from `claude-3-5-sonnet` (teacher) to `claude-sonnet-4-5` (student):

```bash
python examples/03_rag_migration.py
```

### Option D: KD-SPAR (API key required)

Run the student prompt auto-rewriting loop:

```bash
python examples/04_kd_spar.py
```

---

## Step 3 — Run all tests

```bash
pytest                         # all 70+ tests
pytest tests/test_losses.py    # just loss functions (fast, <5s)
pytest -k "not rag"            # everything except RAG
```

---

## Step 4 — Customise via config

Edit `configs/vision_cifar10.yaml`:
```yaml
temperature: 4.0
alpha: 0.6
epochs: 30
batch_size: 128
learning_rate: 0.001
```

Then pass it to any trainer:
```python
from kd.core.utils import load_config
from kd.vision.response_based import ResponseBasedDistiller

cfg = load_config("configs/vision_cifar10.yaml")
distiller = ResponseBasedDistiller.from_config(teacher, student, cfg)
```

---

## Common issues

| Problem | Fix |
|---------|-----|
| `CUDA out of memory` | Reduce `batch_size` in config |
| `ANTHROPIC_API_KEY not set` | `export ANTHROPIC_API_KEY="sk-ant-..."` |
| `No module named 'chromadb'` | `pip install -e ".[rag]"` |
| `No module named 'transformers'` | `pip install -e ".[nlp]"` |
| Tests fail on `import torch` | `pip install torch torchvision` |

---

## Step 5 — Run the KD-SPAR ablation experiment (OryxPro / any machine)

This is the critical experiment needed to support publication — it tests whether
student self-proposed instructions outperform external and random baselines.

```bash
# Quick sanity check (~10 min, ~$0.30)
python experiments/kd_spar_ablation.py --quick --iterations 2

# Full run (~35 min, ~$0.80)
python experiments/kd_spar_ablation.py --iterations 3

# Publication run — 3 seeds for mean ± std (~90 min, ~$2.50)
for seed in 42 123 777; do
    python experiments/kd_spar_ablation.py --iterations 3 --seed $seed
done

# Analyse and compare
python experiments/results_analysis.py
python experiments/results_analysis.py experiments/results/ablation_*.json
```

See `experiments/ORYXPRO_SETUP.md` for a complete step-by-step setup guide
specific to Pop!_OS / OryxPro including GPU notes, cost estimates, and
troubleshooting.

The key number: **A−B gap** (KD-SPAR score minus externally-proposed score).
- > 0.02 = compelling evidence for the self-knowledge claim
- 0.01–0.02 = moderate support
- < 0.01 = needs more iterations or a larger query set
