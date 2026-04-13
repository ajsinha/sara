# Sara (सार) — Quickstart Guide
**Version 1.8.3** · Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com) · AGPL-3.0

---

## What is Sara?

Sara (सार, *sāra*) is Sanskrit for *quintessence* — the refined essence extracted from a
larger whole. This project is named Sara because knowledge distillation is, literally,
extracting the *sāra* of a large teacher model into a smaller student.

The centrepiece is **KD-SPAR** (Knowledge Distillation via Student Prompt Auto-Rewriting)
— a technique where the student model diagnoses its own failures and rewrites its own
system prompt to align more closely with the teacher. No weights are needed; the student
is both the subject and the author of its own improvement.

---

## Installation (5 minutes)

```bash
# Clone or unzip
cd ~/PycharmProjects/sara

# Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install — choose what you need:
pip install -e ".[rag]"      # RAG + KD-SPAR (recommended, no GPU needed)
pip install -e ".[vision]"   # Vision distillation (needs GPU)
pip install -e ".[nlp]"      # BERT distillation (needs GPU)
pip install -e ".[all]"      # Everything
```

---

## Choosing your backend

Sara works with **Ollama (FOSS, default)** or **Anthropic API**.
Edit one file to switch:

**`configs/backend.yaml`** — the single source of truth:
```yaml
# FOSS / local (default — no API key needed)
backend: ollama
teacher_model: llama3.1:8b
student_model: llama3.2:3b

# — or switch to Anthropic —
# backend: anthropic
# teacher_model: claude-3-5-sonnet-20241022
# student_model: claude-sonnet-4-5-20250929
```

Or use environment variables (overrides the config file):
```bash
export SARA_BACKEND=ollama
export SARA_TEACHER_MODEL=qwen2.5:7b    # any Ollama model
export SARA_STUDENT_MODEL=llama3.2:3b
```

---

## Ollama setup (FOSS path — recommended)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models (one-time)
ollama pull llama3.1:8b    # 4.7 GB — teacher
ollama pull llama3.2:3b    # 2.0 GB — student
ollama pull qwen2.5:7b     # 4.4 GB — optional cross-family teacher

# Start server (Pop!_OS auto-starts via systemd; or manually)
ollama serve &
```

---

## Run your first example

```bash
source .venv/bin/activate

# Base KD-SPAR (uses backend from configs/backend.yaml)
python examples/04_kd_spar.py

# Multi-teacher
python examples/05_multi_teacher_kd_spar.py

# Ollama-specific (always uses Ollama regardless of config)
python examples/08_ollama_kd_spar.py
```

---

## Run the ablation experiment

This is the critical experiment that tests the self-knowledge hypothesis
(see §20 of the paper).

```bash
# Full run: 5 seeds × 2 configs (~3 hours on RTX 3070 Ti)
for cfg in 1 2; do
  for seed in 42 123 456 789 101; do
    python experiments/kd_spar_ablation_ollama.py \
      --config $cfg --iterations 3 --seed $seed
  done
done

# Aggregate and patch paper with real numbers
python experiments/collect_results.py
python experiments/patch_paper.py --output docs/paper/Sara_Knowledge_Distillation.pdf

# View results
python experiments/results_analysis.py
```

Quick run (3 seeds, ~20 min):
```bash
for seed in 42 123 456; do
  python experiments/kd_spar_ablation_ollama.py --config 1 --iterations 3 --seed $seed
done
python experiments/collect_results.py && python experiments/results_analysis.py
```

---

## Run tests

```bash
pytest           # all 246 tests, fully offline, no API key needed
pytest tests/test_rag.py       # RAG + KD-SPAR only
pytest tests/test_ollama.py    # Ollama backend
```

---

## Project layout

```
sara/
├── sara/                  Python package  (import as 'from sara...')
│   ├── core/              Losses, progress logging, shared scoring helpers
│   ├── vision/            Vision distillation
│   ├── nlp/               BERT-family distillation
│   ├── advanced/          Progressive, mutual, self-distil, RKD
│   └── rag/               RAG pipeline + all 4 KD-SPAR variants + backends
├── docs/                  All documentation
│   ├── paper/             PDF paper builder (sara_helpers.py + sara_story.py)
│   ├── guides/            Extended guides
│   └── api/               API reference
├── examples/              10 runnable end-to-end examples
├── experiments/           Ablation scripts + setup guide
├── tests/                 246 tests, all offline
├── configs/               backend.yaml + hyperparameter configs
├── LICENSE                AGPL-3.0
└── README.md
```

---

## The key metric: A−B gap

After running experiments, the number that matters is the **A−B gap**:

| Value | Meaning |
|-------|---------|
| > 0.02 | Strong — compelling evidence for the self-knowledge claim |
| 0.01–0.02 | Moderate — supports the claim |
| < 0.01 | Run more seeds or iterations |

---

*Sara (सार) v1.8.3 · Ashutosh Sinha · ajsinha@gmail.com · AGPL-3.0*
