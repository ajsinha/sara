![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL_3.0-blue.svg)  ![Version](https://img.shields.io/badge/version-1.8.3-green.svg)

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
**Python:** 3.10+  **License:** AGPL-3.0

---

## What Sara does

| Module | Techniques |
|--------|-----------|
| `sara.core` | `DistillationLoss`, `FeatureDistillationLoss`, `RKDLoss`, profiler, hyperparams, shared scoring, `SaraLogger` / progress |
| `sara.vision` | Response-based, FitNets feature-based, attention transfer |
| `sara.nlp` | BERT → DistilBERT (3-term loss, HuggingFace Trainer) |
| `sara.advanced` | Progressive, mutual learning, self-distillation, relation-based |
| `sara.rag` | RAG pipeline (Anthropic + Ollama), model migration, equivalence suite |
| `sara.rag.backend` | Provider-agnostic factory — switch Anthropic ↔ Ollama via env/YAML |
| `sara.rag.kd_spar` | **KD-SPAR** — base self-rewriting loop |
| `sara.rag.kd_spar_multi_teacher` | Multi-teacher variant |
| `sara.rag.kd_spar_adversarial` | Adversarial variant |
| `sara.rag.kd_spar_federated` | Federated variant (privacy-preserving) |
| `sara.rag.kd_spar_meta` | **MetaKDSPAR** — metaprompting-enhanced diagnosis |
| `sara.rag.kd_spar_enhanced` | **Enhanced KD-SPAR** — 7 algorithmic improvements |

---

## Install

```bash
cd sara
pip install -e .              # core only (no heavy dependencies)
pip install -e ".[vision]"    # add vision (PyTorch, torchvision)
pip install -e ".[nlp]"       # add NLP (transformers, datasets)
pip install -e ".[rag]"       # add RAG (Anthropic API + Ollama)
pip install -e ".[all]"       # everything
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
pytest                    # 246 tests, all offline
pytest tests/test_losses.py tests/test_rag.py   # core + RAG only
```

---

## Running the ablation experiment

Use the provided shell script — it handles everything from Ollama installation
through results analysis:

```bash
# Full publication run (default)
bash setup_and_run.sh

# Quick first check (~20 min, 3 seeds, Config 1 only)
bash setup_and_run.sh --quick

# Already have Ollama + venv set up — skip to experiments
bash setup_and_run.sh --skip-setup

# Custom config
bash setup_and_run.sh --config 2 --iterations 3 --seeds "42 123 456"

# Help
bash setup_and_run.sh --help
```

See `example_run.sh` at the project root for a copy-paste reference of all common invocations.

### Understanding the experiment parameters

When you run the script you will see:

```
[sara] Configs     : 1 2
[sara] Seeds       : 42 123 456 789 101
[sara] Iterations  : 3
```

**`Configs: 1 2`** — two teacher→student model pairs run back-to-back:

| Config | Teacher | Student | Purpose |
|--------|---------|---------|---------|
| 1 | llama3.1:8b | llama3.2:3b | Same-family — clean controlled test of the KD signal |
| 2 | qwen2.5:7b | llama3.2:3b | Cross-family — tests whether the A−B gap generalises across model families |

Running both configs matters for the paper. If the A−B gap is positive in Config 1
(same-family) *and* Config 2 (cross-family), the self-knowledge mechanism is demonstrably
general — not an artefact of how Llama was trained. Reviewers will ask exactly this.

**`Seeds: 42 123 456 789 101`** — five independent runs of the full 4-condition ablation
per config. Each seed gives one data point for the A−B gap. Running 5 seeds lets you:

- Report results as **mean ± std** (e.g. `0.386 ± 0.006`) rather than a single number
- Run a proper **paired t-test** (df=4, critical value t₄,₀.₀₅ = 2.132) to claim
  statistical significance
- Reproduce any single run exactly by passing `--seed 42` again

Start with `--seeds "42 123 456"` (3 seeds, ~40% less time) to check the direction of
the A−B gap before committing to the full 5.

**`Iterations: 3`** — each KD-SPAR iteration is one full diagnosis → self-interview →
aggregate → validate cycle. Three iterations gives the student three chances to propose
and commit prompt improvements per condition.

- More iterations raise both A and B scores, but the **A−B gap** — the number that
  matters — stabilises after 3
- If you see `✗ REVERTED` on all 3 iterations for a seed, increase to `--iterations 5`
- 3 iterations is the recommended minimum for publication; 5 gives more signal at ~70%
  extra runtime

**Total runs = configs × seeds = 2 × 5 = 10 ablations.** Each writes a JSON to
`experiments/results/` immediately on completion, so the script can be interrupted
and restarted without losing finished work.

### The key metric: A−B gap

After running, `collect_results.py` prints a table like:

```
Cond  KD Score          Δ vs D
A     0.386 ± 0.006     +0.045    ← KD-SPAR (self-proposed)
B     0.360 ± 0.004     +0.019    ← Externally proposed
C     0.340 ± 0.003     −0.001    ← Random
D     0.342 ± 0.003      —        ← No tuning

A−B gap: +0.026  →  STRONG evidence for self-knowledge claim
```

| A−B gap | Meaning |
|---------|---------|
| > 0.02 | **Strong** — submit the paper |
| 0.01–0.02 | **Moderate** — add more seeds or iterations |
| < 0.01 | **Marginal** — investigate or increase iterations |

---

## Project structure

```
sara/
├── sara/              Python package (import as 'from sara...')
│   ├── core/          Losses, utils, progress logging, shared scoring
│   ├── vision/        Vision distillation trainers
│   ├── nlp/           BERT-family distillation
│   ├── advanced/      Progressive, mutual, self-distil, RKD
│   └── rag/           RAG pipeline + all 4 KD-SPAR variants + Ollama backend
├── docs/              All documentation + research paper
│   ├── paper/         PDF builder (sara_helpers + sara_story) + PDF
│   ├── guides/        Backend config, KD-SPAR variants, experiments guide
│   └── api/           API reference
├── experiments/       Ablation scripts + results
├── tests/             246 tests, all offline
├── examples/          10 runnable end-to-end examples
├── configs/           YAML configs (backend.yaml, kd_spar.yaml, ...)
├── setup_and_run.sh   Complete local experimentation script
├── example_run.sh     Quick-reference command examples
└── LICENSE            AGPL-3.0
```

---

*Sara (सार) v1.8.3 · Ashutosh Sinha · ajsinha@gmail.com · AGPL-3.0*

