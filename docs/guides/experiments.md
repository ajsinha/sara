# Ablation Experiment Guide
**Sara v1.4.0** · Copyright (C) 2025 Ashutosh Sinha · AGPL-3.0

---

## The self-knowledge hypothesis

KD-SPAR's core claim: *the student model has privileged self-knowledge
about what instructions will improve its own performance, because the
model that proposes the instruction is the same one that will execute it.*

The ablation tests this by comparing four conditions with **everything else
held constant** — same queries, same KD metric, same teacher, same
validate-and-commit gate:

| Condition | Proposer | KD Signal | Purpose |
|-----------|----------|-----------|---------|
| **A — KD-SPAR** | Student self-proposes | Yes | Our method |
| **B — External** | Teacher proposes | Yes | Same signal, different proposer |
| **C — Random** | Generic pool | No | Structural noise baseline |
| **D — No tuning** | Vanilla DEFAULT_SYSTEM | No | Absolute baseline |

The **A−B gap** is the key metric. It isolates the pure value of
student self-authorship over external proposal with identical KD signal.

---

## Running the experiment

### Prerequisites

```bash
source .venv/bin/activate
pip install -e ".[rag]"

# Ollama models (one-time)
ollama pull llama3.1:8b
ollama pull llama3.2:3b
ollama pull qwen2.5:7b    # for Config 2
ollama serve &
```

### Quick run (~20 min, good first check)

```bash
for seed in 42 123 456; do
    python experiments/kd_spar_ablation_ollama.py \
        --config 1 --iterations 3 --seed $seed
done
python experiments/collect_results.py
python experiments/results_analysis.py
```

### Publication run (~3 hours, 5 seeds × 2 configs)

```bash
for cfg in 1 2; do
  for seed in 42 123 456 789 101; do
    python experiments/kd_spar_ablation_ollama.py \
        --config $cfg --iterations 3 --seed $seed
  done
done
python experiments/collect_results.py
python experiments/patch_paper.py --output docs/paper/Sara_Knowledge_Distillation.pdf
```

---

## Model configurations

| --config | Teacher | Student | Purpose |
|----------|---------|---------|---------|
| 1 | llama3.1:8b | llama3.2:3b | Same-family, controlled |
| 2 | qwen2.5:7b | llama3.2:3b | Cross-family, generalisation |
| 3 | llama3.1:8b | qwen2.5:3b | Qwen student |

Use `--teacher` / `--student` for custom pairs:
```bash
python experiments/kd_spar_ablation_ollama.py \
    --teacher qwen2.5:7b --student llama3.2:3b --iterations 3
```

---

## Results files

After running, check `experiments/results/`:

```
experiments/results/
├── ablation_ollama_llama8b-llama3b_seed42_YYYYMMDD.json
├── ablation_ollama_llama8b-llama3b_seed123_YYYYMMDD.json
├── ...
├── aggregated_results.json    ← written by collect_results.py
└── aggregated_results.txt     ← human-readable summary
```

---

## Interpreting results

```bash
# View latest run
python experiments/results_analysis.py

# Compare multiple runs
python experiments/results_analysis.py experiments/results/ablation_*.json
```

### A−B gap interpretation

| Gap | Meaning |
|-----|---------|
| > 0.02 | **Strong** — compelling evidence, submit the paper |
| 0.01–0.02 | **Moderate** — supports the claim, add more seeds |
| 0.005–0.01 | **Suggestive** — positive trend, needs replication |
| < 0.005 | **Marginal** — within noise, try more iterations |
| Negative | Investigate — external beats self-proposed |

---

## Patching the paper

Once you have real results, replace the synthetic placeholders in §20:

```bash
python experiments/patch_paper.py --output docs/paper/Sara_Knowledge_Distillation.pdf
```

This reads `experiments/results/aggregated_results.json`, replaces Section 20
with your actual measured values, and rebuilds the PDF.

---

## Statistical significance

The experiment reports a one-sided paired t-test:
- **H₀**: E[KD(A)] ≤ E[KD(B)] (self-authorship adds no value)
- **H₁**: E[KD(A)] > E[KD(B)]
- **α** = 0.05

With 5 seeds and A−B gap > 0.01, the t-statistic typically exceeds the
critical value (t₄,₀.₀₅ = 2.132) and H₀ is rejected. Run at least 5 seeds
per configuration for a statistically sound result.

---

*Sara v1.4.0 · Ashutosh Sinha · ajsinha@gmail.com · AGPL-3.0*
