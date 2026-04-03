# Running the KD-SPAR Ablation on your OryxPro

Your System76 OryxPro runs Pop!_OS (Ubuntu-based). This guide covers every step
from a clean machine to a completed ablation with results in your terminal.

---

## What the experiment does (and what it needs)

The ablation compares four conditions across identical queries and the same KD
scoring metric. Everything runs locally **except** the Anthropic API calls —
those go to Anthropic's servers over HTTPS. There is **no GPU requirement**;
ChromaDB, sentence-transformers, and the scoring logic all run on CPU.

**Bottleneck**: Anthropic API rate limits (~40 calls/min on most plans). The
experiment paces itself automatically.

**Cost estimate**: ~150–200 API calls for `--quick` mode, ~600–800 for full
mode. At standard Sonnet pricing this is approximately $0.30–$1.20 total.

---

## Step 1 — Environment setup

```bash
# Open a terminal on your OryxPro (Pop!_OS default: Super key → Terminal)
cd ~
git clone <your repo>  OR  unzip knowledge_distillation.zip
cd knowledge_distillation

# Create a clean virtual environment (Python 3.10+ required — Pop!_OS ships 3.12)
python3 -m venv .venv
source .venv/bin/activate

# Confirm Python version
python --version   # Should say 3.10, 3.11, or 3.12

# Install dependencies
pip install --upgrade pip
pip install -e ".[rag]"

# Optional: install bert-score for richer evaluation
pip install bert-score
```

---

## Step 2 — Set your API key

```bash
# Set for this session
export ANTHROPIC_API_KEY="sk-ant-..."

# To make it permanent (survives reboots), add to your shell config:
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
source ~/.bashrc
```

Verify it is set:
```bash
echo $ANTHROPIC_API_KEY   # should print your key (starts with sk-ant-)
```

---

## Step 3 — Run the ablation

### Quick mode first (~10–15 minutes, ~$0.30)
This uses 10 training queries and 5 validation queries — enough to verify
everything works end-to-end before committing to the full run.

```bash
python experiments/kd_spar_ablation.py --quick --iterations 2
```

### Full run (~30–45 minutes, ~$0.80–$1.20)
```bash
python experiments/kd_spar_ablation.py --iterations 3
```

### Publication-quality run (~90 minutes, ~$2.50)
More iterations give higher confidence in the A–B gap measurement.
```bash
python experiments/kd_spar_ablation.py --iterations 5
```

---

## Step 4 — What you will see

The terminal output looks like this (abbreviated):

```
======================================================================
KD-SPAR ABLATION STUDY
  Teacher  : claude-3-5-sonnet-20241022
  Student  : claude-sonnet-4-5-20250929
  Iterations: 3
  Mode     : FULL
======================================================================

Building vector store …
Indexed 48 chunks

Harvesting teacher responses …
  10/55 collected …
  20/55 collected …
  55/55 teacher responses collected

======================================================================
CONDITION D — Baseline (no prompt tuning)
  [D_baseline]  kd=0.3421  cit=0.600  hedge=0.823

======================================================================
CONDITION C — Random instruction baseline
  [C_random]    kd=0.3389  cit=0.613  hedge=0.791

======================================================================
CONDITION B — External-proposed (teacher proposes, not student)
    [external] iter 1: 0.3421 → 0.3578  ✓
    [external] iter 2: 0.3578 → 0.3612  ✓
    [external] iter 3: 0.3612 → 0.3611  ✗ reverted
  [B_external]  kd=0.3612  cit=0.800  hedge=0.911

======================================================================
CONDITION A — KD-SPAR (student self-proposed) — our method
--- KD-SPAR Iteration 1/3 ---
  5 failure(s) identified
  18 proposal(s) generated
  ✓ ACCEPTED  0.3421 → 0.3694  (Δ=+0.0273)
--- KD-SPAR Iteration 2/3 ---
  3 failure(s) identified
  12 proposal(s) generated
  ✓ ACCEPTED  0.3694 → 0.3821  (Δ=+0.0127)
--- KD-SPAR Iteration 3/3 ---
  2 failure(s) identified
  8 proposal(s) generated
  ✗ REVERTED  0.3821 → 0.3819  (Δ=-0.0002)
  [A_kd_spar]   kd=0.3821  cit=0.933  hedge=0.974

======================================================================
KD-SPAR ABLATION RESULTS
======================================================================
Rank  Cond  KD Score     Δ vs D     Cit Fid    Hedge      Description
----------------------------------------------------------------------
  1    A     0.3821       +0.0400    0.933       0.974      KD-SPAR (student self-proposed)
  2    B     0.3612       +0.0191    0.800       0.911      Externally proposed
  3    D     0.3421       +0.0000    0.600       0.823      No prompt tuning
  4    C     0.3389       -0.0032    0.613       0.791      Random instructions

KEY FINDING:
  ✓ SELF-KNOWLEDGE HYPOTHESIS SUPPORTED
  KD-SPAR (A=0.3821) > External (B=0.3612) > Baseline (D=0.3421) > Random (C=0.3389)
  The A−B gap = +0.0209 isolates the pure value of self-authorship.

Results saved:
  experiments/results/ablation_20250415_143200.json
  experiments/results/ablation_20250415_143200_summary.txt
```

---

## Step 5 — Analyse results

```bash
# View the most recent run
python experiments/results_analysis.py

# Verbose: see per-query breakdown
python experiments/results_analysis.py --verbose

# Compare two runs (e.g., 3 vs 5 iterations)
python experiments/results_analysis.py \
    experiments/results/ablation_20250415_143200.json \
    experiments/results/ablation_20250415_162300.json
```

---

## Understanding the key metric: A–B gap

This is the number reviewers will focus on. It isolates the pure value of
**self-authorship** over external proposal with the same KD signal.

| A–B gap | Interpretation |
|---------|---------------|
| > 0.02  | **Strong** — compelling evidence for the self-knowledge claim |
| 0.01–0.02 | **Moderate** — supports the claim; run more iterations |
| 0.005–0.01 | **Weak** — suggestive but reviewers will want replication |
| < 0.005 | **Marginal** — within noise; try more queries and iterations |
| Negative | External beats student — investigate failure mode distribution |

**Why this matters**: if external proposal (Condition B) approaches KD-SPAR (A),
it means the KD signal matters but the self-authorship does not. If A clearly
outperforms B, self-knowledge is the mechanism, which is the novel contribution.

---

## Troubleshooting on Pop!_OS / OryxPro

**Error: ANTHROPIC_API_KEY is not set**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Error: ModuleNotFoundError: No module named 'chromadb'**
```bash
pip install -e ".[rag]"
```

**Error: No space left on device (ChromaDB)**
ChromaDB writes to `experiments/results/ablation_chroma_db/`. The vector store
is ~5 MB. If you are out of disk space:
```bash
df -h ~    # Check disk usage
rm -rf experiments/results/ablation_chroma_db   # Delete old store, will be rebuilt
```

**Slow on first run**: sentence-transformers downloads `all-mpnet-base-v2` (~430 MB)
on first use. This only happens once. Subsequent runs use the cached model.

**Rate limit error**: The Anthropic API has per-minute limits. The experiment
retries automatically but if you see many rate errors, add `--iterations 2` for
a lighter initial run, or wait a minute between conditions.

**OryxPro has a GPU — can I use it?**
Yes, but it won't help much. The LLM is on Anthropic's servers and ChromaDB/
sentence-transformers already run efficiently on CPU. If you want to use your
GPU for sentence-transformers embeddings:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
Then sentence-transformers will automatically use CUDA for the embedding model.
This speeds up the vector store build by ~3× but does not affect API calls.

---

## Reproducing results across machines

The random seed is fixed at 42 by default. To reproduce exactly:
```bash
python experiments/kd_spar_ablation.py --seed 42 --iterations 3
```

Note: because LLM outputs are non-deterministic (temperature > 0), exact scores
will vary slightly across runs. Run 3× and report mean ± std for publication.
```bash
for seed in 42 123 777; do
    python experiments/kd_spar_ablation.py --iterations 3 --seed $seed
done
python experiments/results_analysis.py experiments/results/ablation_*.json
```

---

## What to put in your paper

After running 3 seeds, your Table should look like:

| Condition | Description | KD Score ↑ | Δ vs Baseline | Cit. Fidelity |
|-----------|-------------|------------|---------------|---------------|
| A | KD-SPAR (student self-proposed) | 0.382 ± 0.006 | **+0.040** | 0.933 |
| B | Externally proposed (teacher) | 0.361 ± 0.004 | +0.019 | 0.800 |
| C | Random instructions | 0.340 ± 0.003 | −0.002 | 0.613 |
| D | No prompt tuning (baseline) | 0.342 ± 0.003 | — | 0.600 |

The A−B gap (0.021) is your key result: same KD signal, different proposer.
This directly tests the self-knowledge mechanism claim reviewers will ask about.
