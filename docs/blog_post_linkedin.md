# What If Your AI Could Write Its Own Instructions? That's What KD-SPAR Does.

*By Ashutosh Sinha · April 2025*

---

I've been working on a problem that every ML engineer deploying LLMs has faced: **how do you make a smaller, cheaper model behave like a bigger, better one — when you can't touch the weights?**

The standard knowledge distillation playbook (Hinton et al., 2015) works beautifully when you have gradient access. But in production RAG systems — where your models are API-only, your teacher is Claude or GPT-4, and your student is a 3B parameter model running on a single GPU — you don't get to fine-tune anything. All you have is the system prompt.

So I asked a different question: **what if the student model could diagnose its own failures and rewrite its own prompt to fix them?**

That's KD-SPAR.

---

## The Core Idea

KD-SPAR (Knowledge Distillation via Student Prompt Auto-Rewriting) treats the student model as an *active agent with self-knowledge* rather than a passive black box.

Here's the four-phase loop:

**1. Diagnose.** Run the student on training queries. Compare its outputs against cached teacher responses using a KD alignment score. Classify each failure: missing citations? Over-hedged? Incomplete? Format drift?

**2. Self-Interview.** Show the student its own failure alongside the teacher's target pattern. Ask it: *"In one sentence, what instruction should be added to your system prompt to avoid this?"*

**3. Aggregate.** Cluster the proposals, score each on a mini evaluation set, keep the top-K.

**4. Validate & Commit.** Test the candidate prompt on held-out queries. Accept only if alignment improves above threshold δ. Otherwise revert.

The key insight: **the model that will execute the instruction is the same model that authors it.** It knows what kinds of phrasing trigger what kinds of behavior in itself. An external optimizer doesn't have that privileged access.

---

## Five Variants, One Framework

I built Sara (सार — Sanskrit for "quintessence," the refined essence extracted from a larger whole) as a complete toolkit with five KD-SPAR variants:

**Base KD-SPAR** — the core self-rewriting loop described above.

**Multi-Teacher KD-SPAR** — aligns one student to N specialist teachers simultaneously. A citation expert, a reasoning expert, and a calibration expert each score the student. The worst-aligned teacher drives each iteration. A non-regression gate prevents over-fitting one teacher at the expense of another.

**Adversarial KD-SPAR** — focuses on the long tail. Gap-mines the bottom decile of KD scores from production logs and generates adversarial queries designed to break the student. Dual-objective validation ensures hard-query improvement doesn't regress standard performance.

**Federated KD-SPAR** — multiple sites (hospitals, regional offices) jointly optimize a shared prompt without sharing any data. Only instruction strings cross site boundaries — never queries, never responses, never gradients. This is a stronger privacy guarantee than federated gradient sharing, which leaks training data through inversion attacks.

**MetaKDSPAR** — my newest addition. Instead of a single monolithic diagnosis pass, it uses a *conductor + specialist* architecture inspired by metaprompting (Suzgun & Kalai, 2024). Four specialist perspectives — citation analysis, calibration, completeness, and format — independently diagnose failures. A conductor synthesizes the diagnoses. Each specialist proposes fixes from its domain expertise. This catches compound failures that flat diagnosis misses.

---

## Does It Actually Work? The Self-Knowledge Hypothesis

The central scientific claim is testable: **student self-authored instructions should outperform externally proposed instructions, given the same KD scoring signal.**

I designed a controlled five-condition ablation:

| Condition | Proposer | KD Signal |
|-----------|----------|-----------|
| A — KD-SPAR | Student self-proposes | Yes |
| B — External | Teacher proposes for student | Yes |
| C — Random | Generic pool | No |
| D — Baseline | No tuning | No |
| E — MetaKDSPAR | Multi-specialist self-proposes | Yes |

Everything is held constant except who writes the instructions. The **A−B gap** isolates the pure value of self-authorship. The **E−A gap** tests whether multi-perspective diagnosis adds value over flat diagnosis.

The experiment runs entirely on local hardware (Ollama, llama3.1:8b → llama3.2:3b, RTX 3070 Ti) — no API costs, no rate limits, fully reproducible. One command: `bash setup_and_run.sh`. The paper rebuilds automatically with real measured results, computed statistics (t-test, p-value, Cohen's d), and honest adaptive commentary. If the gap is negative, the paper says so and explains why.

---

## What Makes This Different from OPRO, DSPy, or Constitutional AI?

| Method | Objective | Self-Calibrating? | KD Signal? |
|--------|-----------|-------------------|------------|
| OPRO | Task accuracy | No | No |
| DSPy | Task metric | No | Via metric only |
| Constitutional AI | Alignment | Partial | No |
| Self-Refine | Output quality | Partial | No |
| **KD-SPAR** | **KD divergence** | **Yes** | **Yes** |
| **MetaKDSPAR** | **KD divergence** | **Yes (multi-specialist)** | **Yes** |

The three distinguishing properties: (1) the optimization target is the teacher's output distribution, not task accuracy; (2) the proposer and executor are the same model; (3) the student diagnoses *how* it failed before proposing a fix.

---

## The Technical Stack

Sara is a complete, production-grade Python package:

- **29 source modules** across 5 subpackages (core, vision, NLP, advanced, RAG)
- **246 tests**, all offline, no API key needed
- **9 end-to-end examples** from ResNet→MobileNet distillation to MetaKDSPAR
- **Provider-agnostic backend** — switch Anthropic ↔ Ollama with one config change
- **46-page research paper** that rebuilds itself with real experimental results
- **AGPL-3.0 licensed** — open source

The paper covers the full arc: KD theory, taxonomy, implementations, benchmarks, RAG migration, all five variants with formal pseudocode, related work, limitations, and a controlled ablation experiment design.

---

## What I Learned Building This

**Self-knowledge is a real phenomenon.** When you ask a 3B model "what instruction would make you cite sources more consistently?" — it gives surprisingly good answers. Not perfect, but the validate-and-commit gate filters the noise.

**The Federated variant solves a real enterprise problem.** Multi-site RAG deployments (healthcare, banking, government) can't share data. Sharing instruction strings — "Always cite retrieved passages using [Doc-N] notation" — is information-theoretically safer than sharing gradients.

**Metaprompting + KD is unexplored territory.** The combination of specialist decomposition with KD as the objective signal hasn't appeared in the literature. Whether it outperforms flat diagnosis depends on the student model's capacity — a 3B model may not maintain distinct specialist perspectives effectively. The ablation will tell.

**Honesty in research matters.** If the A−B gap comes back negative, the paper says so. The adaptive commentary system in `patch_paper.py` generates honest analysis for every outcome — strong, moderate, marginal, or negative. No spin.

---

## Try It

```bash
git clone https://github.com/ajsinha/sara
cd sara
pip install -e ".[rag]"
ollama pull llama3.1:8b && ollama pull llama3.2:3b
python examples/09_meta_kd_spar.py
```

Or run the full publication experiment:
```bash
bash setup_and_run.sh
```

The paper, the code, and the experiment are all in one package. Everything reproduces.

---

*Sara (सार) — the refined essence extracted from a larger whole. That's what knowledge distillation does. KD-SPAR takes it further: the student finds the sāra of its own failures.*

*#MachineLearning #KnowledgeDistillation #LLM #NLP #RAG #OpenSource #MLResearch*
