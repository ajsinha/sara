# I Built an AI That Writes Its Own Instructions. It Failed. Then I Fixed It.

*Ashutosh Sinha · April 2025*

---

What if a student model could diagnose its own failures and rewrite its own system prompt to fix them?

That's what KD-SPAR does. And when I tested it — with real experiments, on real hardware, with reproducible results — the naive version didn't work. The enhanced version did.

Here's the full story.

---

## The Problem

Every ML engineer deploying LLMs faces this: you have a large teacher model producing great results, and you need a smaller model to do the same job cheaply. But you can't fine-tune — it's API-only. All you have is the system prompt.

Standard prompt optimisers (OPRO, DSPy, APE) treat the model as a black box. None of them exploit the fact that **the model knows something about itself**.

---

## KD-SPAR: The Four-Phase Loop

KD-SPAR (Knowledge Distillation via Student Prompt Auto-Rewriting) treats the student as an *active agent with self-knowledge*:

1. **Diagnose** — run student on queries, score against teacher, classify failures
2. **Self-Interview** — "What instruction should be added to your prompt to avoid this?"
3. **Aggregate** — cluster proposals, score on mini-eval, select top-K
4. **Validate & Commit** — accept only if alignment improves, otherwise revert

The bet: the model that *executes* the instruction is the same model that *authors* it. It has privileged self-knowledge.

---

## The Experiment: Six Conditions, Real Hardware

I ran a controlled ablation with 6 conditions on a System76 OryxPro (RTX 3070 Ti) using Ollama. 10 runs, 2 model configs (llama3.1:8b → llama3.2:3b, qwen2.5:7b → llama3.2:3b). Zero API cost.

| Condition | KD Score | Δ vs Baseline |
|-----------|----------|---------------|
| **F — Enhanced KD-SPAR** | **0.312** | **+0.030** |
| **E — MetaKDSPAR** | **0.312** | **+0.030** |
| B — External proposal | 0.297 | +0.015 |
| D — Baseline | 0.282 | — |
| A — Base KD-SPAR | 0.272 | −0.010 |
| C — Random | 0.200 | −0.082 |

---

## The Unexpected Result: Base KD-SPAR Failed

**A−B gap = −0.025.** The self-knowledge hypothesis was not supported in its naive form. External proposal (B) beat self-proposal (A). A 3B student model couldn't effectively diagnose its own failures.

I could have stopped. Instead, I diagnosed *why* the algorithm failed — which is exactly what KD-SPAR is supposed to do. Four root causes:

**1. Meta-cognitive overload.** A 3B model can't simultaneously understand its failure, reason about the fix, and generate the instruction in one shot.

**2. Metric blindness.** Jaccard token overlap penalises semantically correct paraphrases. Good improvements get rejected by the gate.

**3. Greedy proposals.** Single-shot generation produces redundant variations. Five different wordings of "cite sources" don't help when the real problem is reasoning depth.

**4. Cold start + tight gate.** Starting from vanilla with only 3 iterations and strict threshold leaves no room for convergence.

---

## Eight Enhancements That Rescued the Hypothesis

Each root cause got a targeted fix:

1. **Hybrid Proposer** — teacher diagnoses (better meta-reasoning at 8B), student proposes (privileged self-knowledge)
2. **BERTScore** — semantic similarity replaces token overlap; paraphrases get rewarded
3. **Contrastive Interview** — good/bad pair reasoning: "What did you do right here that you missed there?"
4. **Warm-Start from B** — bootstrap with one external-proposal iteration
5. **5 Iterations** — more room for convergence
6. **Soft Commit Gate** — simulated annealing lets marginal improvements through
7. **Teacher-Guided Interview** — shows actual teacher response text, not just failure labels
8. **Tree of Thought** — branch into root-cause hypotheses, evaluate, expand the best

---

## The Result: Enhanced KD-SPAR Wins

**F−A gap = +0.040** — Enhanced outperforms base by a wide margin.

**F−B gap = +0.014** — Enhanced even beats external proposal. First evidence that self-knowledge *can* outperform teacher-proposed instructions when properly supported.

MetaKDSPAR (4 specialist perspectives + conductor) also hit 0.312 — matching Enhanced. Multi-perspective diagnosis alone was enough to outperform everything except the fully enhanced variant.

**The self-knowledge hypothesis isn't wrong — it's capacity-gated.** A 3B model can't self-diagnose in one shot, but with hybrid diagnosis, BERTScore, Tree of Thought, and warm-start, the student's self-authored instructions outperform even the teacher's proposals.

---

## What I Learned

**Self-knowledge is real but fragile.** Extracting it requires structured reasoning (ToT), not single-shot interviews.

**The metric is half the algorithm.** Jaccard actively penalises good work. BERTScore was the single highest-impact change.

**Negative results drive innovation.** The base failure motivated all eight enhancements. A clean positive A−B gap wouldn't have produced the Enhanced variant.

**Federated prompt distillation is a real enterprise need.** Sharing instruction strings is safer than sharing gradients. Healthcare and banking need exactly this.

---

## Try It

```bash
git clone https://github.com/ajsinha/sara
cd sara && bash setup_and_run.sh
```

50-page paper. 31 modules. 246 tests. 10 examples. 24 references. The paper rebuilds with your actual results. AGPL-3.0 — fully open.

---

*Sara (सार) — the refined essence extracted from a larger whole. KD-SPAR finds the sāra of the student's own failures — and when that's not enough, Enhanced KD-SPAR provides the infrastructure to extract it properly.*

#MachineLearning #KnowledgeDistillation #LLM #NLP #RAG #OpenSource #TreeOfThought #Metaprompting #ReproducibleML
