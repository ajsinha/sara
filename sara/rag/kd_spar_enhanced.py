# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.6.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
from __future__ import annotations
"""
sara.rag.kd_spar_enhanced
===========================
Enhanced KD-SPAR: Seven algorithmic improvements over the base variant.

Enhancements
------------
1. Hybrid proposer       — teacher diagnoses, student proposes
2. BERTScore metric      — semantic similarity replaces Jaccard
3. Contrastive interview — good/bad pair reasoning
4. Warm-start from B     — external proposal bootstraps first iteration
5. More iterations       — default 5 (was 3)
6. Soft commit gate      — probabilistic acceptance via simulated annealing
7. Teacher-guided interview — shows actual teacher response text

All enhancements are configurable and can be toggled individually.
"""

import json
import math
import random
from dataclasses import dataclass
from typing import Optional

from sara.core.utils import (
    CITATION_RE,
    HEDGE_WORDS,
    kd_score,
    kd_score_v2,
    jaccard,
    DEFAULT_SYSTEM_PROMPT,
)
from sara.rag.kd_spar import (
    SPARIteration,
    _classify_failure,
    _target_pattern,
    FAILURE_DESCRIPTIONS,
)


# ── Enhanced prompts ─────────────────────────────────────────────────────

# Enhancement 7: Teacher-guided — shows actual teacher text
TEACHER_GUIDED_INTERVIEW = """\
You are analysing your own performance on a retrieval-augmented generation task.

QUERY: {query}

YOUR RESPONSE:
{student_response}

TEACHER'S RESPONSE (the target you should match):
{teacher_response}

FAILURE MODE: {failure_description}

Your response differs from the teacher's in the way described above.
In exactly ONE sentence, propose a specific instruction that should be
added to your system prompt to make your future responses match the
teacher's pattern more closely.

Rules:
- Write ONLY the instruction text (no preamble, no explanation)
- Be specific and actionable
- Reference concrete patterns from the teacher's response
"""

# Enhancement 3: Contrastive — good/bad pair
CONTRASTIVE_INTERVIEW = """\
You are analysing your own performance patterns.

QUERY WHERE YOU SCORED WELL:
  Q: {good_query}
  Your response: {good_response}
  Score: {good_score:.3f}

QUERY WHERE YOU SCORED POORLY:
  Q: {bad_query}
  Your response: {bad_response}
  Score: {bad_score:.3f}
  Failure: {failure_mode}

TEACHER'S RESPONSE FOR THE BAD QUERY:
{teacher_response}

Compare your good and bad responses. What instruction did you implicitly
follow for the good response that you failed to follow for the bad one?

In exactly ONE sentence, write that instruction so it can be added to
your system prompt. Write ONLY the instruction text.
"""

# Enhancement 1: Teacher diagnosis prompt
TEACHER_DIAGNOSIS = """\
Compare this student response against your own response.

QUERY: {query}

STUDENT RESPONSE:
{student_response}

YOUR (TEACHER) RESPONSE:
{teacher_response}

Identify the single most critical failure mode from this list:
- missing_citation: student doesn't cite [Doc-N] when you did
- over_hedged: student hedges excessively relative to you
- under_hedged: student makes confident claims you hedged
- incomplete: student omits important information you included
- format_drift: student's structure/tone differs from yours

Respond in exactly this format:
FAILURE_MODE: <mode>
SEVERITY: <1-5>
DIAGNOSIS: <one sentence>
"""

# ── Enhancement 8: Tree of Thought prompts ────────────────────────────────

TOT_BRANCH_PROMPT = """\
You are analysing WHY your response failed on this query.

QUERY: {query}

YOUR RESPONSE:
{student_response}

TEACHER'S RESPONSE (the target):
{teacher_response}

FAILURE MODE: {failure_mode}

Think about the ROOT CAUSE of this failure — not the surface symptom but
the underlying reason your response diverged from the teacher's.

Generate {n_branches} DIFFERENT hypotheses about WHY you failed. Each
hypothesis should suggest a fundamentally different root cause.

Format each hypothesis on its own line:
HYPOTHESIS 1: <one sentence explaining a root cause>
HYPOTHESIS 2: <one sentence explaining a different root cause>
HYPOTHESIS 3: <one sentence explaining yet another root cause>
"""

TOT_EVALUATE_PROMPT = """\
You are evaluating a hypothesis about why a response failed.

QUERY: {query}
FAILURE MODE: {failure_mode}

HYPOTHESIS: {hypothesis}

TEACHER'S RESPONSE (the target):
{teacher_response}

On a scale of 1-5, how well does this hypothesis explain the actual
divergence between the student and teacher responses?

Respond in exactly this format:
SCORE: <1-5>
REASONING: <one sentence>
"""

TOT_EXPAND_PROMPT = """\
You have identified a root cause for your failure:

QUERY: {query}
ROOT CAUSE: {hypothesis}
FAILURE MODE: {failure_mode}

TEACHER'S RESPONSE (the target):
{teacher_response}

Based on this root cause, generate {n_expansions} DIFFERENT specific
instructions that could be added to your system prompt to fix this.
Each instruction should address the root cause from a different angle.

Format:
INSTRUCTION 1: <specific, actionable instruction>
INSTRUCTION 2: <different approach to the same root cause>
"""


@dataclass
class EnhancedConfig:
    """Configuration for Enhanced KD-SPAR — toggle each enhancement."""
    use_bert_score:      bool  = True    # Enhancement 2
    use_hybrid_proposer: bool  = True    # Enhancement 1
    use_contrastive:     bool  = True    # Enhancement 3
    warm_start_from_b:   bool  = True    # Enhancement 4
    iterations:          int   = 5       # Enhancement 5
    soft_gate:           bool  = True    # Enhancement 6
    soft_gate_temp:      float = 0.01    # Annealing temperature
    teacher_guided:      bool  = True    # Enhancement 7
    use_tree_of_thought: bool  = True    # Enhancement 8
    tot_branches:        int   = 3       # Hypotheses per failure
    tot_expansions:      int   = 2       # Instructions per winning hypothesis
    tot_depth:           int   = 1       # Tree depth (1 = branch→expand, 2 = recurse)
    threshold:           float = 0.002   # Looser threshold
    n_proposals:         int   = 5
    top_k:               int   = 3
    warm_start_iters:    int   = 1       # How many B-style iterations for warm start


class EnhancedKDSPAR:
    """
    Enhanced KD-SPAR with eight algorithmic improvements.

    Parameters
    ----------
    teacher_model : Ollama model ID for teacher
    student_model : Ollama model ID for student
    vector_store  : Shared RAGVectorStore
    config        : EnhancedConfig (all enhancements on by default)
    base_url      : Ollama server URL
    """

    def __init__(
        self,
        teacher_model: str,
        student_model: str,
        vector_store,
        config: Optional[EnhancedConfig] = None,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
    ) -> None:
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.store         = vector_store
        self.cfg           = config or EnhancedConfig()
        self.base_url      = base_url
        self.temperature   = temperature

    def _query_model(self, model_id: str, system: str, user_msg: str,
                     max_tokens: int = 120) -> str:
        try:
            from sara.rag.ollama_client import OllamaClient
            client = OllamaClient(
                model_id=model_id, system_prompt=system,
                base_url=self.base_url, temperature=self.temperature,
            )
            return client.query(user_msg)
        except Exception:
            return ""

    def _score(self, student: str, teacher: str) -> float:
        """Score using BERTScore v2 or fallback to Jaccard."""
        if self.cfg.use_bert_score:
            return kd_score_v2(student, teacher, use_bert=True)
        return kd_score(student, teacher)

    def _mean_score(self, queries, teacher_responses, pipeline) -> float:
        scores = []
        for q in queries:
            if q not in teacher_responses:
                continue
            try:
                resp = pipeline.query(q, return_context=False)
                scores.append(self._score(resp.answer, teacher_responses[q]))
            except Exception:
                scores.append(0.0)
        return sum(scores) / max(len(scores), 1)

    # ── Enhancement 1: Hybrid diagnosis (teacher diagnoses) ──────────────

    def _teacher_diagnose(self, query: str, student_resp: str,
                          teacher_resp: str) -> tuple[str, str]:
        """Use the teacher model to diagnose student failures."""
        prompt = TEACHER_DIAGNOSIS.format(
            query=query,
            student_response=student_resp[:500],
            teacher_response=teacher_resp[:500],
        )
        raw = self._query_model(
            self.teacher_model,
            "You are analysing a student model's response quality.",
            prompt, 100
        )
        mode = _classify_failure(student_resp, teacher_resp)  # fallback
        diagnosis = raw.strip()
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("FAILURE_MODE:"):
                m = line.split(":", 1)[1].strip().lower()
                if m in FAILURE_DESCRIPTIONS:
                    mode = m
            elif line.startswith("DIAGNOSIS:"):
                diagnosis = line.split(":", 1)[1].strip()
        return mode, diagnosis

    # ── Enhancement 3: Contrastive interview ─────────────────────────────

    def _contrastive_propose(self, good_item, bad_item,
                             teacher_responses) -> list[str]:
        """Generate proposals from good/bad pair contrast."""
        prompt = CONTRASTIVE_INTERVIEW.format(
            good_query=good_item["query"],
            good_response=good_item["response"][:400],
            good_score=good_item["score"],
            bad_query=bad_item["query"],
            bad_response=bad_item["response"][:400],
            bad_score=bad_item["score"],
            failure_mode=bad_item["mode"],
            teacher_response=teacher_responses.get(bad_item["query"], "")[:400],
        )
        proposals = []
        for _ in range(self.cfg.n_proposals):
            text = self._query_model(
                self.student_model, DEFAULT_SYSTEM_PROMPT, prompt, 80
            )
            text = text.strip()
            if len(text) > 15 and not text.lower().startswith(("sure", "here", "i")):
                proposals.append(text)
        return proposals

    # ── Enhancement 7: Teacher-guided self-interview ─────────────────────

    def _teacher_guided_propose(self, query, student_resp, teacher_resp,
                                failure_mode) -> list[str]:
        """Self-interview showing the actual teacher response."""
        prompt = TEACHER_GUIDED_INTERVIEW.format(
            query=query,
            student_response=student_resp[:500],
            teacher_response=teacher_resp[:500],
            failure_description=FAILURE_DESCRIPTIONS.get(failure_mode, failure_mode),
        )
        proposals = []
        for _ in range(self.cfg.n_proposals):
            text = self._query_model(
                self.student_model, DEFAULT_SYSTEM_PROMPT, prompt, 80
            )
            text = text.strip()
            if len(text) > 15 and not text.lower().startswith(("sure", "here", "i")):
                proposals.append(text)
        return proposals

    # ── Enhancement 8: Tree of Thought proposals ─────────────────────────

    def _tot_propose(self, query: str, student_resp: str, teacher_resp: str,
                     failure_mode: str) -> list[str]:
        """
        Tree of Thought proposal generation.

        1. Branch: generate K hypotheses about WHY the failure occurred
        2. Evaluate: score each hypothesis for explanatory power
        3. Expand: the best hypotheses generate specific instructions
        4. (Optional depth > 1): recurse on the best instructions

        Returns a list of candidate instructions ranked by the quality
        of the reasoning path that produced them.
        """
        proposals = []

        # ── Step 1: Branch — generate root-cause hypotheses ──────────────
        branch_prompt = TOT_BRANCH_PROMPT.format(
            query=query,
            student_response=student_resp[:400],
            teacher_response=teacher_resp[:400],
            failure_mode=failure_mode,
            n_branches=self.cfg.tot_branches,
        )
        raw = self._query_model(
            self.student_model,
            "You are a careful analyst diagnosing your own failures.",
            branch_prompt, 250
        )

        # Parse hypotheses
        hypotheses = []
        for line in raw.split("\n"):
            line = line.strip()
            for prefix in ["HYPOTHESIS", "Hypothesis", "hypothesis"]:
                if line.startswith(prefix):
                    text = line.split(":", 1)[-1].strip() if ":" in line else ""
                    if len(text) > 15:
                        hypotheses.append(text)
                    break
            else:
                # Also accept numbered lines like "1. ..."
                if line and line[0].isdigit() and "." in line[:3]:
                    text = line.split(".", 1)[-1].strip()
                    if len(text) > 15:
                        hypotheses.append(text)

        if not hypotheses:
            # Fallback: treat the whole response as one hypothesis
            if len(raw.strip()) > 20:
                hypotheses = [raw.strip()[:200]]

        print(f"    [ToT] {len(hypotheses)} hypotheses generated")

        # ── Step 2: Evaluate — score each hypothesis ─────────────────────
        scored_hypotheses = []
        for hyp in hypotheses[:self.cfg.tot_branches]:
            eval_prompt = TOT_EVALUATE_PROMPT.format(
                query=query,
                failure_mode=failure_mode,
                hypothesis=hyp,
                teacher_response=teacher_resp[:400],
            )
            eval_raw = self._query_model(
                self.student_model,
                "You are evaluating root-cause hypotheses.",
                eval_prompt, 60
            )

            score = 3  # default
            for line in eval_raw.split("\n"):
                if line.strip().startswith("SCORE:"):
                    try:
                        score = int(line.split(":")[1].strip()[0])
                    except (ValueError, IndexError):
                        pass
            scored_hypotheses.append((hyp, min(max(score, 1), 5)))

        scored_hypotheses.sort(key=lambda x: x[1], reverse=True)

        # Select top hypotheses for expansion
        top_n = max(1, self.cfg.tot_branches // 2 + 1)
        best = scored_hypotheses[:top_n]
        print(f"    [ToT] Top {len(best)} hypotheses selected "
              f"(scores: {[s for _, s in best]})")

        # ── Step 3: Expand — generate instructions from best hypotheses ──
        for hyp, score in best:
            expand_prompt = TOT_EXPAND_PROMPT.format(
                query=query,
                hypothesis=hyp,
                failure_mode=failure_mode,
                teacher_response=teacher_resp[:400],
                n_expansions=self.cfg.tot_expansions,
            )
            expand_raw = self._query_model(
                self.student_model,
                "You are writing precise system prompt instructions.",
                expand_prompt, 200
            )

            for line in expand_raw.split("\n"):
                line = line.strip()
                for prefix in ["INSTRUCTION", "Instruction", "instruction"]:
                    if line.startswith(prefix):
                        text = line.split(":", 1)[-1].strip() if ":" in line else ""
                        if len(text) > 15:
                            proposals.append(text)
                        break
                else:
                    if line and line[0].isdigit() and "." in line[:3]:
                        text = line.split(".", 1)[-1].strip()
                        if len(text) > 15:
                            proposals.append(text)

        # ── Optional depth > 1: recurse (expensive but thorough) ────────
        if self.cfg.tot_depth > 1 and proposals:
            # Take the best instruction, treat it as a new "hypothesis"
            # about what instruction works, and expand further
            best_instr = proposals[0]
            refine_prompt = (
                f"This instruction was proposed to fix a {failure_mode} failure:\n"
                f"  \"{best_instr}\"\n\n"
                f"Generate {self.cfg.tot_expansions} more specific variations "
                f"that make this instruction clearer, more actionable, or "
                f"more targeted to the specific pattern in the teacher's response.\n\n"
                f"TEACHER RESPONSE:\n{teacher_resp[:300]}\n\n"
                f"Format:\nVARIATION 1: <instruction>\nVARIATION 2: <instruction>"
            )
            refine_raw = self._query_model(
                self.student_model,
                "You are refining system prompt instructions.",
                refine_prompt, 200
            )
            for line in refine_raw.split("\n"):
                line = line.strip()
                if ":" in line and len(line.split(":", 1)[-1].strip()) > 15:
                    text = line.split(":", 1)[-1].strip()
                    proposals.append(text)

        print(f"    [ToT] {len(proposals)} instructions from tree expansion")
        return proposals

    # ── Enhancement 4: Warm-start from B ─────────────────────────────────

    def _warm_start(self, train_queries, teacher_responses, store) -> str:
        """Run one external-proposal iteration to bootstrap the prompt."""
        from sara.rag.ollama_pipeline import OllamaRAGPipeline

        current = DEFAULT_SYSTEM_PROMPT
        pipe = OllamaRAGPipeline(
            self.student_model, store=store, system_prompt=current,
            base_url=self.base_url, auto_pull=False, temperature=self.temperature,
        )

        for warm_it in range(self.cfg.warm_start_iters):
            # Teacher proposes
            proposals = []
            for q in train_queries[:6]:
                if q not in teacher_responses:
                    continue
                try:
                    s_resp = pipe.query(q, return_context=False).answer
                    t_resp = teacher_responses[q]
                    mode = _classify_failure(s_resp, t_resp)
                    prompt = TEACHER_GUIDED_INTERVIEW.format(
                        query=q,
                        student_response=s_resp[:500],
                        teacher_response=t_resp[:500],
                        failure_description=FAILURE_DESCRIPTIONS.get(mode, mode),
                    )
                    text = self._query_model(
                        self.teacher_model,
                        "You are proposing prompt improvements for a student model.",
                        prompt, 80
                    )
                    text = text.strip()
                    if len(text) > 15:
                        proposals.append(text)
                except Exception:
                    pass

            if proposals:
                unique = list({p for p in proposals if len(p) > 15})[:self.cfg.top_k]
                candidate = (
                    current
                    + "\n\n# Warm-start instructions:\n"
                    + "\n".join(f"- {ins}" for ins in unique)
                )
                pipe.client.update_system(candidate)
                current = candidate

        print(f"  Warm-start complete: {len(current) - len(DEFAULT_SYSTEM_PROMPT)} chars added")
        return current

    # ── Enhancement 6: Soft commit gate ──────────────────────────────────

    def _soft_accept(self, delta: float, iteration: int) -> bool:
        """Probabilistic acceptance — simulated annealing for prompts."""
        if delta >= self.cfg.threshold:
            return True
        if not self.cfg.soft_gate or delta <= -0.05:
            return delta >= self.cfg.threshold
        # Acceptance probability: exp(delta / temperature)
        # Temperature decreases with iteration for convergence
        temp = self.cfg.soft_gate_temp / (1 + 0.3 * iteration)
        prob = math.exp(delta / max(temp, 1e-6))
        accept = random.random() < prob
        if accept:
            print(f"    [soft gate] Δ={delta:+.4f} accepted (p={prob:.3f}, temp={temp:.4f})")
        return accept

    # ── Main loop ────────────────────────────────────────────────────────

    def run(
        self,
        train_queries: list[str],
        val_queries: list[str],
        teacher_responses: dict[str, str],
        base_prompt: Optional[str] = None,
        iterations: Optional[int] = None,
    ) -> tuple[str, list[SPARIteration]]:
        """
        Run the Enhanced KD-SPAR loop.

        Same return type as base KDSPAR.run() for compatibility.
        """
        from sara.rag.ollama_pipeline import OllamaRAGPipeline

        iters = iterations or self.cfg.iterations

        # Enhancement 4: Warm-start from external proposals
        if self.cfg.warm_start_from_b and base_prompt is None:
            current_prompt = self._warm_start(
                train_queries, teacher_responses, self.store
            )
        else:
            current_prompt = base_prompt or DEFAULT_SYSTEM_PROMPT

        student_pipe = OllamaRAGPipeline(
            self.student_model, store=self.store, system_prompt=current_prompt,
            base_url=self.base_url, auto_pull=False, temperature=self.temperature,
        )
        history: list[SPARIteration] = []

        for it in range(1, iters + 1):
            print(f"\n--- Enhanced KD-SPAR Iteration {it}/{iters} ---")

            # Phase 1: Diagnose — collect scores + responses
            scored_items = []
            for q in train_queries[:10]:
                if q not in teacher_responses:
                    continue
                try:
                    s_resp = student_pipe.query(q, return_context=False).answer
                    t_resp = teacher_responses[q]
                    sc = self._score(s_resp, t_resp)

                    # Enhancement 1: Use teacher for diagnosis if hybrid mode
                    if self.cfg.use_hybrid_proposer:
                        mode, diag = self._teacher_diagnose(q, s_resp, t_resp)
                    else:
                        mode = _classify_failure(s_resp, t_resp)
                        diag = FAILURE_DESCRIPTIONS.get(mode, mode)

                    scored_items.append({
                        "query": q, "response": s_resp,
                        "teacher": t_resp, "score": sc,
                        "mode": mode, "diagnosis": diag,
                    })
                except Exception as exc:
                    print(f"  [diag] error: {exc}")

            if not scored_items:
                print("  No items scored — stopping.")
                break

            scored_items.sort(key=lambda x: x["score"])
            bad_items  = scored_items[:5]
            good_items = scored_items[-3:] if len(scored_items) > 3 else scored_items[:1]

            n_fail = len(bad_items)
            print(f"  {len(scored_items)} scored, {n_fail} failures identified")

            # Phase 2: Generate proposals
            all_proposals: list[str] = []

            # Enhancement 7: Teacher-guided proposals for worst failures
            for item in bad_items[:3]:
                if self.cfg.teacher_guided:
                    props = self._teacher_guided_propose(
                        item["query"], item["response"],
                        item["teacher"], item["mode"],
                    )
                else:
                    # Standard self-interview (fallback)
                    props = self._teacher_guided_propose(
                        item["query"], item["response"],
                        item["teacher"], item["mode"],
                    )
                all_proposals.extend(props)

            # Enhancement 3: Contrastive proposals
            if self.cfg.use_contrastive and good_items and bad_items:
                for bad in bad_items[:2]:
                    good = good_items[0]
                    props = self._contrastive_propose(
                        good, bad, teacher_responses
                    )
                    all_proposals.extend(props)

            # Enhancement 8: Tree of Thought proposals
            if self.cfg.use_tree_of_thought:
                for item in bad_items[:2]:  # ToT is expensive, limit to worst 2
                    tot_props = self._tot_propose(
                        item["query"], item["response"],
                        item["teacher"], item["mode"],
                    )
                    all_proposals.extend(tot_props)

            print(f"  {len(all_proposals)} proposal(s) generated")
            if not all_proposals:
                print("  No proposals — stopping.")
                break

            # Phase 3: Select top instructions
            unique = list({p for p in all_proposals if len(p) > 15})
            top_instrs = unique[:self.cfg.top_k]

            # Phase 4: Validate and commit
            old_score = self._mean_score(val_queries, teacher_responses, student_pipe)
            candidate = (
                current_prompt
                + f"\n\n# Enhanced KD-SPAR refinements (iteration {it}):\n"
                + "\n".join(f"- {ins}" for ins in top_instrs)
            )
            student_pipe.client.update_system(candidate)
            new_score = self._mean_score(val_queries, teacher_responses, student_pipe)
            delta = new_score - old_score

            # Enhancement 6: Soft commit gate
            accepted = self._soft_accept(delta, it)

            if not accepted:
                student_pipe.client.update_system(current_prompt)

            spar_it = SPARIteration(
                iteration     = it,
                prompt_before = current_prompt,
                prompt_after  = candidate if accepted else current_prompt,
                score_before  = round(old_score, 4),
                score_after   = round(new_score, 4),
                delta         = round(delta, 4),
                accepted      = accepted,
                proposals     = all_proposals,
                selected      = top_instrs,
            )
            history.append(spar_it)

            status = "✓ ACCEPTED" if accepted else "✗ REVERTED"
            print(f"  {status}  {old_score:.4f} → {new_score:.4f}  (Δ={delta:+.4f})")

            if accepted:
                current_prompt = candidate

        accepted_count = sum(1 for s in history if s.accepted)
        print(f"\nEnhanced KD-SPAR complete: {accepted_count}/{len(history)} "
              f"iterations accepted")
        return current_prompt, history
