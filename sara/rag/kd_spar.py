# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.1.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
from __future__ import annotations
"""
sara.rag.kd_spar
==============
KD-SPAR: Knowledge Distillation via Student Prompt Auto-Rewriting.

The student model diagnoses its own failure modes and proposes targeted
amendments to its own system prompt — a self-calibrating loop that
requires no weight updates and no external optimiser.

Algorithm
---------
    For each iteration:
        1. Diagnostic Pass   — run student, identify top-k failing traces
        2. Self-Interview     — student proposes one instruction per failure
        3. Aggregation        — cluster proposals, score on mini eval set
        4. Validate & Commit  — accept prompt update only if KD improves

Reference
---------
    Sinha, A. (2025). KD-SPAR: Knowledge Distillation via Student Prompt
    Auto-Rewriting. Technical Reference, Harvard Crimson Edition.
"""


import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sara.core.utils import (
    CITATION_RE,
    HEDGE_WORDS,
    jaccard,
    kd_score,
)
from sara.rag.pipeline import (
    RAGPipeline,
    RAGVectorStore,
    AnthropicClient,
    TEACHER_MODEL,
    STUDENT_MODEL,
    DEFAULT_SYSTEM,
)

# ── Failure taxonomy ──────────────────────────────────────────────────────────
FAILURE_DESCRIPTIONS = {
    "missing_citation": (
        "The response does not cite retrieved passages with [Doc-N] notation, "
        "even though the teacher response does."
    ),
    "over_hedged": (
        "The response hedges excessively (too many 'may', 'might', 'possibly') "
        "relative to the teacher response."
    ),
    "under_hedged": (
        "The response makes confident claims that the teacher hedges or qualifies."
    ),
    "incomplete": (
        "The response is significantly shorter than the teacher response and "
        "omits important information."
    ),
    "format_drift": (
        "The response structure, tone, or formatting differs from the teacher pattern."
    ),
}


SELF_INTERVIEW_PROMPT = """\
You are analysing your own performance on a retrieval-augmented generation task.

QUERY: {query}

YOUR RESPONSE:
{student_response}

FAILURE MODE:
{failure_description}

TARGET PATTERN:
{target_pattern}

In exactly ONE sentence, propose a specific instruction that should be added to
your system prompt to help you avoid this failure in future responses.

Rules:
- Write ONLY the instruction text (no preamble, no explanation, no quotes)
- Be specific and actionable
  BAD: "be more careful"
  GOOD: "Always cite retrieved passages using [Doc-N] notation after each claim"
- Focus exclusively on the described failure mode
"""


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class FailureDiagnosis:
    """Diagnosis for one failing trace."""
    query:            str
    student_response: str
    teacher_response: str
    failure_mode:     str
    kd_score:         float


@dataclass
class SPARIteration:
    """Record of one KD-SPAR iteration."""
    iteration:     int
    prompt_before: str
    prompt_after:  str
    score_before:  float
    score_after:   float
    delta:         float
    accepted:      bool
    proposals:     list[str]
    selected:      list[str]


# ── Helpers (delegate to canonical implementations in core.utils) ────────────

# Re-export under private names for backward compat with variant modules
_jaccard  = jaccard
_kd_score = kd_score


def _classify_failure(student: str, teacher: str) -> str:
    """Return the primary failure mode label."""
    if not CITATION_RE.search(student) and CITATION_RE.search(teacher):
        return "missing_citation"
    t_lower, s_lower = teacher.lower(), student.lower()
    t_h = sum(1 for w in HEDGE_WORDS if w in t_lower)
    s_h = sum(1 for w in HEDGE_WORDS if w in s_lower)
    if s_h > t_h * 2 and t_h > 0:
        return "over_hedged"
    if s_h < t_h // 2 and t_h > 2:
        return "under_hedged"
    if len(student) < len(teacher) * 0.5:
        return "incomplete"
    return "format_drift"


def _target_pattern(teacher: str) -> str:
    parts = []
    if CITATION_RE.search(teacher):
        parts.append("Uses inline citations [Doc-N]")
    if any(w in teacher.lower() for w in HEDGE_WORDS):
        parts.append("Expresses uncertainty with hedging language")
    if len(teacher) > 200:
        parts.append("Provides detailed multi-sentence answers")
    return "; ".join(parts) if parts else "Matches overall teacher response style"


def _mean_kd(
    queries:          list[str],
    teacher_responses: dict[str, str],
    pipeline:          RAGPipeline,
) -> float:
    """Compute mean KD score across a set of queries."""
    scores = []
    for q in queries:
        if q not in teacher_responses:
            continue
        try:
            resp = pipeline.query(q, return_context=False)
            scores.append(_kd_score(resp.answer, teacher_responses[q]))
        except Exception:
            scores.append(0.0)
    return sum(scores) / max(len(scores), 1)


# ── KDSPAR class ──────────────────────────────────────────────────────────────

class KDSPAR:
    """
    KD-SPAR: iteratively refines the student's system prompt by asking the
    student to propose corrections for its own diagnosed failure modes.

    Parameters
    ----------
    teacher_model : Anthropic model ID for the teacher
    student_model : Anthropic model ID for the student (same model acts as
                    both inference engine and self-interviewer)
    vector_store  : Shared RAGVectorStore

    Examples
    --------
    >>> spar = KDSPAR(
    ...     teacher_model="claude-3-5-sonnet-20241022",
    ...     student_model="claude-sonnet-4-5-20250929",
    ...     vector_store=store,
    ... )
    >>> final_prompt = spar.run(
    ...     train_queries=train_q,
    ...     val_queries=val_q,
    ...     teacher_responses=teacher_resp,
    ...     iterations=10,
    ... )
    """

    def __init__(
        self,
        teacher_model: str = TEACHER_MODEL,
        student_model: str = STUDENT_MODEL,
        vector_store:  Optional[RAGVectorStore] = None,
    ) -> None:
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.store         = vector_store or RAGVectorStore()

    def run(
        self,
        train_queries:     list[str],
        val_queries:       list[str],
        teacher_responses: dict[str, str],
        base_prompt:       Optional[str] = None,
        iterations:        int   = 10,
        threshold:         float = 0.003,
        n_proposals:       int   = 4,
        top_k:             int   = 3,
        log_path:          Optional[str] = None,
    ) -> tuple[str, list[SPARIteration]]:
        """
        Run the KD-SPAR optimisation loop.

        Parameters
        ----------
        train_queries      : Queries used for failure diagnosis & mini-eval
        val_queries        : Held-out queries for commit validation
        teacher_responses  : Dict mapping query → teacher response string
        base_prompt        : Starting system prompt (uses DEFAULT_SYSTEM if None)
        iterations         : Maximum SPAR iterations
        threshold          : Minimum KD delta required to accept a new prompt
        n_proposals        : Proposals to generate per failure diagnosis
        top_k              : Instructions to keep per iteration
        log_path           : Optional JSONL file for iteration logs

        Returns
        -------
        (final_prompt, history)  where history is a list of SPARIteration objects
        """
        current_prompt = base_prompt or DEFAULT_SYSTEM
        student_pipe   = RAGPipeline(
            self.student_model, store=self.store, system_prompt=current_prompt
        )
        # Separate client for self-interview (no RAG context)
        interviewer = AnthropicClient(
            model_id=self.student_model, system_prompt=current_prompt
        )
        history: list[SPARIteration] = []
        log_fh = open(log_path, "w") if log_path else None

        try:
            for it in range(1, iterations + 1):
                print(f"\n--- KD-SPAR Iteration {it}/{iterations} ---")

                # Phase 1: Diagnose
                failures = self._diagnose(
                    train_queries, teacher_responses, student_pipe
                )
                if not failures:
                    print("  No failures detected — converged early.")
                    break
                print(f"  {len(failures)} failure(s) identified")

                # Phase 2: Self-interview
                all_proposals: list[str] = []
                for diag in failures:
                    props = self._self_interview(diag, interviewer, n_proposals)
                    all_proposals.extend(props)
                print(f"  {len(all_proposals)} proposal(s) generated")

                if not all_proposals:
                    print("  No proposals — stopping.")
                    break

                # Phase 3: Select top instructions
                base_score   = _mean_kd(train_queries[:6], teacher_responses, student_pipe)
                top_instrs   = self._select_top(
                    all_proposals, train_queries[:6],
                    teacher_responses, student_pipe, base_score, top_k,
                )

                # Phase 4: Validate and commit
                old_score = _mean_kd(val_queries, teacher_responses, student_pipe)
                candidate = (
                    current_prompt
                    + "\n\n# Auto-generated refinements:\n"
                    + "\n".join(f"- {ins}" for ins in top_instrs)
                )
                student_pipe.client.update_system(candidate)
                new_score = _mean_kd(val_queries, teacher_responses, student_pipe)
                delta     = new_score - old_score
                accepted  = delta >= threshold

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

                if log_fh:
                    log_fh.write(json.dumps({
                        "it": it, "delta": round(delta, 4),
                        "accepted": accepted, "selected": top_instrs,
                    }) + "\n")

                status = "✓ ACCEPTED" if accepted else "✗ REVERTED"
                print(f"  {status}  {old_score:.4f} → {new_score:.4f}  (Δ={delta:+.4f})")

                if accepted:
                    current_prompt = candidate
                    interviewer.update_system(current_prompt)

        finally:
            if log_fh:
                log_fh.close()

        accepted_count = sum(1 for s in history if s.accepted)
        print(f"\nKD-SPAR complete: {accepted_count}/{len(history)} iterations accepted")
        return current_prompt, history

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _diagnose(
        self,
        queries:          list[str],
        teacher_responses: dict[str, str],
        pipeline:          RAGPipeline,
        top_k:            int = 5,
    ) -> list[FailureDiagnosis]:
        """Run student on queries, return worst-performing cases."""
        diagnoses: list[FailureDiagnosis] = []
        for q in queries:
            if q not in teacher_responses:
                continue
            try:
                resp  = pipeline.query(q, return_context=False)
                score = _kd_score(resp.answer, teacher_responses[q])
                mode  = _classify_failure(resp.answer, teacher_responses[q])
                diagnoses.append(FailureDiagnosis(
                    query            = q,
                    student_response = resp.answer,
                    teacher_response = teacher_responses[q],
                    failure_mode     = mode,
                    kd_score         = score,
                ))
            except Exception as exc:
                print(f"  [diagnose] error: {exc}")
        diagnoses.sort(key=lambda d: d.kd_score)
        return diagnoses[:top_k]

    def _self_interview(
        self,
        diag:       FailureDiagnosis,
        client:     AnthropicClient,
        n:          int,
    ) -> list[str]:
        """Ask the student to propose one instruction per failure."""
        prompt = SELF_INTERVIEW_PROMPT.format(
            query              = diag.query,
            student_response   = diag.student_response[:500],
            failure_description= FAILURE_DESCRIPTIONS.get(diag.failure_mode, diag.failure_mode),
            target_pattern     = _target_pattern(diag.teacher_response),
        )
        proposals = []
        for _ in range(n):
            try:
                resp = client._client.messages.create(
                    model      = client.model_id,
                    max_tokens = 80,
                    messages   = [{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text.strip()
                if len(text) > 15 and not text.lower().startswith(("sure", "here", "i'll")):
                    proposals.append(text)
            except Exception:
                pass
        return proposals

    def _select_top(
        self,
        proposals:         list[str],
        eval_queries:      list[str],
        teacher_responses: dict[str, str],
        pipeline:          RAGPipeline,
        base_score:        float,
        top_k:             int,
    ) -> list[str]:
        """Score proposals on a mini eval set and return the top-k."""
        unique = list({p for p in proposals if len(p) > 15})
        if len(unique) <= top_k:
            return unique

        original_system = pipeline.client.system
        scored: list[tuple[str, float]] = []
        for prop in unique[:top_k * 3]:   # cap to avoid too many API calls
            test_system = original_system + "\n- " + prop
            pipeline.client.update_system(test_system)
            score = _mean_kd(eval_queries[:4], teacher_responses, pipeline)
            scored.append((prop, score))
            pipeline.client.update_system(original_system)

        scored.sort(key=lambda x: x[1], reverse=True)
        for prop, score in scored[:top_k]:
            print(f"    {score - base_score:+.4f}  {prop[:70]}")
        return [p for p, _ in scored[:top_k]]
