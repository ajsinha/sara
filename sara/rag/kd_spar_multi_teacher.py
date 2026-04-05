# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.2.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
from __future__ import annotations
"""
sara.rag.kd_spar_multi_teacher
=============================
Multi-Teacher KD-SPAR: the student satisfies constraints from N teacher
models simultaneously — useful when migrating from a committee of specialist
models to a single general-purpose student.

Key design decisions
--------------------
* Each teacher independently scores the student; the **worst-aligned** teacher
  drives the self-interview prompt for that iteration.
* Validation requires improvement on the primary teacher AND no regression
  (within a tolerance) on any secondary teacher — preventing the student from
  over-fitting one teacher's style at the expense of another.
* Teachers are weighted; a primary teacher carries more weight in the combined
  KD score used for commit decisions.

Practical use-case
------------------
Migrating from:
    claude-3-5-sonnet  →  citation style    (primary teacher)
    gpt-4o             →  reasoning depth   (secondary)
    gemini-1.5-pro     →  calibration style (secondary)
to a single student: claude-sonnet-4-5
"""


import json
from dataclasses import dataclass, field
from typing import Optional

from sara.rag.pipeline import (
    RAGPipeline,
    RAGVectorStore,
    AnthropicClient,
    DEFAULT_SYSTEM,
    STUDENT_MODEL,
    TEACHER_MODEL,
)
from sara.rag.kd_spar import (
    FAILURE_DESCRIPTIONS,
    SELF_INTERVIEW_PROMPT,
    FailureDiagnosis,
    SPARIteration,
    _kd_score,
    _classify_failure,
    _target_pattern,
)


# ── Teacher spec ──────────────────────────────────────────────────────────────

@dataclass
class TeacherSpec:
    """Configuration for one teacher in a multi-teacher setup."""
    name:      str           # human label, e.g. "citation_expert"
    model_id:  str           # Anthropic model string
    weight:    float = 1.0   # relative weight in combined KD score
    is_primary: bool = False  # primary teacher drives commit decisions


# ── Per-teacher diagnostic ────────────────────────────────────────────────────

@dataclass
class MultiTeacherDiagnosis:
    """
    Failure diagnosis across all teachers for a single query.

    Fields
    ------
    query              : The user query
    student_response   : Student response under the current prompt
    per_teacher_scores : Dict teacher_name → KD score (0–1, higher = better)
    worst_teacher      : Name of the teacher with lowest alignment
    failure_mode       : Primary failure mode relative to worst teacher
    teacher_responses  : Dict teacher_name → teacher response
    """
    query:              str
    student_response:   str
    per_teacher_scores: dict[str, float]
    worst_teacher:      str
    failure_mode:       str
    teacher_responses:  dict[str, str]

    @property
    def worst_score(self) -> float:
        return self.per_teacher_scores.get(self.worst_teacher, 0.0)

    @property
    def mean_score(self) -> float:
        scores = list(self.per_teacher_scores.values())
        return sum(scores) / max(len(scores), 1)


# ── Multi-Teacher KD-SPAR ─────────────────────────────────────────────────────

class MultiTeacherKDSPAR:
    """
    KD-SPAR variant that aligns the student simultaneously to multiple teachers.

    Parameters
    ----------
    student_model   : Anthropic model ID for the student
    teachers        : List of TeacherSpec objects (at least one must be primary)
    vector_store    : Shared RAGVectorStore
    regression_tol  : Max allowed KD score drop on secondary teachers when
                      validating a candidate prompt (e.g. 0.02 = 2% tolerance)

    Examples
    --------
    >>> specs = [
    ...     TeacherSpec("citation",  "claude-3-5-sonnet-20241022", weight=2.0, is_primary=True),
    ...     TeacherSpec("reasoning", "claude-sonnet-4-5-20250929", weight=1.0),
    ... ]
    >>> spar = MultiTeacherKDSPAR(
    ...     student_model="claude-sonnet-4-5-20250929",
    ...     teachers=specs,
    ...     vector_store=store,
    ... )
    >>> prompt, history = spar.run(train_queries, val_queries, teacher_response_sets)
    """

    def __init__(
        self,
        student_model:  str,
        teachers:       list[TeacherSpec],
        vector_store:   Optional[RAGVectorStore] = None,
        regression_tol: float = 0.02,
    ) -> None:
        if not any(t.is_primary for t in teachers):
            teachers[0].is_primary = True   # default first to primary

        self.student_model  = student_model
        self.teachers       = teachers
        self.store          = vector_store or RAGVectorStore()
        self.regression_tol = regression_tol

        # Build pipeline for each teacher
        self._teacher_pipes: dict[str, RAGPipeline] = {}
        for spec in teachers:
            self._teacher_pipes[spec.name] = RAGPipeline(
                model_id=spec.model_id, store=self.store
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def harvest_teacher_responses(
        self,
        queries: list[str],
    ) -> dict[str, dict[str, str]]:
        """
        Collect teacher responses from all teachers for a list of queries.

        Returns
        -------
        Dict mapping teacher_name → {query: response}
        """
        teacher_responses: dict[str, dict[str, str]] = {t.name: {} for t in self.teachers}
        for spec in self.teachers:
            pipe = self._teacher_pipes[spec.name]
            print(f"  Harvesting from teacher '{spec.name}' ({spec.model_id}) …")
            for q in queries:
                try:
                    resp = pipe.query(q, return_context=False)
                    teacher_responses[spec.name][q] = resp.answer
                except Exception as exc:
                    print(f"    Error on '{q[:40]}': {exc}")
        return teacher_responses

    def run(
        self,
        train_queries:        list[str],
        val_queries:          list[str],
        teacher_response_sets: dict[str, dict[str, str]],
        base_prompt:          Optional[str] = None,
        iterations:           int   = 10,
        threshold:            float = 0.003,
        n_proposals:          int   = 4,
        top_k:                int   = 3,
        log_path:             Optional[str] = None,
    ) -> tuple[str, list[SPARIteration]]:
        """
        Run the Multi-Teacher KD-SPAR loop.

        Parameters
        ----------
        train_queries          : Queries for diagnosis and mini-eval
        val_queries            : Held-out queries for commit validation
        teacher_response_sets  : {teacher_name: {query: response}} from all teachers
        base_prompt            : Starting system prompt
        iterations             : Maximum iterations
        threshold              : Min KD delta on primary teacher to accept
        n_proposals            : Proposals per failure diagnosis
        top_k                  : Instructions kept per iteration
        log_path               : Optional JSONL log file

        Returns
        -------
        (final_prompt, history)
        """
        current_prompt = base_prompt or DEFAULT_SYSTEM
        student_pipe   = RAGPipeline(
            self.student_model, store=self.store, system_prompt=current_prompt
        )
        interviewer    = AnthropicClient(self.student_model, system_prompt=current_prompt)
        history: list[SPARIteration] = []
        primary_name   = next(t.name for t in self.teachers if t.is_primary)

        log_fh = open(log_path, "w") if log_path else None
        try:
            for it in range(1, iterations + 1):
                print(f"\n--- Multi-Teacher KD-SPAR  Iteration {it}/{iterations} ---")

                # Phase 1: Diagnose across all teachers
                mt_diagnoses = self._diagnose_multi(
                    train_queries, teacher_response_sets, student_pipe
                )
                if not mt_diagnoses:
                    print("  No failures detected — converged early.")
                    break
                print(f"  {len(mt_diagnoses)} failure(s) identified")
                for d in mt_diagnoses[:3]:
                    print(f"    worst={d.worst_teacher}  score={d.worst_score:.3f}  "
                          f"mode={d.failure_mode}")

                # Phase 2: Self-interview targeting worst teacher
                proposals: list[str] = []
                for diag in mt_diagnoses:
                    t_resp = diag.teacher_responses.get(diag.worst_teacher, "")
                    props  = self._self_interview_targeted(diag, t_resp, interviewer, n_proposals)
                    proposals.extend(props)
                print(f"  {len(proposals)} proposal(s) generated")

                if not proposals:
                    break

                # Phase 3: Aggregate — score proposals on primary teacher only
                primary_resps = teacher_response_sets.get(primary_name, {})
                base_score    = self._mean_kd(
                    train_queries[:6], primary_resps, student_pipe
                )
                top_instrs    = self._select_top(
                    proposals, train_queries[:6], primary_resps,
                    student_pipe, base_score, top_k,
                )

                # Phase 4: Validate — improve primary + don't regress secondaries
                old_scores  = self._per_teacher_kd(
                    val_queries, teacher_response_sets, student_pipe
                )
                candidate   = (
                    current_prompt
                    + "\n\n# Multi-teacher refinements:\n"
                    + "\n".join(f"- {ins}" for ins in top_instrs)
                )
                student_pipe.client.update_system(candidate)
                new_scores  = self._per_teacher_kd(
                    val_queries, teacher_response_sets, student_pipe
                )

                primary_delta  = new_scores.get(primary_name, 0) - old_scores.get(primary_name, 0)
                no_regression  = all(
                    new_scores.get(t.name, 0) >= old_scores.get(t.name, 0) - self.regression_tol
                    for t in self.teachers
                )
                accepted       = primary_delta >= threshold and no_regression

                if not accepted:
                    student_pipe.client.update_system(current_prompt)

                spar_it = SPARIteration(
                    iteration     = it,
                    prompt_before = current_prompt,
                    prompt_after  = candidate if accepted else current_prompt,
                    score_before  = round(old_scores.get(primary_name, 0), 4),
                    score_after   = round(new_scores.get(primary_name, 0), 4),
                    delta         = round(primary_delta, 4),
                    accepted      = accepted,
                    proposals     = proposals,
                    selected      = top_instrs,
                )
                history.append(spar_it)

                if log_fh:
                    log_fh.write(json.dumps({
                        "it": it, "delta": round(primary_delta, 4),
                        "accepted": accepted,
                        "no_regression": no_regression,
                        "per_teacher_delta": {
                            t: round(new_scores.get(t, 0) - old_scores.get(t, 0), 4)
                            for t in old_scores
                        },
                        "selected": top_instrs,
                    }) + "\n")

                status = "✓ ACCEPTED" if accepted else "✗ REVERTED"
                rstr   = "✓" if no_regression else f"✗ regression"
                print(f"  {status}  Δprimary={primary_delta:+.4f}  regression_check={rstr}")
                for t_name in old_scores:
                    delta_t = new_scores.get(t_name, 0) - old_scores.get(t_name, 0)
                    print(f"    teacher={t_name:20s}  {delta_t:+.4f}")

                if accepted:
                    current_prompt = candidate
                    interviewer.update_system(current_prompt)

        finally:
            if log_fh:
                log_fh.close()

        return current_prompt, history

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _diagnose_multi(
        self,
        queries:              list[str],
        teacher_response_sets: dict[str, dict[str, str]],
        student_pipe:          RAGPipeline,
        top_k:                int = 5,
    ) -> list[MultiTeacherDiagnosis]:
        """Run student on all queries; rank by weighted combined KD gap."""
        diagnoses: list[MultiTeacherDiagnosis] = []
        for q in queries:
            try:
                s_resp = student_pipe.query(q, return_context=False).answer
            except Exception:
                continue

            per_teacher: dict[str, float] = {}
            teacher_resps: dict[str, str]  = {}
            for spec in self.teachers:
                t_resp = teacher_response_sets.get(spec.name, {}).get(q, "")
                if t_resp:
                    per_teacher[spec.name]  = _kd_score(s_resp, t_resp)
                    teacher_resps[spec.name] = t_resp

            if not per_teacher:
                continue

            # Find worst teacher (lowest weighted score)
            weighted = {
                name: score * next(t.weight for t in self.teachers if t.name == name)
                for name, score in per_teacher.items()
            }
            worst = min(weighted, key=weighted.get)
            mode  = _classify_failure(s_resp, teacher_resps.get(worst, ""))

            diagnoses.append(MultiTeacherDiagnosis(
                query              = q,
                student_response   = s_resp,
                per_teacher_scores = per_teacher,
                worst_teacher      = worst,
                failure_mode       = mode,
                teacher_responses  = teacher_resps,
            ))

        # Sort by lowest worst-teacher score
        diagnoses.sort(key=lambda d: d.worst_score)
        return diagnoses[:top_k]

    def _self_interview_targeted(
        self,
        diag:       MultiTeacherDiagnosis,
        t_response: str,
        client:     AnthropicClient,
        n:          int,
    ) -> list[str]:
        """Ask the student to fix divergence from a specific teacher's style."""
        teacher_name = diag.worst_teacher
        prompt = SELF_INTERVIEW_PROMPT.format(
            query              = diag.query,
            student_response   = diag.student_response[:500],
            failure_description= (
                FAILURE_DESCRIPTIONS.get(diag.failure_mode, diag.failure_mode)
                + f" (relative to the '{teacher_name}' teacher model)"
            ),
            target_pattern     = _target_pattern(t_response),
        )
        proposals = []
        for _ in range(n):
            try:
                resp = client._client.messages.create(
                    model=client.model_id, max_tokens=80,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text.strip()
                if len(text) > 15 and not text.lower().startswith(("sure", "here", "i'll")):
                    proposals.append(text)
            except Exception:
                pass
        return proposals

    def _mean_kd(
        self,
        queries:    list[str],
        teacher_resps: dict[str, str],
        pipeline:   RAGPipeline,
    ) -> float:
        scores = []
        for q in queries:
            if q not in teacher_resps:
                continue
            try:
                s = pipeline.query(q, return_context=False).answer
                scores.append(_kd_score(s, teacher_resps[q]))
            except Exception:
                scores.append(0.0)
        return sum(scores) / max(len(scores), 1)

    def _per_teacher_kd(
        self,
        queries:              list[str],
        teacher_response_sets: dict[str, dict[str, str]],
        student_pipe:          RAGPipeline,
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for spec in self.teachers:
            t_resps = teacher_response_sets.get(spec.name, {})
            result[spec.name] = self._mean_kd(queries, t_resps, student_pipe)
        return result

    def _select_top(
        self,
        proposals: list[str],
        eval_qs:   list[str],
        t_resps:   dict[str, str],
        pipeline:  RAGPipeline,
        base:      float,
        top_k:     int,
    ) -> list[str]:
        unique = list({p for p in proposals if len(p) > 15})[:top_k * 3]
        if len(unique) <= top_k:
            return unique
        orig = pipeline.client.system
        scored: list[tuple[str, float]] = []
        for p in unique:
            pipeline.client.update_system(orig + "\n- " + p)
            sc = self._mean_kd(eval_qs[:4], t_resps, pipeline)
            scored.append((p, sc))
            pipeline.client.update_system(orig)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:top_k]]
