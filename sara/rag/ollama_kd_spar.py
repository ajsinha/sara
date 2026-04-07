# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.7.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ajsinha/sara
from __future__ import annotations
"""
sara.rag.ollama_kd_spar
======================
Local-model KD-SPAR using Ollama.  No API key, no rate limits, no cost.

This module contains thin wrappers around the base KD-SPAR classes that
replace :class:`sara.rag.pipeline.RAGPipeline` / :class:`AnthropicClient`
with their Ollama equivalents.  All core logic is unchanged.

Classes
-------
OllamaKDSPAR                  Base KD-SPAR using local models
OllamaMultiTeacherKDSPAR      Multi-teacher variant
OllamaAdversarialKDSPAR       Adversarial variant
OllamaFederatedSimulation      Federated simulation harness

Recommended model pairs
-----------------------
  llama3.1:8b  →  llama3.2:3b   (same family, controlled comparison)
  qwen2.5:7b   →  llama3.2:3b   (cross-family, tests generalisation)
  llama3.1:8b  →  qwen2.5:3b    (cross-family, Qwen as student)
"""


import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sara.rag.ollama_client import (
    OllamaClient,
    OLLAMA_TEACHER_MODEL,
    OLLAMA_STUDENT_MODEL,
    OLLAMA_DEFAULT_URL,
    OLLAMA_DEFAULT_SYSTEM,
    check_ollama_running,
    ensure_model,
)
from sara.rag.ollama_pipeline import OllamaRAGPipeline
from sara.rag.pipeline import RAGVectorStore
from sara.rag.kd_spar import (
    KDSPAR,
    SPARIteration,
    _kd_score,
    _classify_failure,
    _target_pattern,
    _mean_kd,
    FAILURE_DESCRIPTIONS,
    SELF_INTERVIEW_PROMPT,
)
from sara.rag.kd_spar_multi_teacher import TeacherSpec
from sara.rag.kd_spar_adversarial import AdversarialQuery
from sara.rag.kd_spar_federated import FederatedClientConfig


# ── Ollama-aware base KD-SPAR ──────────────────────────────────────────────

class OllamaKDSPAR:
    """
    KD-SPAR using local Ollama models.

    Identical algorithm to :class:`sara.rag.kd_spar.KDSPAR` but uses
    :class:`OllamaRAGPipeline` and :class:`OllamaClient`.

    Parameters
    ----------
    teacher_model : Ollama model string for the teacher (e.g. ``"llama3.1:8b"``)
    student_model : Ollama model string for the student (e.g. ``"llama3.2:3b"``)
    vector_store  : Shared :class:`RAGVectorStore`
    base_url      : Ollama server URL
    auto_pull     : Pull models automatically if not found

    Examples
    --------
    >>> spar = OllamaKDSPAR(
    ...     teacher_model="llama3.1:8b",
    ...     student_model="llama3.2:3b",
    ...     vector_store=store,
    ... )
    >>> prompt, history = spar.run(train_q, val_q, teacher_responses, iterations=5)
    """

    def __init__(
        self,
        teacher_model: str = OLLAMA_TEACHER_MODEL,
        student_model: str = OLLAMA_STUDENT_MODEL,
        vector_store:  Optional[RAGVectorStore] = None,
        base_url:      str = OLLAMA_DEFAULT_URL,
        auto_pull:     bool = True,
        temperature:   float = 0.1,
    ) -> None:
        if not check_ollama_running(base_url):
            raise ConnectionError(
                f"Ollama not running at {base_url}.\n"
                "Start it with:  ollama serve\n"
                "Install from:   https://ollama.com/install.sh"
            )
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.store         = vector_store or RAGVectorStore()
        self.base_url      = base_url
        self.temperature   = temperature

        if auto_pull:
            ensure_model(teacher_model, base_url)
            ensure_model(student_model, base_url)

    def build_teacher_pipeline(self, system_prompt: Optional[str] = None) -> OllamaRAGPipeline:
        return OllamaRAGPipeline(
            model_id=self.teacher_model, store=self.store,
            system_prompt=system_prompt, base_url=self.base_url,
            auto_pull=False, temperature=self.temperature,
        )

    def build_student_pipeline(self, system_prompt: Optional[str] = None) -> OllamaRAGPipeline:
        return OllamaRAGPipeline(
            model_id=self.student_model, store=self.store,
            system_prompt=system_prompt, base_url=self.base_url,
            auto_pull=False, temperature=self.temperature,
        )

    def harvest_teacher_responses(
        self,
        queries: list[str],
        system_prompt: Optional[str] = None,
    ) -> dict[str, str]:
        """
        Collect teacher responses for a list of queries.

        Parameters
        ----------
        queries        : Query strings
        system_prompt  : Optional custom system prompt for the teacher

        Returns
        -------
        Dict mapping query → teacher response
        """
        pipe   = self.build_teacher_pipeline(system_prompt)
        result = {}
        print(f"Harvesting {len(queries)} teacher responses "
              f"({self.teacher_model}) …")
        for i, q in enumerate(queries, 1):
            try:
                result[q] = pipe.query(q, return_context=False).answer
                if i % 10 == 0:
                    print(f"  {i}/{len(queries)} …")
            except Exception as exc:
                print(f"  error on '{q[:45]}': {exc}")
        print(f"  {len(result)}/{len(queries)} collected")
        return result

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
        Run the KD-SPAR loop.  Same signature as :meth:`KDSPAR.run`.

        Delegates all algorithm logic to the parent class after substituting
        the Ollama pipeline for the Anthropic one.
        """
        import time as _time
        from sara.core.progress import SaraLogger, Heartbeat

        log = SaraLogger("KD-SPAR")
        log.info(f"Teacher: {self.teacher_model}  Student: {self.student_model}")
        log.info(f"Iterations: {iterations}  threshold: {threshold}  "
                 f"proposals: {n_proposals}  top_k: {top_k}")

        current_prompt  = base_prompt or OLLAMA_DEFAULT_SYSTEM
        student_pipe    = self.build_student_pipeline(current_prompt)
        interviewer     = OllamaClient(
            self.student_model, system_prompt=current_prompt,
            base_url=self.base_url, temperature=self.temperature,
        )
        history: list[SPARIteration] = []
        log_fh = open(log_path, "w") if log_path else None

        try:
            for it in range(1, iterations + 1):
                it_start = _time.monotonic()
                log.section(f"KD-SPAR  Iteration {it}/{iterations}")

                # Phase 1: diagnose
                log.step("Phase 1 — Diagnosing failures",
                         total=min(len(train_queries), 8))
                failures = self._diagnose(
                    train_queries, teacher_responses, student_pipe,
                    progress_cb=lambda i: log.tick(i),
                )
                if not failures:
                    log.done("No failures found — converged early")
                    break
                log.done(f"{len(failures)} failure(s) identified")
                for _, _, _, mode, sc in failures[:3]:
                    log.info(f"  mode={mode}  score={sc:.4f}")

                # Phase 2: self-interview
                log.step("Phase 2 — Self-interview",
                         total=len(failures) * n_proposals)
                proposals: list[str] = []
                prop_count = 0
                for q, s_resp, t_resp, mode, _ in failures:
                    props = self._interview(
                        q, s_resp, t_resp, mode, interviewer, n_proposals
                    )
                    proposals.extend(props)
                    prop_count += len(props)
                    log.tick(prop_count)
                log.done(f"{len(proposals)} proposal(s) generated")

                if not proposals:
                    log.warn("No proposals — skipping iteration")
                    break

                # Phase 3: select top-k
                log.step(f"Phase 3 — Scoring & selecting top {top_k}")
                base_score = self._batch_kd(
                    train_queries[:6], teacher_responses, student_pipe
                )
                top_instrs = self._select_top(
                    proposals, train_queries[:6], teacher_responses,
                    student_pipe, base_score, top_k,
                )
                log.done(f"Selected {len(top_instrs)} instruction(s)")
                for ins in top_instrs:
                    log.info(f"  » {ins[:75]}")

                # Phase 4: validate & commit
                log.step("Phase 4 — Validate & commit")
                old_score = self._batch_kd(val_queries, teacher_responses, student_pipe)
                candidate = (
                    current_prompt
                    + "\n\n# SPAR refinements:\n"
                    + "\n".join(f"- {ins}" for ins in top_instrs)
                )
                student_pipe.client.update_system(candidate)
                new_score = self._batch_kd(val_queries, teacher_responses, student_pipe)
                delta      = new_score - old_score
                accepted   = delta >= threshold

                if not accepted:
                    student_pipe.client.update_system(current_prompt)

                spar_it = SPARIteration(
                    iteration=it, prompt_before=current_prompt,
                    prompt_after=candidate if accepted else current_prompt,
                    score_before=round(old_score, 4), score_after=round(new_score, 4),
                    delta=round(delta, 4), accepted=accepted,
                    proposals=proposals, selected=top_instrs,
                )
                history.append(spar_it)

                if log_fh:
                    log_fh.write(json.dumps({
                        "it": it, "delta": round(delta, 4),
                        "accepted": accepted, "selected": top_instrs,
                    }) + "\n")

                it_secs = _time.monotonic() - it_start
                status  = "✓ ACCEPTED" if accepted else "✗ REVERTED"
                from sara.core.progress import _fmt_elapsed
                log.result("SPAR", new_score, delta, 0.0, accepted)
                log.info(f"  {old_score:.4f} → {new_score:.4f}  "
                         f"({status})  iteration took {_fmt_elapsed(it_secs)}")

                if accepted:
                    current_prompt = candidate
                    interviewer.update_system(current_prompt)

        finally:
            if log_fh:
                log_fh.close()

        accepted_count = sum(1 for h in history if h.accepted)
        log.info(f"Loop complete — {accepted_count}/{len(history)} iterations accepted")
        return current_prompt, history

    # ── Helpers (same logic as KDSPAR, but Ollama pipeline) ─────────────────

    def _diagnose(
        self, queries: list[str], teacher_resps: dict[str, str],
        pipeline: OllamaRAGPipeline, top_k: int = 5,
        progress_cb=None,
    ) -> list[tuple]:
        scored = []
        for i, q in enumerate(queries, 1):
            if q not in teacher_resps: continue
            try:
                s = pipeline.query(q, return_context=False).answer
                sc = _kd_score(s, teacher_resps[q])
                md = _classify_failure(s, teacher_resps[q])
                scored.append((q, s, teacher_resps[q], md, sc))
            except Exception:
                pass
            if progress_cb:
                progress_cb(i)
        scored.sort(key=lambda x: x[4])
        return scored[:top_k]

    def _interview(
        self, query: str, s_resp: str, t_resp: str, mode: str,
        client: OllamaClient, n: int,
    ) -> list[str]:
        prompt = SELF_INTERVIEW_PROMPT.format(
            query              = query,
            student_response   = s_resp[:500],
            failure_description= FAILURE_DESCRIPTIONS.get(mode, mode),
            target_pattern     = _target_pattern(t_resp),
        )
        proposals = []
        for _ in range(n):
            try:
                resp = client._client.create(
                    model=client.model_id, max_tokens=80,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text.strip()
                if len(text) > 15 and not text.lower().startswith(("sure","here","i'll")):
                    proposals.append(text)
            except Exception: pass
        return proposals

    def _batch_kd(self, queries: list[str], teacher: dict, pipeline: OllamaRAGPipeline) -> float:
        scores = []
        for q in queries:
            if q not in teacher: continue
            try:
                s = pipeline.query(q, return_context=False).answer
                scores.append(_kd_score(s, teacher[q]))
            except Exception: scores.append(0.0)
        return sum(scores) / max(len(scores), 1)

    def _select_top(
        self, proposals: list[str], eval_qs: list[str],
        teacher: dict, pipeline: OllamaRAGPipeline,
        base: float, top_k: int,
    ) -> list[str]:
        unique = list({p for p in proposals if len(p) > 15})[:top_k * 3]
        if len(unique) <= top_k:
            return unique
        orig = pipeline.client.system
        scored: list[tuple[str, float]] = []
        for p in unique:
            pipeline.client.update_system(orig + "\n- " + p)
            sc = self._batch_kd(eval_qs[:4], teacher, pipeline)
            scored.append((p, sc))
            pipeline.client.update_system(orig)
        scored.sort(key=lambda x: x[1], reverse=True)
        for p, s in scored[:top_k]:
            print(f"    {s - base:+.4f}  {p[:70]}")
        return [p for p, _ in scored[:top_k]]


# ── Ollama Multi-Teacher KD-SPAR ───────────────────────────────────────────

class OllamaMultiTeacherKDSPAR(OllamaKDSPAR):
    """
    Multi-Teacher KD-SPAR using local Ollama models.

    Each teacher spec uses a different Ollama model (or the same model
    with a different system prompt to simulate specialist behaviour).

    Parameters
    ----------
    student_model  : Ollama model for the student
    teacher_specs  : List of OllamaTeacherSpec objects
    vector_store   : Shared RAGVectorStore
    regression_tol : Max allowed KD drop on secondary teachers

    Examples
    --------
    >>> specs = [
    ...     OllamaTeacherSpec("citation",  "llama3.1:8b",
    ...         system="Answer with [Doc-N] citations on every claim.",
    ...         weight=2.0, is_primary=True),
    ...     OllamaTeacherSpec("reasoning", "llama3.1:8b",
    ...         system="Reason step by step before answering.",
    ...         weight=1.0),
    ... ]
    >>> spar = OllamaMultiTeacherKDSPAR("llama3.2:3b", specs, store)
    """

    def __init__(
        self,
        student_model:   str,
        teacher_specs:   list["OllamaTeacherSpec"],
        vector_store:    Optional[RAGVectorStore] = None,
        regression_tol:  float = 0.02,
        base_url:        str = OLLAMA_DEFAULT_URL,
        auto_pull:       bool = True,
    ) -> None:
        # Use first teacher model as the "teacher_model" for parent init
        primary_model = teacher_specs[0].model_id if teacher_specs else OLLAMA_TEACHER_MODEL
        super().__init__(
            teacher_model=primary_model, student_model=student_model,
            vector_store=vector_store, base_url=base_url, auto_pull=False,
        )
        self.teacher_specs  = teacher_specs
        self.regression_tol = regression_tol

        if auto_pull:
            # Pull all unique teacher models + student
            all_models = list({s.model_id for s in teacher_specs} | {student_model})
            for m in all_models:
                ensure_model(m, base_url)

        # Build a pipeline per teacher spec
        self._teacher_pipes: dict[str, OllamaRAGPipeline] = {}
        for spec in teacher_specs:
            self._teacher_pipes[spec.name] = OllamaRAGPipeline(
                model_id=spec.model_id, store=self.store,
                system_prompt=spec.system_prompt, base_url=base_url,
                auto_pull=False, temperature=self.temperature,
            )

    def harvest_all_teacher_responses(
        self, queries: list[str]
    ) -> dict[str, dict[str, str]]:
        """Collect responses from all teacher specs."""
        all_resps: dict[str, dict[str, str]] = {}
        for spec in self.teacher_specs:
            print(f"\nHarvesting from '{spec.name}' ({spec.model_id}) …")
            pipe = self._teacher_pipes[spec.name]
            all_resps[spec.name] = {}
            for q in queries:
                try:
                    all_resps[spec.name][q] = pipe.query(q, return_context=False).answer
                except Exception as exc:
                    print(f"  error: {exc}")
        return all_resps

    def run_multi(
        self,
        train_queries:         list[str],
        val_queries:           list[str],
        teacher_response_sets: dict[str, dict[str, str]],
        base_prompt:           Optional[str] = None,
        iterations:            int   = 8,
        threshold:             float = 0.003,
        n_proposals:           int   = 4,
        top_k:                 int   = 3,
        log_path:              Optional[str] = None,
    ) -> tuple[str, list[SPARIteration]]:
        """Run the multi-teacher loop. Same logic as MultiTeacherKDSPAR.run()."""
        current_prompt = base_prompt or OLLAMA_DEFAULT_SYSTEM
        student_pipe   = self.build_student_pipeline(current_prompt)
        interviewer    = OllamaClient(
            self.student_model, system_prompt=current_prompt,
            base_url=self.base_url, temperature=self.temperature,
        )
        primary_name = next(s.name for s in self.teacher_specs if s.is_primary)
        history: list[SPARIteration] = []
        log_fh = open(log_path, "w") if log_path else None

        try:
            for it in range(1, iterations + 1):
                print(f"\n--- Ollama Multi-Teacher SPAR  Iteration {it}/{iterations} ---")

                # Per-teacher scores
                per_teacher: dict[str, list[float]] = {s.name: [] for s in self.teacher_specs}
                worst_name, worst_score = primary_name, 1.0
                for spec in self.teacher_specs:
                    t_resps = teacher_response_sets.get(spec.name, {})
                    sc = self._batch_kd(train_queries[:6], t_resps, student_pipe)
                    per_teacher[spec.name] = sc
                    if sc * spec.weight < worst_score:
                        worst_score = sc * spec.weight
                        worst_name  = spec.name

                print(f"  Per-teacher scores: "
                      + " | ".join(f"{n}={v:.3f}" for n, v in per_teacher.items()))
                print(f"  Worst: {worst_name} ({worst_score:.3f})")

                worst_resps = teacher_response_sets.get(worst_name, {})
                failures    = self._diagnose(train_queries[:8], worst_resps, student_pipe)
                proposals   = []
                for q, s_resp, t_resp, mode, _ in failures[:3]:
                    props = self._interview(q, s_resp, t_resp, mode, interviewer, n_proposals)
                    proposals.extend(props)
                print(f"  {len(proposals)} proposal(s)")

                if not proposals:
                    break

                primary_resps = teacher_response_sets.get(primary_name, {})
                base_sc       = self._batch_kd(train_queries[:6], primary_resps, student_pipe)
                top_ins       = self._select_top(
                    proposals, train_queries[:6], primary_resps,
                    student_pipe, base_sc, top_k,
                )

                # Validate: primary improves AND secondaries don't regress
                old_scores = {
                    s.name: self._batch_kd(
                        val_queries,
                        teacher_response_sets.get(s.name, {}),
                        student_pipe,
                    )
                    for s in self.teacher_specs
                }
                candidate = (
                    current_prompt + "\n\n# Multi-teacher refinements:\n"
                    + "\n".join(f"- {i}" for i in top_ins)
                )
                student_pipe.client.update_system(candidate)
                new_scores = {
                    s.name: self._batch_kd(
                        val_queries,
                        teacher_response_sets.get(s.name, {}),
                        student_pipe,
                    )
                    for s in self.teacher_specs
                }
                p_delta    = new_scores[primary_name] - old_scores[primary_name]
                no_regress = all(
                    new_scores[s.name] >= old_scores[s.name] - self.regression_tol
                    for s in self.teacher_specs
                )
                accepted = p_delta >= threshold and no_regress

                if not accepted:
                    student_pipe.client.update_system(current_prompt)

                spar_it = SPARIteration(
                    iteration=it, prompt_before=current_prompt,
                    prompt_after=candidate if accepted else current_prompt,
                    score_before=round(old_scores[primary_name], 4),
                    score_after=round(new_scores[primary_name], 4),
                    delta=round(p_delta, 4), accepted=accepted,
                    proposals=proposals, selected=top_ins,
                )
                history.append(spar_it)
                if log_fh:
                    log_fh.write(json.dumps({"it": it, "delta": round(p_delta, 4),
                                              "accepted": accepted}) + "\n")

                status = "✓" if accepted else "✗"
                print(f"  {status} Δprimary={p_delta:+.4f}  regress_ok={no_regress}")
                if accepted:
                    current_prompt = candidate
                    interviewer.update_system(current_prompt)

        finally:
            if log_fh:
                log_fh.close()

        return current_prompt, history


@dataclass
class OllamaTeacherSpec:
    """
    Configuration for one teacher in an Ollama multi-teacher setup.

    Unlike :class:`sara.rag.kd_spar_multi_teacher.TeacherSpec`, this uses
    a system_prompt to simulate specialist behaviour from the SAME base
    model (e.g. llama3.1:8b with three different system prompts = three
    specialist teachers without downloading three separate models).

    Parameters
    ----------
    name          : Human label (e.g. "citation_expert")
    model_id      : Ollama model string (e.g. "llama3.1:8b")
    system_prompt : Specialist system prompt that shapes this teacher's style
    weight        : Relative weight in combined scoring
    is_primary    : True → this teacher drives the commit decision
    """
    name:          str
    model_id:      str
    system_prompt: str = OLLAMA_DEFAULT_SYSTEM
    weight:        float = 1.0
    is_primary:    bool = False
