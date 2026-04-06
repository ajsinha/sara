# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.6.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
from __future__ import annotations
"""
sara.rag.prompt_opt
=================
Prompt distillation for RAG systems where foundation model weights are
inaccessible (API-only access).

Two strategies:
    GridSearch          Exhaustive / random search over component combinations
    EvolutionaryAPO     Evolutionary prompt optimisation with KD fitness

Both use the teacher's response distribution as the optimisation target.
"""


import itertools
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sara.rag.pipeline import (
    RAGPipeline,
    RAGVectorStore,
    AnthropicClient,
    STUDENT_MODEL,
    DEFAULT_SYSTEM,
)

CITATION_RE = re.compile(r"\[Doc-\d+\]")


# ── Prompt component candidates ───────────────────────────────────────────────

CONTEXT_FORMATS = [
    "Context:\n{chunks}",
    "Relevant knowledge passages:\n{chunks}",
    "[RETRIEVED CONTEXT]\n{chunks}\n[/RETRIEVED CONTEXT]",
    "Use these verified passages to answer:\n{chunks}",
]
CITATION_INSTRUCTIONS = [
    "Cite every claim using [Doc-N] inline citations.",
    "Reference source passages as [Doc-N] immediately after each claim.",
    "Always use [Doc-N] citations when drawing on context passages.",
    "Add [Doc-N] citations only for direct quotes or paraphrases.",
]
UNCERTAINTY_INSTRUCTIONS = [
    "Express uncertainty explicitly when context support is partial.",
    "Use hedging language (may, might, appears) for unconfirmed claims.",
    "State 'I cannot confirm this from the provided context' when unsure.",
    "",   # no explicit uncertainty instruction
]
COT_SCAFFOLDS = [
    "",
    "Think step by step before answering.",
    "First identify which passages are relevant, then synthesise your answer.",
    "Reason through the evidence before forming your conclusion.",
]


# ── Scoring helpers ────────────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(len(sa | sb), 1)


def _kd_score(student: str, teacher: str) -> float:
    t_cit = bool(CITATION_RE.search(teacher))
    s_cit = bool(CITATION_RE.search(student))
    cit   = 1.0 if (not t_cit) or s_cit else 0.0
    return 0.3 * cit + 0.7 * _jaccard(student, teacher)


def _batch_kd_score(
    queries:          list[str],
    teacher_responses: dict[str, str],
    pipeline:          RAGPipeline,
) -> float:
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


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(
    context_format:       str = CONTEXT_FORMATS[0],
    citation_instruction: str = CITATION_INSTRUCTIONS[0],
    uncertainty:          str = UNCERTAINTY_INSTRUCTIONS[0],
    cot_scaffold:         str = "",
) -> str:
    """Assemble a system prompt from individual component strings."""
    parts = [
        "You are a precise RAG knowledge assistant.",
        citation_instruction,
        uncertainty,
        cot_scaffold,
    ]
    return "\n".join(p for p in parts if p.strip())


# ── Grid Search ───────────────────────────────────────────────────────────────

@dataclass
class GridSearchResult:
    """Results from :class:`GridSearch.run`."""
    best_prompt:    str
    best_score:     float
    best_components: dict[str, str]
    all_results:    list[dict] = field(default_factory=list)


class GridSearch:
    """
    Exhaustive (or randomly-sampled) grid search over prompt components.

    Parameters
    ----------
    student_model     : Anthropic model ID for the student
    vector_store      : Shared RAGVectorStore
    max_combinations  : Cap on number of combinations tested

    Examples
    --------
    >>> gs = GridSearch(student_model=STUDENT_MODEL, vector_store=store)
    >>> result = gs.run(eval_queries, teacher_responses)
    >>> print(result.best_score, result.best_prompt)
    """

    def __init__(
        self,
        student_model:    str = STUDENT_MODEL,
        vector_store:     Optional[RAGVectorStore] = None,
        max_combinations: int = 24,
    ) -> None:
        self.student_model    = student_model
        self.store            = vector_store or RAGVectorStore()
        self.max_combinations = max_combinations

    def run(
        self,
        eval_queries:      list[str],
        teacher_responses: dict[str, str],
        log_path:          Optional[str] = None,
        verbose:           bool = True,
    ) -> GridSearchResult:
        """
        Search all combinations of component candidates.

        Parameters
        ----------
        eval_queries       : Queries for scoring each prompt
        teacher_responses  : Dict query → teacher response
        log_path           : Optional JSONL log file
        verbose            : Print each combination's score

        Returns
        -------
        GridSearchResult with best prompt and score
        """
        all_combos = list(itertools.product(
            CONTEXT_FORMATS, CITATION_INSTRUCTIONS,
            UNCERTAINTY_INSTRUCTIONS, COT_SCAFFOLDS,
        ))
        if len(all_combos) > self.max_combinations:
            all_combos = random.sample(all_combos, self.max_combinations)

        if verbose:
            print(f"\nGrid Search: {len(all_combos)} combinations  "
                  f"(eval queries: {len(eval_queries)})")

        best_score      = -1.0
        best_prompt     = DEFAULT_SYSTEM
        best_components: dict[str, str] = {}
        all_results: list[dict] = []

        log_fh = open(log_path, "w") if log_path else None
        try:
            for i, (ctx, cit, unc, cot) in enumerate(all_combos):
                prompt    = build_prompt(ctx, cit, unc, cot)
                pipeline  = RAGPipeline(self.student_model, self.store, system_prompt=prompt)
                score     = _batch_kd_score(eval_queries, teacher_responses, pipeline)

                record = {"i": i, "score": round(score, 4),
                          "ctx": ctx[:40], "cit": cit[:40],
                          "unc": unc[:40], "cot": cot[:40]}
                all_results.append(record)
                if log_fh:
                    log_fh.write(json.dumps(record) + "\n")

                if score > best_score:
                    best_score      = score
                    best_prompt     = prompt
                    best_components = {
                        "context_format": ctx,
                        "citation":       cit,
                        "uncertainty":    unc,
                        "cot":            cot,
                    }

                if verbose:
                    print(f"  [{i+1:02d}] {score:.4f}  cit='{cit[:35]}...'")

        finally:
            if log_fh:
                log_fh.close()

        if verbose:
            print(f"\nBest: {best_score:.4f}")

        return GridSearchResult(
            best_prompt     = best_prompt,
            best_score      = best_score,
            best_components = best_components,
            all_results     = all_results,
        )


# ── Evolutionary APO ──────────────────────────────────────────────────────────

@dataclass
class EvoCandidate:
    """One candidate prompt in the evolutionary population."""
    text:       str
    score:      float = 0.0
    generation: int   = 0


@dataclass
class EvoResult:
    """Results from :class:`EvolutionaryAPO.run`."""
    best_prompt:   str
    best_score:    float
    history:       list[dict] = field(default_factory=list)


_MUTATE_PROMPT = """\
Below is a system prompt for a RAG (retrieval-augmented generation) assistant.
Make exactly ONE small targeted improvement — for example: clarify citation format,
adjust hedging language, strengthen a rule, or add a missing instruction.
Do NOT rewrite the whole prompt.
Return ONLY the improved prompt text, with no preamble or explanation.

ORIGINAL PROMPT:
{prompt}
"""


class EvolutionaryAPO:
    """
    Evolutionary prompt optimisation using KD fitness (BERTScore / Jaccard).

    Parameters
    ----------
    student_model    : Anthropic model ID (also used for mutation)
    vector_store     : Shared RAGVectorStore
    generations      : Number of evolution cycles
    population_size  : Population size per generation
    top_k_survivors  : Elite candidates kept across generations
    mutation_rate    : Probability of mutation vs. crossover

    Examples
    --------
    >>> evo = EvolutionaryAPO(student_model=STUDENT_MODEL, vector_store=store)
    >>> result = evo.run(
    ...     seed_prompts=[DEFAULT_SYSTEM, best_grid_prompt],
    ...     eval_queries=eval_q,
    ...     teacher_responses=teacher_resp,
    ... )
    >>> print(result.best_prompt)
    """

    def __init__(
        self,
        student_model:   str = STUDENT_MODEL,
        vector_store:    Optional[RAGVectorStore] = None,
        generations:     int   = 8,
        population_size: int   = 6,
        top_k_survivors: int   = 2,
        mutation_rate:   float = 0.7,
    ) -> None:
        self.student_model   = student_model
        self.store           = vector_store or RAGVectorStore()
        self.generations     = generations
        self.pop_size        = population_size
        self.top_k           = top_k_survivors
        self.mutation_rate   = mutation_rate
        self._llm            = AnthropicClient(model_id=student_model)

    def run(
        self,
        seed_prompts:      list[str],
        eval_queries:      list[str],
        teacher_responses: dict[str, str],
        log_path:          Optional[str] = None,
        verbose:           bool = True,
    ) -> EvoResult:
        """
        Run the evolutionary optimisation loop.

        Parameters
        ----------
        seed_prompts       : Initial prompt candidates (at least 2)
        eval_queries       : Queries for scoring each candidate
        teacher_responses  : Dict query → teacher response
        log_path           : Optional JSONL log file
        verbose            : Print per-generation best score

        Returns
        -------
        EvoResult with best prompt and history
        """
        def score(prompt: str) -> float:
            pipeline = RAGPipeline(self.student_model, self.store, system_prompt=prompt)
            return _batch_kd_score(eval_queries, teacher_responses, pipeline)

        population = [EvoCandidate(text=p) for p in seed_prompts[:self.pop_size]]
        for c in population:
            c.score = score(c.text)

        history: list[dict] = []
        log_fh = open(log_path, "w") if log_path else None

        try:
            for gen in range(1, self.generations + 1):
                population.sort(key=lambda c: c.score, reverse=True)
                best = population[0]
                if verbose:
                    print(f"Gen {gen:02d}  best={best.score:.4f}  {best.text[:60]}…")

                entry = {"gen": gen, "best": round(best.score, 4),
                         "scores": [round(c.score, 4) for c in population]}
                history.append(entry)
                if log_fh:
                    log_fh.write(json.dumps(entry) + "\n")

                if gen == self.generations:
                    break

                survivors = population[:self.top_k]
                new_pop   = list(survivors)

                while len(new_pop) < self.pop_size:
                    parent = random.choice(survivors)
                    if random.random() < self.mutation_rate or len(survivors) < 2:
                        child_text = self._mutate(parent.text)
                    else:
                        parent2    = random.choice(
                            [s for s in survivors if s is not parent] or survivors
                        )
                        child_text = self._crossover(parent.text, parent2.text)
                    child = EvoCandidate(text=child_text, generation=gen)
                    child.score = score(child_text)
                    new_pop.append(child)

                population = new_pop

        finally:
            if log_fh:
                log_fh.close()

        population.sort(key=lambda c: c.score, reverse=True)
        return EvoResult(
            best_prompt = population[0].text,
            best_score  = population[0].score,
            history     = history,
        )

    def _mutate(self, prompt: str) -> str:
        """Ask the student LLM to make one small improvement to the prompt."""
        try:
            resp = self._llm._client.messages.create(
                model      = self._llm.model_id,
                max_tokens = 500,
                messages   = [{"role": "user",
                                "content": _MUTATE_PROMPT.format(prompt=prompt)}],
            )
            return resp.content[0].text.strip() or prompt
        except Exception:
            return prompt

    @staticmethod
    def _crossover(p1: str, p2: str) -> str:
        """Sentence-level crossover: first half of p1 + second half of p2."""
        s1 = [s.strip() for s in p1.split(".") if s.strip()]
        s2 = [s.strip() for s in p2.split(".") if s.strip()]
        half1 = s1[:max(1, len(s1) // 2)]
        half2 = s2[max(0, len(s2) // 2):]
        return ". ".join(half1 + half2) + "."
