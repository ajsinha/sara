# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.4.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
from __future__ import annotations
"""
sara.rag.kd_spar_adversarial
===========================
Adversarial KD-SPAR: builds prompt robustness by focusing the self-rewriting
loop exclusively on hard examples — queries where the teacher-student gap is
largest or the distribution shift is greatest.

Two hard-example sources
------------------------
1. **Gap-mined hard examples**: queries from production logs where the student's
   KD score is already low (bottom decile).
2. **Adversarially generated queries**: the teacher is asked to generate
   challenging questions about a topic — questions designed to expose the
   student's weakest failure modes.

Dual-objective validation
--------------------------
A candidate prompt must improve on adversarial queries AND not regress on
standard (easy) queries — preventing over-specialisation to hard examples.

Practical use-case
------------------
After a baseline KD-SPAR run stabilises on common queries, run AdversarialKDSPAR
to close the long-tail performance gap on edge cases, multi-hop reasoning,
contradictory context, and out-of-distribution topics.
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
    KDSPAR,
    SPARIteration,
    FailureDiagnosis,
    SELF_INTERVIEW_PROMPT,
    FAILURE_DESCRIPTIONS,
    _kd_score,
    _classify_failure,
    _target_pattern,
    _mean_kd,
)

# ── Hard-example mining prompt ────────────────────────────────────────────────
HARD_QUERY_GEN_PROMPT = """\
You are generating challenging test questions for a RAG (retrieval-augmented
generation) knowledge assistant about the following topic:

TOPIC: {topic}

Generate {n} difficult questions that test:
- Multi-hop reasoning (require connecting information across multiple passages)
- Contradictory or ambiguous context handling
- Fine-grained distinction between similar concepts
- Edge cases and boundary conditions
- Out-of-distribution topics related to but not directly covered by the main topic

Return ONLY the questions, one per line, no numbering or preamble.
"""

ADVERSARIAL_INTERVIEW_PROMPT = """\
You are analysing your performance on a DIFFICULT retrieval-augmented generation task
that you consistently struggle with.

QUERY: {query}

YOUR RESPONSE (which diverges from the target):
{student_response}

FAILURE MODE:
{failure_description}

ADVERSARIAL CONTEXT (why this query is hard):
{adversarial_context}

In exactly ONE sentence, propose a specific instruction for your system prompt
that would help you handle this TYPE of difficult query more robustly.

Rules:
- Write ONLY the instruction text, no preamble
- Focus on the structural difficulty, not just this specific query
- Make it generalisable to similar hard queries
"""


# ── Adversarial query record ──────────────────────────────────────────────────

@dataclass
class AdversarialQuery:
    """A query selected or generated for its difficulty."""
    query:             str
    source:            str    # "gap_mined" | "generated"
    difficulty_score:  float  # 1 - kd_score (higher = harder)
    adversarial_type:  str    # "multi_hop" | "contradiction" | "edge_case" | "unknown"


# ── Adversarial KD-SPAR ───────────────────────────────────────────────────────

class AdversarialKDSPAR(KDSPAR):
    """
    KD-SPAR variant that hardness-mines examples and generates adversarial queries.

    Inherits the full base KD-SPAR loop but overrides the query sourcing
    and self-interview prompt to emphasise structural robustness.

    Parameters
    ----------
    teacher_model         : Anthropic model ID for the teacher
    student_model         : Anthropic model ID for the student
    vector_store          : Shared RAGVectorStore
    adversarial_topics    : List of topic strings for query generation
    n_generated_per_topic : Hard queries to generate per topic
    hardness_percentile   : Bottom X% of KD scores to mine from production logs
    dual_threshold        : Min improvement on adversarial queries to accept
    standard_regression   : Max allowed regression on standard queries

    Examples
    --------
    >>> spar = AdversarialKDSPAR(
    ...     teacher_model="claude-3-5-sonnet-20241022",
    ...     student_model="claude-sonnet-4-5-20250929",
    ...     vector_store=store,
    ...     adversarial_topics=["knowledge distillation", "RAG retrieval"],
    ... )
    >>> hard_queries = spar.build_hard_query_set(production_queries, teacher_responses)
    >>> prompt, history = spar.run_adversarial(
    ...     adversarial_queries=hard_queries,
    ...     standard_queries=standard_queries,
    ...     teacher_responses=teacher_responses,
    ... )
    """

    def __init__(
        self,
        teacher_model:         str = TEACHER_MODEL,
        student_model:         str = STUDENT_MODEL,
        vector_store:          Optional[RAGVectorStore] = None,
        adversarial_topics:    Optional[list[str]] = None,
        n_generated_per_topic: int   = 10,
        hardness_percentile:   float = 0.25,  # bottom 25% = hard
        dual_threshold:        float = 0.005,
        standard_regression:   float = 0.02,
    ) -> None:
        super().__init__(teacher_model, student_model, vector_store)
        self.adversarial_topics      = adversarial_topics or []
        self.n_generated_per_topic   = n_generated_per_topic
        self.hardness_percentile     = hardness_percentile
        self.dual_threshold          = dual_threshold
        self.standard_regression     = standard_regression

    # ── Hard query set construction ───────────────────────────────────────────

    def mine_hard_queries(
        self,
        production_queries: list[str],
        teacher_responses:  dict[str, str],
        student_pipeline:   Optional[RAGPipeline] = None,
        base_prompt:        Optional[str] = None,
    ) -> list[AdversarialQuery]:
        """
        Select the hardest queries from a production log by KD gap.

        Parameters
        ----------
        production_queries : All production query strings
        teacher_responses  : Dict query → teacher response
        student_pipeline   : Student pipeline (created fresh if None)
        base_prompt        : System prompt for the student (uses DEFAULT_SYSTEM if None)

        Returns
        -------
        List of AdversarialQuery sorted by difficulty_score descending
        """
        pipe = student_pipeline or RAGPipeline(
            self.student_model, store=self.store,
            system_prompt=base_prompt or DEFAULT_SYSTEM,
        )
        scored: list[tuple[str, float]] = []
        for q in production_queries:
            if q not in teacher_responses:
                continue
            try:
                s_resp = pipe.query(q, return_context=False).answer
                score  = _kd_score(s_resp, teacher_responses[q])
                scored.append((q, score))
            except Exception:
                pass

        scored.sort(key=lambda x: x[1])  # ascending: hardest first
        cutoff = max(1, int(len(scored) * self.hardness_percentile))
        hard   = scored[:cutoff]

        return [
            AdversarialQuery(
                query             = q,
                source            = "gap_mined",
                difficulty_score  = round(1.0 - score, 4),
                adversarial_type  = self._classify_adversarial_type(q),
            )
            for q, score in hard
        ]

    def generate_adversarial_queries(
        self,
        teacher_pipeline: Optional[RAGPipeline] = None,
    ) -> list[AdversarialQuery]:
        """
        Ask the teacher model to generate challenging queries on configured topics.

        Returns
        -------
        List of AdversarialQuery with source="generated"
        """
        if not self.adversarial_topics:
            return []

        pipe   = teacher_pipeline or RAGPipeline(self.teacher_model, store=self.store)
        client = AnthropicClient(self.teacher_model)
        result: list[AdversarialQuery] = []

        for topic in self.adversarial_topics:
            print(f"  Generating hard queries for topic: '{topic}' …")
            prompt = HARD_QUERY_GEN_PROMPT.format(
                topic=topic, n=self.n_generated_per_topic
            )
            try:
                resp = client._client.messages.create(
                    model=client.model_id, max_tokens=600,
                    messages=[{"role": "user", "content": prompt}],
                )
                lines = [l.strip() for l in resp.content[0].text.strip().split("\n")
                         if l.strip() and len(l.strip()) > 10]
                for q in lines[:self.n_generated_per_topic]:
                    result.append(AdversarialQuery(
                        query            = q,
                        source           = "generated",
                        difficulty_score = 1.0,   # assumed hard until scored
                        adversarial_type = self._classify_adversarial_type(q),
                    ))
            except Exception as exc:
                print(f"  Error generating for '{topic}': {exc}")

        print(f"  Generated {len(result)} adversarial queries across "
              f"{len(self.adversarial_topics)} topic(s)")
        return result

    def build_hard_query_set(
        self,
        production_queries: list[str],
        teacher_responses:  dict[str, str],
    ) -> list[AdversarialQuery]:
        """
        Combine gap-mined hard examples with generated adversarial queries.

        Returns
        -------
        Deduplicated list of AdversarialQuery objects
        """
        mined     = self.mine_hard_queries(production_queries, teacher_responses)
        generated = self.generate_adversarial_queries()
        combined  = mined + generated
        # Deduplicate by query text
        seen: set[str] = set()
        unique = []
        for aq in combined:
            if aq.query not in seen:
                seen.add(aq.query)
                unique.append(aq)
        print(f"\nHard query set: {len(unique)} queries "
              f"({len(mined)} mined + {len(generated)} generated)")
        return unique

    # ── Main adversarial run ──────────────────────────────────────────────────

    def run_adversarial(
        self,
        adversarial_queries: list[AdversarialQuery],
        standard_queries:    list[str],
        teacher_responses:   dict[str, str],
        base_prompt:         Optional[str] = None,
        iterations:          int   = 10,
        n_proposals:         int   = 4,
        top_k:               int   = 3,
        log_path:            Optional[str] = None,
    ) -> tuple[str, list[SPARIteration]]:
        """
        Run the adversarial KD-SPAR loop with dual-objective validation.

        Parameters
        ----------
        adversarial_queries  : Hard queries to optimise on
        standard_queries     : Standard queries used for regression checking
        teacher_responses    : Dict query → teacher response (covers both sets)
        base_prompt          : Starting system prompt
        iterations           : Max iterations
        n_proposals          : Proposals per failure
        top_k                : Instructions to keep per iteration
        log_path             : Optional JSONL log file

        Returns
        -------
        (final_prompt, history)
        """
        current_prompt  = base_prompt or DEFAULT_SYSTEM
        adv_queries     = [aq.query for aq in adversarial_queries]
        student_pipe    = RAGPipeline(
            self.student_model, store=self.store, system_prompt=current_prompt
        )
        interviewer     = AnthropicClient(self.student_model, system_prompt=current_prompt)
        history: list[SPARIteration] = []

        log_fh = open(log_path, "w") if log_path else None
        try:
            for it in range(1, iterations + 1):
                print(f"\n--- Adversarial KD-SPAR  Iteration {it}/{iterations} ---")

                # Phase 1: Diagnose on adversarial queries only
                failures = self._diagnose_adversarial(
                    adv_queries, teacher_responses, student_pipe,
                    adversarial_queries,
                )
                if not failures:
                    print("  No adversarial failures found — converged.")
                    break
                print(f"  {len(failures)} hard failure(s) identified")

                # Phase 2: Adversarially-framed self-interview
                proposals: list[str] = []
                for diag, aq in failures:
                    props = self._self_interview_adversarial(
                        diag, aq, interviewer, n_proposals
                    )
                    proposals.extend(props)
                print(f"  {len(proposals)} proposal(s) generated")

                if not proposals:
                    break

                # Phase 3: Select top on adversarial queries
                base_adv = self._batch_kd(adv_queries[:5], teacher_responses, student_pipe)
                top_ins  = self._select_top_queries(
                    proposals, adv_queries[:5], teacher_responses,
                    student_pipe, base_adv, top_k,
                )

                # Phase 4: Dual-objective validation
                old_adv  = self._batch_kd(adv_queries, teacher_responses, student_pipe)
                old_std  = self._batch_kd(standard_queries, teacher_responses, student_pipe)

                candidate = (
                    current_prompt
                    + "\n\n# Adversarial robustness refinements:\n"
                    + "\n".join(f"- {ins}" for ins in top_ins)
                )
                student_pipe.client.update_system(candidate)

                new_adv  = self._batch_kd(adv_queries, teacher_responses, student_pipe)
                new_std  = self._batch_kd(standard_queries, teacher_responses, student_pipe)

                adv_delta  = new_adv - old_adv
                std_delta  = new_std - old_std
                adv_ok     = adv_delta >= self.dual_threshold
                std_ok     = std_delta >= -self.standard_regression
                accepted   = adv_ok and std_ok

                if not accepted:
                    student_pipe.client.update_system(current_prompt)

                spar_it = SPARIteration(
                    iteration     = it,
                    prompt_before = current_prompt,
                    prompt_after  = candidate if accepted else current_prompt,
                    score_before  = round(old_adv, 4),
                    score_after   = round(new_adv, 4),
                    delta         = round(adv_delta, 4),
                    accepted      = accepted,
                    proposals     = proposals,
                    selected      = top_ins,
                )
                history.append(spar_it)

                if log_fh:
                    log_fh.write(json.dumps({
                        "it": it, "adv_delta": round(adv_delta, 4),
                        "std_delta": round(std_delta, 4),
                        "accepted": accepted, "selected": top_ins,
                    }) + "\n")

                status = "✓ ACCEPTED" if accepted else "✗ REVERTED"
                print(f"  {status}  Δadv={adv_delta:+.4f}  Δstd={std_delta:+.4f}")

                if accepted:
                    current_prompt = candidate
                    interviewer.update_system(current_prompt)

        finally:
            if log_fh:
                log_fh.close()

        return current_prompt, history

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _diagnose_adversarial(
        self,
        queries:     list[str],
        teacher_resps: dict[str, str],
        pipeline:    RAGPipeline,
        aq_list:     list[AdversarialQuery],
        top_k:       int = 5,
    ) -> list[tuple[FailureDiagnosis, AdversarialQuery]]:
        aq_map = {aq.query: aq for aq in aq_list}
        results: list[tuple[FailureDiagnosis, float, AdversarialQuery]] = []
        for q in queries:
            if q not in teacher_resps:
                continue
            try:
                s_resp = pipeline.query(q, return_context=False).answer
                score  = _kd_score(s_resp, teacher_resps[q])
                mode   = _classify_failure(s_resp, teacher_resps[q])
                diag   = FailureDiagnosis(
                    query=q, student_response=s_resp,
                    teacher_response=teacher_resps[q],
                    failure_mode=mode, kd_score=score,
                )
                aq = aq_map.get(q, AdversarialQuery(q, "gap_mined", 1.0 - score, "unknown"))
                results.append((diag, score, aq))
            except Exception:
                pass

        results.sort(key=lambda x: x[1])  # worst first
        return [(d, aq) for d, _, aq in results[:top_k]]

    def _self_interview_adversarial(
        self,
        diag:  FailureDiagnosis,
        aq:    AdversarialQuery,
        client: AnthropicClient,
        n:     int,
    ) -> list[str]:
        adversarial_context = (
            f"Query type: {aq.adversarial_type}  |  "
            f"Source: {aq.source}  |  "
            f"Difficulty: {aq.difficulty_score:.2f}"
        )
        prompt = ADVERSARIAL_INTERVIEW_PROMPT.format(
            query              = diag.query,
            student_response   = diag.student_response[:500],
            failure_description= FAILURE_DESCRIPTIONS.get(diag.failure_mode, diag.failure_mode),
            adversarial_context= adversarial_context,
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

    @staticmethod
    def _classify_adversarial_type(query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["compare", "difference", "versus", "vs", "which"]):
            return "comparative"
        if any(w in q for w in ["why", "cause", "reason", "explain why", "how does"]):
            return "multi_hop"
        if any(w in q for w in ["not", "never", "except", "unless", "contradict"]):
            return "negation_edge"
        if any(w in q for w in ["always", "every", "all", "must", "never"]):
            return "boundary"
        return "unknown"

    @staticmethod
    def _batch_kd(
        queries:    list[str],
        teacher:    dict[str, str],
        pipeline:   RAGPipeline,
    ) -> float:
        scores = []
        for q in queries:
            if q not in teacher:
                continue
            try:
                s = pipeline.query(q, return_context=False).answer
                scores.append(_kd_score(s, teacher[q]))
            except Exception:
                scores.append(0.0)
        return sum(scores) / max(len(scores), 1)

    def _select_top_queries(
        self,
        proposals:  list[str],
        eval_qs:    list[str],
        teacher:    dict[str, str],
        pipeline:   RAGPipeline,
        base:       float,
        top_k:      int,
    ) -> list[str]:
        unique = list({p for p in proposals if len(p) > 15})[:top_k * 3]
        if len(unique) <= top_k:
            return unique
        orig = pipeline.client.system
        scored: list[tuple[str, float]] = []
        for p in unique:
            pipeline.client.update_system(orig + "\n- " + p)
            sc = self._batch_kd(eval_qs, teacher, pipeline)
            scored.append((p, sc))
            pipeline.client.update_system(orig)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored[:top_k]]
