# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.7.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ajsinha/sara
from __future__ import annotations
"""
sara.rag.kd_spar_meta
======================
MetaKDSPAR: Metaprompting-Enhanced Knowledge Distillation via Student
Prompt Auto-Rewriting.

Extends the base KD-SPAR loop with a *conductor + specialist* architecture
inspired by metaprompting (Suzgun & Kalai, 2024). Instead of a single
monolithic diagnosis and interview pass, MetaKDSPAR:

  1. Spawns K specialist diagnostic perspectives (citation, calibration,
     completeness, format) from the student model.
  2. A conductor synthesises the multi-perspective diagnoses into a
     prioritised failure list.
  3. Each specialist proposes targeted fixes from its domain expertise.
  4. The conductor selects and reconciles cross-specialist proposals.
  5. Standard validate-and-commit (identical to base KD-SPAR).

The hypothesis: multi-perspective self-diagnosis catches compound failures
that flat single-pass diagnosis misses, producing higher-quality prompt
amendments.

Experiment condition: E (MetaKDSPAR) — compared against A (base KD-SPAR),
B (external), C (random), D (baseline).
"""

import json
from dataclasses import dataclass, field
from typing import Optional

from sara.core.utils import (
    CITATION_RE,
    HEDGE_WORDS,
    jaccard,
    kd_score,
)
from sara.rag.kd_spar import (
    SPARIteration,
    FailureDiagnosis,
    FAILURE_DESCRIPTIONS,
    _kd_score,
    _classify_failure,
    _target_pattern,
)


# ── Specialist definitions ────────────────────────────────────────────────

@dataclass
class Specialist:
    """One diagnostic perspective."""
    name:   str
    system: str
    focus:  str   # failure modes this specialist looks for


SPECIALISTS = [
    Specialist(
        name="citation_expert",
        system=(
            "You are a citation analysis specialist. Your sole focus is whether "
            "responses properly cite retrieved passages using [Doc-N] notation. "
            "You detect: missing citations, incorrect citation numbering, "
            "citations without supporting evidence, and unsupported claims."
        ),
        focus="missing_citation",
    ),
    Specialist(
        name="calibration_expert",
        system=(
            "You are a calibration and uncertainty language specialist. You "
            "analyse whether responses use appropriate hedging — neither over-confident "
            "nor excessively uncertain. You detect: over_hedged (too many 'may', "
            "'might', 'possibly'), under_hedged (confident claims the teacher hedges), "
            "and miscalibrated confidence language."
        ),
        focus="over_hedged,under_hedged",
    ),
    Specialist(
        name="completeness_expert",
        system=(
            "You are a completeness and coverage specialist. You compare the depth "
            "and breadth of information in the student response against the teacher's. "
            "You detect: incomplete responses, missing key points, truncated reasoning, "
            "and insufficient detail."
        ),
        focus="incomplete",
    ),
    Specialist(
        name="format_expert",
        system=(
            "You are a format and structure specialist. You analyse response "
            "organisation, tone, paragraph structure, and overall presentation. "
            "You detect: format_drift, tone mismatches, structural inconsistencies, "
            "and response layout differences."
        ),
        focus="format_drift",
    ),
]


SPECIALIST_DIAGNOSIS_PROMPT = """\
Analyse the student's response against the teacher's target pattern.

QUERY: {query}

STUDENT RESPONSE:
{student_response}

TARGET PATTERN:
{target_pattern}

From your specialist perspective ({specialist_name}), identify the single
most critical failure. Respond in EXACTLY this format:

FAILURE_MODE: <one of: missing_citation, over_hedged, under_hedged, incomplete, format_drift>
SEVERITY: <1-5, where 5 is worst>
DIAGNOSIS: <one sentence explaining the specific failure>
"""

SPECIALIST_PROPOSAL_PROMPT = """\
You are analysing your own performance from the perspective of a {specialist_name}.

QUERY: {query}

YOUR RESPONSE:
{student_response}

FAILURE DIAGNOSED:
{diagnosis}

TARGET PATTERN:
{target_pattern}

In exactly ONE sentence, propose a specific instruction that should be added
to your system prompt to fix this failure. Write ONLY the instruction text.
Be specific and actionable. Focus on your area of expertise.
"""

CONDUCTOR_SYNTHESIS_PROMPT = """\
You are a conductor synthesising diagnoses from multiple specialist perspectives.

SPECIALIST DIAGNOSES:
{diagnoses_text}

QUERY: {query}

Rank these diagnoses by impact (which failure most harms the response quality).
Return the top {top_k} as a numbered list. For each, write:
N. FAILURE_MODE: <mode> | SPECIALIST: <name> | REASON: <one sentence>
"""


# ── Core MetaKDSPAR ──────────────────────────────────────────────────────

@dataclass
class MetaDiagnosis:
    """Diagnosis from one specialist perspective."""
    specialist:  str
    query:       str
    failure_mode: str
    severity:    int
    diagnosis:   str
    student_response: str
    teacher_response: str
    kd_score:    float


class MetaKDSPAR:
    """
    Metaprompting-enhanced KD-SPAR.

    Uses K specialist diagnostic perspectives from the student model plus
    a conductor that synthesises multi-perspective diagnoses and reconciles
    proposals. All other mechanics (KD scoring, validate-and-commit gate,
    prompt accumulation) are identical to base KD-SPAR.

    Parameters
    ----------
    student_model : Model ID for student (also used for specialists)
    vector_store  : Shared RAGVectorStore
    specialists   : List of Specialist definitions (defaults to 4 built-in)
    """

    def __init__(
        self,
        student_model: str,
        vector_store,
        specialists: Optional[list[Specialist]] = None,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
    ) -> None:
        self.student_model = student_model
        self.store         = vector_store
        self.specialists   = specialists or SPECIALISTS
        self.base_url      = base_url
        self.temperature   = temperature

    def _query_model(self, model_id: str, system: str, user_msg: str,
                     max_tokens: int = 150) -> str:
        """Send a single query to the model. Override in subclasses."""
        try:
            from sara.rag.ollama_client import OllamaClient
            client = OllamaClient(
                model_id=model_id, system_prompt=system,
                base_url=self.base_url, temperature=self.temperature,
            )
            return client.query(user_msg)
        except Exception:
            return ""

    # ── Phase 1: Multi-specialist diagnosis ──────────────────────────────

    def diagnose_multi(
        self,
        query: str,
        student_response: str,
        teacher_response: str,
    ) -> list[MetaDiagnosis]:
        """Run all specialists on one (query, student, teacher) triple."""
        diagnoses = []
        target = _target_pattern(teacher_response)

        for spec in self.specialists:
            prompt = SPECIALIST_DIAGNOSIS_PROMPT.format(
                query=query,
                student_response=student_response[:500],
                target_pattern=target,
                specialist_name=spec.name,
            )
            raw = self._query_model(self.student_model, spec.system, prompt, 100)
            diag = self._parse_diagnosis(raw, spec, query, student_response,
                                          teacher_response)
            if diag:
                diagnoses.append(diag)

        # Sort by severity (worst first)
        diagnoses.sort(key=lambda d: d.severity, reverse=True)
        return diagnoses

    def _parse_diagnosis(self, raw: str, spec: Specialist, query: str,
                         student: str, teacher: str) -> Optional[MetaDiagnosis]:
        """Parse specialist diagnosis text into a MetaDiagnosis."""
        mode = _classify_failure(student, teacher)  # fallback
        severity = 3
        diagnosis_text = raw.strip()

        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("FAILURE_MODE:"):
                m = line.split(":", 1)[1].strip().lower()
                if m in FAILURE_DESCRIPTIONS:
                    mode = m
            elif line.startswith("SEVERITY:"):
                try:
                    severity = int(line.split(":", 1)[1].strip()[0])
                except (ValueError, IndexError):
                    pass
            elif line.startswith("DIAGNOSIS:"):
                diagnosis_text = line.split(":", 1)[1].strip()

        if len(diagnosis_text) < 10:
            return None

        return MetaDiagnosis(
            specialist=spec.name,
            query=query,
            failure_mode=mode,
            severity=min(max(severity, 1), 5),
            diagnosis=diagnosis_text,
            student_response=student,
            teacher_response=teacher,
            kd_score=_kd_score(student, teacher),
        )

    # ── Phase 1b: Conductor synthesis ────────────────────────────────────

    def conductor_synthesise(
        self,
        all_diagnoses: list[MetaDiagnosis],
        query: str,
        top_k: int = 3,
    ) -> list[MetaDiagnosis]:
        """Conductor selects the most impactful diagnoses across specialists."""
        if len(all_diagnoses) <= top_k:
            return all_diagnoses

        # Build synthesis prompt
        diag_lines = []
        for i, d in enumerate(all_diagnoses, 1):
            diag_lines.append(
                f"{i}. [{d.specialist}] mode={d.failure_mode} "
                f"severity={d.severity}: {d.diagnosis}"
            )

        prompt = CONDUCTOR_SYNTHESIS_PROMPT.format(
            diagnoses_text="\n".join(diag_lines),
            query=query,
            top_k=top_k,
        )

        conductor_sys = (
            "You are a conductor synthesising specialist diagnoses. "
            "Return a numbered list of the most impactful failures."
        )
        raw = self._query_model(self.student_model, conductor_sys, prompt, 200)

        # Parse conductor output — extract referenced indices
        selected_indices = set()
        for line in raw.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                # Try to match original diagnosis by failure mode
                for i, d in enumerate(all_diagnoses):
                    if d.failure_mode in line.lower() or d.specialist in line.lower():
                        selected_indices.add(i)

        if selected_indices:
            return [all_diagnoses[i] for i in sorted(selected_indices)][:top_k]
        # Fallback: just take top-k by severity
        return all_diagnoses[:top_k]

    # ── Phase 2: Specialist-perspective proposals ────────────────────────

    def specialist_propose(
        self,
        diag: MetaDiagnosis,
        n_proposals: int = 3,
    ) -> list[str]:
        """Generate proposals from the specialist perspective that diagnosed the failure."""
        # Find the matching specialist
        spec = next((s for s in self.specialists if s.name == diag.specialist),
                     self.specialists[0])

        prompt = SPECIALIST_PROPOSAL_PROMPT.format(
            specialist_name=spec.name,
            query=diag.query,
            student_response=diag.student_response[:500],
            diagnosis=diag.diagnosis,
            target_pattern=_target_pattern(diag.teacher_response),
        )

        proposals = []
        for _ in range(n_proposals):
            text = self._query_model(self.student_model, spec.system, prompt, 80)
            text = text.strip()
            if len(text) > 15 and not text.lower().startswith(("sure", "here", "i'll")):
                proposals.append(text)
        return proposals

    # ── Full MetaKDSPAR loop ─────────────────────────────────────────────

    def run(
        self,
        train_queries: list[str],
        val_queries: list[str],
        teacher_responses: dict[str, str],
        base_prompt: Optional[str] = None,
        iterations: int = 3,
        threshold: float = 0.003,
        n_proposals: int = 3,
        top_k_diag: int = 3,
        top_k_instr: int = 3,
    ) -> tuple[str, list[SPARIteration]]:
        """
        Run the MetaKDSPAR optimisation loop.

        Same signature and return type as base KDSPAR.run() for
        drop-in compatibility with the ablation framework.
        """
        from sara.rag.ollama_pipeline import OllamaRAGPipeline

        current_prompt = base_prompt or (
            "You are a precise knowledge assistant. "
            "Answer questions using ONLY the provided context passages. "
            "Cite sources inline as [Doc-N] where N is the passage number. "
            "If the context does not contain the answer, say: "
            "'I cannot find this in the provided context.' "
            "Express uncertainty explicitly when evidence is partial."
        )
        student_pipe = OllamaRAGPipeline(
            self.student_model, store=self.store, system_prompt=current_prompt,
            base_url=self.base_url, auto_pull=False, temperature=self.temperature,
        )
        history: list[SPARIteration] = []

        for it in range(1, iterations + 1):
            print(f"\n--- MetaKDSPAR Iteration {it}/{iterations} ---")

            # Phase 1: Multi-specialist diagnosis across training queries
            all_meta_diags: list[MetaDiagnosis] = []
            for q in train_queries[:8]:
                if q not in teacher_responses:
                    continue
                try:
                    s_resp = student_pipe.query(q, return_context=False).answer
                    diags = self.diagnose_multi(q, s_resp, teacher_responses[q])
                    all_meta_diags.extend(diags)
                except Exception as exc:
                    print(f"  [meta-diag] error: {exc}")

            if not all_meta_diags:
                print("  No diagnoses — converged.")
                break

            # Phase 1b: Conductor selects top failures
            top_diags = self.conductor_synthesise(
                all_meta_diags, train_queries[0], top_k=top_k_diag
            )
            print(f"  {len(all_meta_diags)} specialist diagnoses → "
                  f"{len(top_diags)} selected by conductor")

            # Phase 2: Specialist-perspective proposals
            all_proposals: list[str] = []
            for diag in top_diags:
                props = self.specialist_propose(diag, n_proposals)
                all_proposals.extend(props)
            print(f"  {len(all_proposals)} proposal(s) generated")

            if not all_proposals:
                print("  No proposals — stopping.")
                break

            # Phase 3: Score and select top instructions
            unique = list({p for p in all_proposals if len(p) > 15})
            if not unique:
                break

            old_score = self._mean_kd(val_queries, teacher_responses, student_pipe)

            # Phase 4: Build candidate prompt and validate
            top_instrs = unique[:top_k_instr]
            candidate = (
                current_prompt
                + "\n\n# Meta-KD-SPAR refinements:\n"
                + "\n".join(f"- {ins}" for ins in top_instrs)
            )
            student_pipe.client.update_system(candidate)
            new_score = self._mean_kd(val_queries, teacher_responses, student_pipe)
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

            status = "✓ ACCEPTED" if accepted else "✗ REVERTED"
            print(f"  {status}  {old_score:.4f} → {new_score:.4f}  (Δ={delta:+.4f})")

            if accepted:
                current_prompt = candidate

        accepted_count = sum(1 for s in history if s.accepted)
        print(f"\nMetaKDSPAR complete: {accepted_count}/{len(history)} iterations accepted")
        return current_prompt, history

    def _mean_kd(self, queries, teacher_responses, pipeline) -> float:
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
