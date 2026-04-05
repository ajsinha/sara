# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.2.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
from __future__ import annotations
"""
sara.rag.migration
================
Five-phase RAG model migration pipeline.

    Phase 1  Harvest teacher traces  (query, context, teacher_response)
    Phase 2  Evaluate student baseline on same queries
    Phase 3  Partition traces by query route
    Phase 4  Compute KD alignment scores (semantic similarity)
    Phase 5  Run equivalence test suite and produce a migration report

Designed for migrating from claude-3-5-sonnet-20241022 (teacher)
to claude-sonnet-4-5-20250929 (student) while preserving citation
format, reasoning depth, and calibration.
"""


import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from sara.rag.pipeline import (
    RAGPipeline,
    RAGVectorStore,
    RAGResponse,
    TEACHER_MODEL,
    STUDENT_MODEL,
)
from sara.rag.evaluation import EquivalenceReport, run_equivalence_suite

# ── Constants ─────────────────────────────────────────────────────────────────
CITATION_RE   = re.compile(r"\[Doc-\d+\]")
ROUTE_KEYWORDS = {
    "complex_reasoning": ["explain", "compare", "analyse", "analyze", "why", "how does"],
    "document_summary":  ["summarise", "summarize", "overview", "brief"],
    "simple_lookup":     ["what is", "define", "who is", "when was", "list"],
    "structured_extract":["extract", "enumerate", "table", "give me all"],
}


# ── Trace dataclass ────────────────────────────────────────────────────────────

@dataclass
class RAGTrace:
    """
    A single distillation datum: one query run through the teacher pipeline.

    Fields
    ------
    trace_id         : Unique string identifier
    query            : The user query
    retrieved_docs   : List of {content, source} dicts from retrieval
    teacher_response : Full teacher model response  (distillation target)
    citations        : [Doc-N] references found in the teacher response
    timestamp        : ISO-format UTC timestamp
    route            : Query route classification (filled by partitioner)
    student_response : Student response on the same query (filled by Phase 2)
    kd_score         : Alignment score vs. teacher (filled by Phase 4)
    """

    trace_id:          str
    query:             str
    retrieved_docs:    list[dict]
    teacher_response:  str
    citations:         list[str]
    timestamp:         str            = field(default_factory=lambda: datetime.utcnow().isoformat())
    route:             str            = "default"
    student_response:  Optional[str]  = None
    kd_score:          Optional[float]= None

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "RAGTrace":
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ── Phase 1: Harvest teacher traces ───────────────────────────────────────────

def harvest_teacher_traces(
    queries:          list[str],
    teacher_pipeline: RAGPipeline,
    output_path:      Optional[str] = None,
    max_queries:      int = 500,
) -> list[RAGTrace]:
    """
    Run each query through the *teacher* pipeline and record full responses.

    Parameters
    ----------
    queries          : List of production query strings
    teacher_pipeline : RAGPipeline configured with the teacher model
    output_path      : If given, write traces to this JSONL file
    max_queries      : Cap on the number of queries traced

    Returns
    -------
    List of RAGTrace objects
    """
    traces: list[RAGTrace] = []
    queries = queries[:max_queries]

    writer = open(output_path, "w") if output_path else None
    try:
        for i, query in enumerate(queries):
            try:
                resp  = teacher_pipeline.query(query, return_context=True)
                trace = RAGTrace(
                    trace_id         = f"t{i:05d}",
                    query            = query,
                    retrieved_docs   = [
                        {"content": d.content, "source": d.source}
                        for d in resp.retrieved_docs
                    ],
                    teacher_response = resp.answer,
                    citations        = resp.citations,
                )
                traces.append(trace)
                if writer:
                    writer.write(json.dumps(trace.to_dict()) + "\n")
            except Exception as exc:
                print(f"  [harvest] error on query {i}: {exc}")
    finally:
        if writer:
            writer.close()

    return traces


def load_traces(path: str) -> list[RAGTrace]:
    """Load RAGTrace objects from a JSONL file written by :func:`harvest_teacher_traces`."""
    traces = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                traces.append(RAGTrace.from_dict(json.loads(line)))
    return traces


# ── Phase 2: Student baseline ─────────────────────────────────────────────────

def evaluate_student_baseline(
    traces:           list[RAGTrace],
    student_pipeline: RAGPipeline,
) -> list[RAGTrace]:
    """
    Query the *student* on the same (query, context) pairs as the teacher.
    Fills in ``trace.student_response`` for each trace.

    Parameters
    ----------
    traces           : Traces with teacher responses already populated
    student_pipeline : RAGPipeline configured with the student model

    Returns
    -------
    Same list of traces with student_response populated
    """
    for i, trace in enumerate(traces):
        try:
            resp = student_pipeline.query(trace.query, return_context=False)
            trace.student_response = resp.answer
        except Exception as exc:
            print(f"  [baseline] error on trace {trace.trace_id}: {exc}")
    return traces


# ── Phase 3: Route partitioning ───────────────────────────────────────────────

def classify_route(query: str) -> str:
    """Rule-based query router. Returns one of the ROUTE_KEYWORDS keys."""
    q = query.lower()
    for route, keywords in ROUTE_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return route
    return "complex_reasoning"


def partition_by_route(traces: list[RAGTrace]) -> dict[str, list[RAGTrace]]:
    """
    Classify each trace into a query route and return partitioned groups.

    Returns
    -------
    Dict mapping route_name → list[RAGTrace]
    """
    partitioned: dict[str, list[RAGTrace]] = defaultdict(list)
    for trace in traces:
        trace.route = classify_route(trace.query)
        partitioned[trace.route].append(trace)
    return dict(partitioned)


# ── Phase 4: KD alignment scoring ─────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(len(sa | sb), 1)


def score_traces(traces: list[RAGTrace]) -> list[RAGTrace]:
    """
    Compute a KD alignment score for each trace (student vs teacher).

    Tries BERTScore if available; falls back to Jaccard token overlap.
    Fills in ``trace.kd_score`` in-place.

    Returns
    -------
    Same list with kd_score populated
    """
    valid = [t for t in traces if t.student_response and t.teacher_response]
    if not valid:
        return traces

    try:
        from bert_score import score as bscore  # type: ignore
        _, _, F = bscore(
            [t.student_response for t in valid],
            [t.teacher_response  for t in valid],
            lang="en", verbose=False,
        )
        for trace, s in zip(valid, F.tolist()):
            trace.kd_score = round(float(s), 4)
    except ImportError:
        for trace in valid:
            trace.kd_score = round(
                _jaccard(trace.student_response, trace.teacher_response), 4
            )

    return traces


# ── Migration orchestrator ────────────────────────────────────────────────────

@dataclass
class MigrationResult:
    """Summary produced by :class:`RAGMigration.run`."""
    n_traces:    int
    mean_kd:     float
    report:      EquivalenceReport
    traces:      list[RAGTrace]
    partitions:  dict[str, int]   # route → trace count

    def print(self) -> None:
        print(f"\n{'='*60}")
        print(f"Migration Summary")
        print(f"  Traces       : {self.n_traces}")
        print(f"  Mean KD score: {self.mean_kd:.4f}")
        print(f"  Routes       : {self.partitions}")
        self.report.print()


class RAGMigration:
    """
    Orchestrates the full teacher → student RAG migration pipeline.

    Parameters
    ----------
    teacher_model : Anthropic model ID for the teacher  (departing model)
    student_model : Anthropic model ID for the student  (incoming model)
    vector_store  : Shared RAGVectorStore

    Examples
    --------
    >>> store = RAGVectorStore()
    >>> store.add_documents(chunks)
    >>> migration = RAGMigration(
    ...     teacher_model="claude-3-5-sonnet-20241022",
    ...     student_model="claude-sonnet-4-5-20250929",
    ...     vector_store=store,
    ... )
    >>> result = migration.run(query_log, n_harvest=200)
    >>> result.print()
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
        query_log:    list[str],
        n_harvest:    int  = 200,
        traces_path:  Optional[str] = None,
        load_existing: bool = False,
        verbose:      bool = True,
    ) -> MigrationResult:
        """
        Execute all five migration phases.

        Parameters
        ----------
        query_log      : Production query strings
        n_harvest      : Number of queries to trace
        traces_path    : JSONL file path to save/load traces
        load_existing  : If True and traces_path exists, skip Phase 1
        verbose        : Print progress messages

        Returns
        -------
        MigrationResult
        """
        teacher_pipe = RAGPipeline(self.teacher_model, store=self.store)
        student_pipe = RAGPipeline(self.student_model, store=self.store)

        if verbose:
            print(f"\nRAG Migration: {self.teacher_model} → {self.student_model}")

        # Phase 1
        if load_existing and traces_path and Path(traces_path).exists():
            traces = load_traces(traces_path)
            if verbose:
                print(f"[P1] Loaded {len(traces)} existing traces from {traces_path}")
        else:
            traces = harvest_teacher_traces(
                query_log, teacher_pipe, traces_path, n_harvest
            )
            if verbose:
                print(f"[P1] Harvested {len(traces)} traces")

        # Phase 2
        traces = evaluate_student_baseline(traces, student_pipe)
        if verbose:
            print(f"[P2] Student baseline evaluated")

        # Phase 3
        partitions = partition_by_route(traces)
        if verbose:
            for route, tlist in partitions.items():
                print(f"     Route '{route}': {len(tlist)} traces")

        # Phase 4
        traces = score_traces(traces)
        scored = [t for t in traces if t.kd_score is not None]
        mean_kd = sum(t.kd_score for t in scored) / max(len(scored), 1)
        if verbose:
            print(f"[P4] Mean KD score: {mean_kd:.4f}")

        # Phase 5
        report = run_equivalence_suite(traces)
        if verbose:
            report.print()

        return MigrationResult(
            n_traces   = len(traces),
            mean_kd    = round(mean_kd, 4),
            report     = report,
            traces     = traces,
            partitions = {r: len(t) for r, t in partitions.items()},
        )
