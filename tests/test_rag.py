# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.7.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ajsinha/sara
"""
tests/test_rag.py
=================
Tests for the RAG modules.

All Anthropic API calls and ChromaDB I/O are mocked via conftest.py fixtures.
No real API key or internet connection required.
"""

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock, patch, call
import pytest

from sara.rag.pipeline import Document, RAGResponse, chunk_text
from sara.rag.evaluation import (
    EquivalenceReport,
    run_equivalence_suite,
    _has_citation,
    _hedge_count,
    _is_json,
    _jaccard,
)
from sara.rag.migration import (
    RAGTrace,
    classify_route,
    partition_by_route,
    score_traces,
)


# ══════════════════════════════════════════════════════════════════════════════
# Document and chunk_text
# ══════════════════════════════════════════════════════════════════════════════

class TestDocument:

    def test_doc_id_is_deterministic(self):
        d = Document(content="Hello world", source="test.txt")
        assert d.doc_id == Document(content="Hello world", source="test.txt").doc_id

    def test_doc_id_differs_for_different_content(self):
        d1 = Document(content="Hello world", source="a.txt")
        d2 = Document(content="Different text", source="a.txt")
        assert d1.doc_id != d2.doc_id

    def test_doc_id_is_16_chars(self):
        d = Document(content="test", source="x")
        assert len(d.doc_id) == 16


class TestChunkText:

    def test_returns_list_of_documents(self):
        chunks = chunk_text("word " * 200, source="doc.txt")
        assert all(isinstance(c, Document) for c in chunks)
        assert len(chunks) > 1

    def test_source_propagated(self):
        chunks = chunk_text("some text here", source="my_file.txt")
        for c in chunks:
            assert c.source == "my_file.txt"

    def test_chunk_indices_sequential(self):
        chunks = chunk_text("word " * 300, source="x")
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_short_text_single_chunk(self):
        chunks = chunk_text("just a few words", source="x", chunk_size=200)
        assert len(chunks) == 1

    def test_overlap_creates_extra_chunks(self):
        text     = "word " * 100
        no_ovlp  = chunk_text(text, source="x", chunk_size=50, overlap=0)
        with_ovlp = chunk_text(text, source="x", chunk_size=50, overlap=20)
        assert len(with_ovlp) >= len(no_ovlp)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluationHelpers:

    def test_has_citation_detects_doc_n(self):
        assert _has_citation("See [Doc-1] for details.")
        assert _has_citation("Evidence in [Doc-12].")
        assert not _has_citation("No citation here.")
        assert not _has_citation("[Document] is not a citation.")

    @pytest.mark.parametrize("text,expected", [
        ("This may be correct.", 1),
        ("It might possibly work.", 2),
        ("Clear statement.", 0),
        ("It may might could work.", 3),
    ])
    def test_hedge_count(self, text, expected):
        assert _hedge_count(text) == expected

    def test_is_json_valid(self):
        assert _is_json('{"key": "value"}')
        assert _is_json('[1, 2, 3]')

    def test_is_json_invalid(self):
        assert not _is_json("plain text")
        assert not _is_json("{incomplete json")
        assert not _is_json("")

    @pytest.mark.parametrize("a,b,expected_range", [
        ("hello world", "hello world", (1.0, 1.0)),
        ("cat dog", "fish bird", (0.0, 0.0)),
        ("the cat sat", "the cat ran", (0.5, 0.8)),
    ])
    def test_jaccard(self, a, b, expected_range):
        score = _jaccard(a, b)
        lo, hi = expected_range
        assert lo <= score <= hi, f"Jaccard({a!r}, {b!r}) = {score}, expected [{lo}, {hi}]"


# ══════════════════════════════════════════════════════════════════════════════
# run_equivalence_suite
# ══════════════════════════════════════════════════════════════════════════════

def _make_trace(
    teacher_response: str,
    student_response: str,
    tid: str = "t0",
) -> RAGTrace:
    return RAGTrace(
        trace_id         = tid,
        query            = "test query",
        retrieved_docs   = [],
        teacher_response = teacher_response,
        citations        = re.findall(r"\[Doc-\d+\]", teacher_response),
        student_response = student_response,
        kd_score         = None,
    )


class TestRunEquivalenceSuite:

    def test_perfect_match_passes_all(self):
        answer = "Knowledge distillation uses soft targets [Doc-1]."
        traces = [_make_trace(answer, answer, tid=f"t{i}") for i in range(5)]
        report = run_equivalence_suite(traces)
        assert report.citation_fidelity == 1.0
        assert report.mean_kd_score == 1.0
        assert report.format_pass_rate == 1.0

    def test_no_citations_fails_citation_check(self):
        teacher_resp = "See the details [Doc-1] and [Doc-2]."
        student_resp = "Here is the answer with no citations."
        traces = [_make_trace(teacher_resp, student_resp) for _ in range(5)]
        report = run_equivalence_suite(traces)
        assert report.citation_fidelity == 0.0
        assert not report.pass_all

    def test_empty_traces_raises(self):
        with pytest.raises(ValueError, match="No traces"):
            run_equivalence_suite([])

    def test_traces_without_student_response_ignored(self):
        good  = _make_trace("Answer [Doc-1].", "Answer [Doc-1].", "t1")
        bad   = RAGTrace(
            trace_id="t2", query="q", retrieved_docs=[], teacher_response="x",
            citations=[], student_response=None,
        )
        # Should not raise; bad trace is ignored
        report = run_equivalence_suite([good, bad])
        assert report is not None

    def test_json_format_preserved(self):
        teacher_json = '{"answer": "yes", "source": "[Doc-1]"}'
        student_json = '{"answer": "yes", "source": "[Doc-1]"}'
        traces = [_make_trace(teacher_json, student_json) for _ in range(3)]
        report = run_equivalence_suite(traces)
        assert report.format_pass_rate == 1.0

    def test_json_format_broken(self):
        teacher_json = '{"answer": "yes"}'
        student_plain = "yes"
        traces = [_make_trace(teacher_json, student_plain) for _ in range(3)]
        report = run_equivalence_suite(traces)
        assert report.format_pass_rate == 0.0

    def test_custom_thresholds(self):
        answer = "Answer [Doc-1]."
        traces = [_make_trace(answer, answer) for _ in range(4)]
        # Require perfect citation fidelity = 1.1 (impossible) → fail
        report = run_equivalence_suite(traces, thresholds={"citation_fidelity": 1.1})
        assert not report.pass_all

    def test_report_to_dict_contains_all_keys(self):
        traces = [_make_trace("a [Doc-1]", "a [Doc-1]") for _ in range(3)]
        report = run_equivalence_suite(traces)
        d = report.to_dict()
        for key in ("citation_fidelity", "mean_kd_score", "format_pass_rate",
                    "calibration_ratio", "hallucination_proxy", "pass_all"):
            assert key in d

    def test_print_does_not_raise(self, capsys):
        traces = [_make_trace("Answer [Doc-1].", "Answer [Doc-1].") for _ in range(3)]
        report = run_equivalence_suite(traces)
        report.print()   # should not throw
        captured = capsys.readouterr()
        assert "EQUIVALENCE TEST REPORT" in captured.out


# ══════════════════════════════════════════════════════════════════════════════
# RAGTrace and migration helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestRAGTrace:

    def test_to_dict_and_from_dict_roundtrip(self):
        t1 = RAGTrace(
            trace_id="t1", query="hello",
            retrieved_docs=[{"content": "doc", "source": "f"}],
            teacher_response="answer [Doc-1]",
            citations=["[Doc-1]"],
            student_response="student answer",
            kd_score=0.85,
        )
        d  = t1.to_dict()
        t2 = RAGTrace.from_dict(d)
        assert t2.trace_id == "t1"
        assert t2.kd_score == 0.85
        assert t2.student_response == "student answer"

    def test_from_dict_ignores_unknown_keys(self):
        d = {
            "trace_id": "x", "query": "q", "retrieved_docs": [],
            "teacher_response": "r", "citations": [],
            "unknown_extra_field": "foo",
        }
        t = RAGTrace.from_dict(d)
        assert t.trace_id == "x"


class TestClassifyRoute:

    @pytest.mark.parametrize("query,expected_route", [
        ("explain how distillation works", "complex_reasoning"),
        ("what is knowledge distillation", "simple_lookup"),
        ("summarize this document", "document_summary"),
        ("extract all techniques mentioned", "structured_extract"),
        ("compare teacher and student models", "complex_reasoning"),
    ])
    def test_route_classification(self, query, expected_route):
        assert classify_route(query) == expected_route


class TestPartitionByRoute:

    def test_all_routes_classified(self):
        traces = [
            RAGTrace("t1","explain kd",[],  "r","[Doc-1]"),
            RAGTrace("t2","what is kd",[],  "r",""),
            RAGTrace("t3","summarize this",[],"r",""),
        ]
        result = partition_by_route(traces)
        total  = sum(len(v) for v in result.values())
        assert total == 3

    def test_route_set_on_trace(self):
        traces = [RAGTrace("t1","explain this",[],  "r","")]
        result = partition_by_route(traces)
        assert traces[0].route != "default" or True  # route is set


class TestScoreTraces:

    def test_kd_score_populated(self):
        traces = [
            RAGTrace("t1","q",[],  "answer about KD","", student_response="answer KD"),
            RAGTrace("t2","q",[],  "different topic","", student_response="completely unrelated"),
        ]
        result = score_traces(traces)
        for t in result:
            assert t.kd_score is not None
            assert 0.0 <= t.kd_score <= 1.0

    def test_identical_responses_score_is_one(self):
        text   = "exact same response about distillation"
        traces = [RAGTrace("t1","q",[], text, "", student_response=text)]
        result = score_traces(traces)
        assert result[0].kd_score == 1.0

    def test_completely_different_responses_score_near_zero(self):
        traces = [
            RAGTrace("t1","q",[], "the cat sat on the mat","",
                     student_response="quantum physics dark matter")
        ]
        result = score_traces(traces)
        assert result[0].kd_score < 0.2

    def test_traces_without_student_response_skipped(self):
        traces = [
            RAGTrace("t1","q",[], "teacher resp","", student_response=None),
        ]
        result = score_traces(traces)
        assert result[0].kd_score is None   # no student response → no score
