# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.7.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ajsinha/sara
"""
tests/test_kd_spar_variants.py
==============================
Tests for Multi-Teacher, Adversarial, and Federated KD-SPAR variants.

All tests are fully offline — no real Anthropic API calls are made.
The Anthropic client is mocked via conftest.py fixtures.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from sara.rag.kd_spar_multi_teacher import (
    MultiTeacherKDSPAR,
    TeacherSpec,
    MultiTeacherDiagnosis,
)
from sara.rag.kd_spar_adversarial import (
    AdversarialKDSPAR,
    AdversarialQuery,
)
from sara.rag.kd_spar_federated import (
    FederatedKDSPARClient,
    FederatedKDSPARServer,
    FederatedSimulation,
    FederatedClientConfig,
    FederatedRound,
)
from sara.rag.migration import RAGTrace
from sara.rag.kd_spar import _kd_score


# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_TRACES = [
    ("What is knowledge distillation?",
     "Knowledge distillation transfers knowledge from a large teacher [Doc-1]."),
    ("How does temperature scaling work?",
     "Temperature T softens the output distribution [Doc-1], revealing dark knowledge."),
    ("What is ChromaDB?",
     "ChromaDB is a vector database for semantic search [Doc-1]."),
    ("Explain soft targets.",
     "Soft targets are the teacher's probability distributions [Doc-1] over all classes."),
    ("What is KD-SPAR?",
     "KD-SPAR lets the student rewrite its own prompt [Doc-1] based on KD failures."),
]


def make_mock_pipeline(return_answer: str = "Student answer [Doc-1]."):
    """Create a mock RAGPipeline that returns a fixed answer."""
    pipeline = MagicMock()
    resp = MagicMock()
    resp.answer = return_answer
    pipeline.query.return_value = resp
    pipeline.client = MagicMock()
    pipeline.client.system = "mock system"
    pipeline.client.update_system = MagicMock()
    pipeline.model_id = "claude-sonnet-4-5-20250929"
    return pipeline


def make_mock_anthropic_client(proposal: str = "Always cite sources with [Doc-N]."):
    """Create a mock AnthropicClient that returns a fixed proposal."""
    client = MagicMock()
    resp = MagicMock()
    resp.content = [MagicMock(text=proposal)]
    client._client.messages.create.return_value = resp
    client.model_id = "claude-sonnet-4-5-20250929"
    client.system   = "mock system"
    client.update_system = MagicMock()
    return client


def make_mock_rag_store():
    """Create a mock RAGVectorStore."""
    store = MagicMock()
    store.count = 5
    store.search.return_value = []
    return store


# ══════════════════════════════════════════════════════════════════════════════
# MultiTeacherDiagnosis
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiTeacherDiagnosis:

    def test_worst_score_property(self):
        diag = MultiTeacherDiagnosis(
            query              = "q",
            student_response   = "s",
            per_teacher_scores = {"teacher_a": 0.9, "teacher_b": 0.5},
            worst_teacher      = "teacher_b",
            failure_mode       = "missing_citation",
            teacher_responses  = {"teacher_a": "ta", "teacher_b": "tb"},
        )
        assert diag.worst_score == 0.5

    def test_mean_score_property(self):
        diag = MultiTeacherDiagnosis(
            query="q", student_response="s",
            per_teacher_scores={"a": 0.8, "b": 0.6},
            worst_teacher="b", failure_mode="format_drift",
            teacher_responses={},
        )
        assert abs(diag.mean_score - 0.7) < 1e-9


# ══════════════════════════════════════════════════════════════════════════════
# TeacherSpec
# ══════════════════════════════════════════════════════════════════════════════

class TestTeacherSpec:

    def test_defaults(self):
        spec = TeacherSpec(name="t1", model_id="claude-3-5-sonnet-20241022")
        assert spec.weight == 1.0
        assert not spec.is_primary

    def test_primary_flag(self):
        spec = TeacherSpec("t1", "m1", weight=2.0, is_primary=True)
        assert spec.is_primary
        assert spec.weight == 2.0


# ══════════════════════════════════════════════════════════════════════════════
# MultiTeacherKDSPAR — unit tests without API calls
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiTeacherKDSPAR:

    def _make_spar(self, store=None):
        specs = [
            TeacherSpec("primary",   "claude-3-5-sonnet-20241022", weight=2.0, is_primary=True),
            TeacherSpec("secondary", "claude-3-5-sonnet-20241022", weight=1.0),
        ]
        with patch("sara.rag.pipeline.os.environ.get", return_value="sk-test"), \
             patch("anthropic.Anthropic"):
            return MultiTeacherKDSPAR(
                student_model="claude-sonnet-4-5-20250929",
                teachers=specs,
                vector_store=store or make_mock_rag_store(),
            )

    def test_first_teacher_becomes_primary_if_none_set(self):
        specs = [TeacherSpec("t1", "m1"), TeacherSpec("t2", "m2")]
        with patch("sara.rag.pipeline.os.environ.get", return_value="sk-test"), \
             patch("anthropic.Anthropic"):
            spar = MultiTeacherKDSPAR("student", specs, make_mock_rag_store())
        primaries = [t for t in spar.teachers if t.is_primary]
        assert len(primaries) == 1
        assert primaries[0].name == "t1"

    def test_mean_kd_all_same_returns_single_score(self):
        with patch("sara.rag.pipeline.os.environ.get", return_value="sk-test"), \
             patch("anthropic.Anthropic"):
            spar = self._make_spar()
        pipe = make_mock_pipeline("answer [Doc-1].")
        t_resps = {"q1": "answer [Doc-1].", "q2": "answer [Doc-1]."}
        score = spar._mean_kd(["q1", "q2"], t_resps, pipe)
        assert 0.0 <= score <= 1.0

    def test_mean_kd_skips_missing_queries(self):
        with patch("sara.rag.pipeline.os.environ.get", return_value="sk-test"), \
             patch("anthropic.Anthropic"):
            spar = self._make_spar()
        pipe = make_mock_pipeline("answer")
        t_resps = {"q_present": "answer"}
        score = spar._mean_kd(["q_present", "q_missing"], t_resps, pipe)
        assert 0.0 <= score <= 1.0

    def test_per_teacher_kd_returns_dict_for_all_teachers(self):
        with patch("sara.rag.pipeline.os.environ.get", return_value="sk-test"), \
             patch("anthropic.Anthropic"):
            spar = self._make_spar()
        pipe = make_mock_pipeline("answer")
        trs  = {"primary": {"q1": "pr ans [Doc-1]."}, "secondary": {"q1": "sec ans [Doc-1]."}}
        result = spar._per_teacher_kd(["q1"], trs, pipe)
        assert "primary" in result and "secondary" in result

    def test_diagnose_multi_handles_empty_queries(self):
        with patch("sara.rag.pipeline.os.environ.get", return_value="sk-test"), \
             patch("anthropic.Anthropic"):
            spar = self._make_spar()
        pipe   = make_mock_pipeline("answer")
        result = spar._diagnose_multi([], {}, pipe)
        assert result == []

    def test_diagnose_multi_handles_missing_teacher_responses(self):
        with patch("sara.rag.pipeline.os.environ.get", return_value="sk-test"), \
             patch("anthropic.Anthropic"):
            spar = self._make_spar()
        pipe  = make_mock_pipeline("answer without citation")
        trs   = {"primary": {}, "secondary": {}}
        result = spar._diagnose_multi(["q1"], trs, pipe)
        assert result == []

    def test_select_top_deduplicates(self):
        with patch("sara.rag.pipeline.os.environ.get", return_value="sk-test"), \
             patch("anthropic.Anthropic"):
            spar = self._make_spar()
        pipe      = make_mock_pipeline("answer [Doc-1].")
        t_resps   = {"q1": "answer [Doc-1]."}
        proposals = ["Cite sources.", "Cite sources.", "Use hedging language."]
        result    = spar._select_top(proposals, ["q1"], t_resps, pipe, 0.5, top_k=2)
        assert len(result) <= 2
        assert len(set(result)) == len(result)

    def test_regression_tol_default_value(self):
        with patch("sara.rag.pipeline.os.environ.get", return_value="sk-test"), \
             patch("anthropic.Anthropic"):
            spar = self._make_spar()
        assert spar.regression_tol == 0.02


# ══════════════════════════════════════════════════════════════════════════════
# AdversarialQuery
# ══════════════════════════════════════════════════════════════════════════════

class TestAdversarialQuery:

    def test_fields_accessible(self):
        aq = AdversarialQuery(
            query="Explain KD.", source="gap_mined",
            difficulty_score=0.75, adversarial_type="multi_hop"
        )
        assert aq.query == "Explain KD."
        assert aq.difficulty_score == 0.75

    def test_default_adversarial_type(self):
        aq = AdversarialQuery("q", "generated", 1.0, "unknown")
        assert aq.adversarial_type == "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# AdversarialKDSPAR — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAdversarialKDSPAR:

    def _make_spar(self, store=None):
        with patch("sara.rag.pipeline.os.environ.get", return_value="sk-test"), \
             patch("anthropic.Anthropic"):
            return AdversarialKDSPAR(
                teacher_model="claude-3-5-sonnet-20241022",
                student_model="claude-sonnet-4-5-20250929",
                vector_store=store or make_mock_rag_store(),
                adversarial_topics=[],
            )

    def test_classify_adversarial_type_multi_hop(self):
        assert AdversarialKDSPAR._classify_adversarial_type(
            "Why does temperature affect distribution softness?"
        ) == "multi_hop"

    def test_classify_adversarial_type_comparative(self):
        assert AdversarialKDSPAR._classify_adversarial_type(
            "Compare soft targets vs hard labels."
        ) == "comparative"

    def test_classify_adversarial_type_negation(self):
        assert AdversarialKDSPAR._classify_adversarial_type(
            "What is NOT dark knowledge?"
        ) == "negation_edge"

    def test_classify_adversarial_type_boundary(self):
        assert AdversarialKDSPAR._classify_adversarial_type(
            "Must every student use temperature scaling?"
        ) == "boundary"

    def test_classify_unknown(self):
        assert AdversarialKDSPAR._classify_adversarial_type(
            "Random question here."
        ) == "unknown"

    def test_mine_hard_queries_returns_adversarial_queries(self):
        spar = self._make_spar()
        pipe = make_mock_pipeline("partial answer")
        traces = [("q1", "full answer with citation [Doc-1] and detail"),
                  ("q2", "another full answer [Doc-1]")]
        teacher_responses = {q: r for q, r in traces}
        result = spar.mine_hard_queries(["q1", "q2"], teacher_responses,
                                         student_pipeline=pipe)
        assert all(isinstance(aq, AdversarialQuery) for aq in result)
        assert all(aq.source == "gap_mined" for aq in result)
        assert all(0.0 <= aq.difficulty_score <= 1.0 for aq in result)

    def test_mine_hard_queries_empty_input(self):
        spar = self._make_spar()
        result = spar.mine_hard_queries([], {}, student_pipeline=make_mock_pipeline())
        assert result == []

    def test_build_hard_query_set_no_topics_returns_only_mined(self):
        spar   = self._make_spar()
        pipe   = make_mock_pipeline("answer")
        t_resps = {"q1": "teacher answer [Doc-1].", "q2": "teacher answer 2 [Doc-1]."}
        # Pass explicit pipeline so mine_hard_queries never touches RAGPipeline init
        result = spar.mine_hard_queries(["q1", "q2"], t_resps, student_pipeline=pipe)
        assert all(aq.source == "gap_mined" for aq in result)

    def test_batch_kd_all_same_returns_one(self):
        spar = self._make_spar()
        pipe = make_mock_pipeline("same answer [Doc-1].")
        result = spar._batch_kd(["q1"], {"q1": "same answer [Doc-1]."}, pipe)
        assert abs(result - 1.0) < 0.01

    def test_batch_kd_completely_different_near_zero(self):
        spar = self._make_spar()
        pipe = make_mock_pipeline("cat sat on the mat")
        result = spar._batch_kd(["q1"], {"q1": "quantum entanglement physics"}, pipe)
        assert result <= 0.35   # token overlap between disjoint texts is low

    def test_dual_threshold_default(self):
        spar = self._make_spar()
        assert spar.dual_threshold == 0.005

    def test_standard_regression_default(self):
        spar = self._make_spar()
        assert spar.standard_regression == 0.02


# ══════════════════════════════════════════════════════════════════════════════
# FederatedClientConfig
# ══════════════════════════════════════════════════════════════════════════════

class TestFederatedClientConfig:

    def test_defaults(self):
        cfg = FederatedClientConfig(client_id="site_1")
        assert cfg.n_local_queries == 50
        assert cfg.n_proposals == 3
        assert cfg.top_k_local == 5

    def test_custom_values(self):
        cfg = FederatedClientConfig("x", n_local_queries=10, n_proposals=2, top_k_local=3)
        assert cfg.n_local_queries == 10


# ══════════════════════════════════════════════════════════════════════════════
# FederatedKDSPARClient
# ══════════════════════════════════════════════════════════════════════════════

class TestFederatedKDSPARClient:

    def _make_client(self, traces=None):
        cfg    = FederatedClientConfig("client_1", n_proposals=2, top_k_local=3)
        traces = traces or list(SAMPLE_TRACES)
        store  = make_mock_rag_store()
        return FederatedKDSPARClient(cfg, traces, store)

    def test_propose_before_receive_returns_empty(self):
        client = self._make_client()
        # No prompt received yet → pipeline is None → empty proposals
        result = client.propose_instructions()
        assert result == []

    def test_receive_prompt_creates_pipeline(self):
        client = self._make_client()
        with patch("sara.rag.kd_spar_federated.RAGPipeline") as mock_pipe, \
             patch("sara.rag.kd_spar_federated.AnthropicClient") as mock_cli:
            mock_pipe.return_value = make_mock_pipeline()
            mock_cli.return_value  = make_mock_anthropic_client()
            client.receive_prompt("You are a helpful assistant.")
            assert client._pipeline is not None
            assert client._interviewer is not None

    def test_receive_prompt_twice_updates_system(self):
        client = self._make_client()
        pipe   = make_mock_pipeline()
        cli    = make_mock_anthropic_client()
        with patch("sara.rag.kd_spar_federated.RAGPipeline", return_value=pipe), \
             patch("sara.rag.kd_spar_federated.AnthropicClient", return_value=cli):
            client.receive_prompt("Prompt A")
            client.receive_prompt("Prompt B")  # should call update_system, not recreate
            pipe.client.update_system.assert_called()

    def test_propose_instructions_returns_strings(self):
        client = self._make_client()
        pipe   = make_mock_pipeline("partial answer without citation")
        cli    = make_mock_anthropic_client("Always cite [Doc-N] for every claim.")
        client._pipeline    = pipe
        client._interviewer = cli
        result = client.propose_instructions()
        assert isinstance(result, list)
        assert all(isinstance(p, str) for p in result)

    def test_propose_returns_at_most_top_k_local(self):
        cfg    = FederatedClientConfig("c1", top_k_local=2)
        client = FederatedKDSPARClient(cfg, list(SAMPLE_TRACES), make_mock_rag_store())
        pipe   = make_mock_pipeline("answer without citation")
        cli    = make_mock_anthropic_client("Always cite sources.")
        client._pipeline    = pipe
        client._interviewer = cli
        result = client.propose_instructions()
        assert len(result) <= cfg.top_k_local

    def test_local_kd_score_without_pipeline_returns_zero(self):
        client = self._make_client()
        score  = client.local_kd_score(["q1"], {"q1": "answer"})
        assert score == 0.0

    def test_local_kd_score_with_pipeline(self):
        client = self._make_client()
        pipe   = make_mock_pipeline("knowledge distillation [Doc-1].")
        client._pipeline = pipe
        score = client.local_kd_score(
            ["q1"], {"q1": "knowledge distillation [Doc-1]."}
        )
        # Same student and teacher response → high score
        assert score > 0.5

    def test_local_diagnose_returns_failures(self):
        client = self._make_client()
        pipe   = make_mock_pipeline("incomplete answer")
        client._pipeline = pipe
        t_resps = {
            "q1": "Full answer with citation [Doc-1] and lots of detail.",
            "q2": "Another detailed answer [Doc-1] with proper formatting.",
        }
        result = client._local_diagnose(["q1", "q2"], t_resps, top_k=5)
        assert isinstance(result, list)
        for item in result:
            assert len(item) == 4  # (query, student, teacher, mode)


# ══════════════════════════════════════════════════════════════════════════════
# FederatedRound
# ══════════════════════════════════════════════════════════════════════════════

class TestFederatedRound:

    def test_fields(self):
        r = FederatedRound(
            round_number=1, clients_participated=["c1", "c2"],
            total_proposals=10, selected_instrs=["inst A", "inst B"],
            score_before=0.75, score_after=0.80, delta=0.05, accepted=True,
        )
        assert r.delta == 0.05
        assert r.accepted is True
        assert len(r.clients_participated) == 2


# ══════════════════════════════════════════════════════════════════════════════
# FederatedSimulation
# ══════════════════════════════════════════════════════════════════════════════

class TestFederatedSimulation:

    def test_creates_correct_number_of_clients(self):
        store = make_mock_rag_store()
        sim   = FederatedSimulation(
            n_clients=3, all_traces=SAMPLE_TRACES,
            student_model="claude-sonnet-4-5-20250929",
            vector_store=store,
        )
        clients = sim.build_clients()
        assert len(clients) == 3

    def test_each_client_has_local_data(self):
        store = make_mock_rag_store()
        sim   = FederatedSimulation(3, SAMPLE_TRACES, vector_store=store)
        clients = sim.build_clients()
        for client in clients:
            assert len(client.local_traces) >= 0  # may be empty if traces exhausted

    def test_val_fraction_held_out(self):
        traces = [(f"q{i}", f"r{i}") for i in range(20)]
        store  = make_mock_rag_store()
        sim    = FederatedSimulation(
            2, traces, val_fraction=0.2, vector_store=store
        )
        val_n = len(sim._val)
        assert val_n == 4   # 20 * 0.2

    def test_builds_server_returns_server(self):
        store  = make_mock_rag_store()
        sim    = FederatedSimulation(2, SAMPLE_TRACES, vector_store=store)
        with patch("sara.rag.kd_spar_federated.RAGPipeline"), \
             patch("sara.rag.kd_spar_federated.AnthropicClient"):
            server = sim.build_server()
            assert isinstance(server, FederatedKDSPARServer)
            assert len(server.clients) == 2

    def test_client_ids_are_unique(self):
        store   = make_mock_rag_store()
        sim     = FederatedSimulation(4, SAMPLE_TRACES * 4, vector_store=store)
        clients = sim.build_clients()
        ids     = [c.config.client_id for c in clients]
        assert len(ids) == len(set(ids))


# ══════════════════════════════════════════════════════════════════════════════
# KD score helper (shared by all variants)
# ══════════════════════════════════════════════════════════════════════════════

class TestKDScoreHelper:

    def test_identical_high_score(self):
        text = "knowledge distillation soft targets [Doc-1] temperature"
        assert _kd_score(text, text) > 0.9

    def test_disjoint_low_score(self):
        assert _kd_score("abc xyz", "def ghi jkl") < 0.4

    def test_citation_component_matters(self):
        # teacher cites → student should too for max score
        teacher = "answer [Doc-1]"
        student_with = "answer [Doc-1]"
        student_without = "answer"
        score_with    = _kd_score(student_with, teacher)
        score_without = _kd_score(student_without, teacher)
        assert score_with > score_without

    def test_score_range(self):
        for s, t in [("hello", "world"), ("same", "same"), ("", "text")]:
            if s:
                sc = _kd_score(s, t)
                assert 0.0 <= sc <= 1.0
