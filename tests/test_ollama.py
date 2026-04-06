# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.4.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
"""
tests/test_ollama.py
====================
Unit tests for the Ollama backend.

All tests mock the HTTP layer so no real Ollama server is needed.
The tests verify the client interface, compatibility shim, and pipeline
all work correctly against mocked responses.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# ── Helpers ────────────────────────────────────────────────────────────────

def mock_ollama_running():
    """Patch requests.get to simulate a running Ollama server."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [
            {"name": "llama3.1:8b"},
            {"name": "llama3.2:3b"},
            {"name": "qwen2.5:7b"},
        ]
    }
    return mock_resp

def mock_ollama_chat_response(content: str = "Knowledge distillation [Doc-1]."):
    """Patch requests.post to return a chat response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "message": {"role": "assistant", "content": content}
    }
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ══════════════════════════════════════════════════════════════════════════════
# check_ollama_running
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckOllamaRunning:

    def test_returns_true_when_server_up(self):
        from sara.rag.ollama_client import check_ollama_running
        with patch("requests.get", return_value=mock_ollama_running()):
            assert check_ollama_running() is True

    def test_returns_false_when_server_down(self):
        from sara.rag.ollama_client import check_ollama_running
        import requests
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError):
            assert check_ollama_running() is False

    def test_returns_false_on_non_200(self):
        from sara.rag.ollama_client import check_ollama_running
        m = MagicMock(); m.status_code = 500
        with patch("requests.get", return_value=m):
            assert check_ollama_running() is False


# ══════════════════════════════════════════════════════════════════════════════
# list_available_models
# ══════════════════════════════════════════════════════════════════════════════

class TestListAvailableModels:

    def test_returns_model_names(self):
        from sara.rag.ollama_client import list_available_models
        with patch("requests.get", return_value=mock_ollama_running()):
            models = list_available_models()
        assert "llama3.1:8b" in models
        assert "llama3.2:3b" in models
        assert "qwen2.5:7b"  in models

    def test_raises_on_connection_error(self):
        from sara.rag.ollama_client import list_available_models
        import requests
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError), \
             pytest.raises(ConnectionError):
            list_available_models()


# ══════════════════════════════════════════════════════════════════════════════
# OllamaClient
# ══════════════════════════════════════════════════════════════════════════════

class TestOllamaClient:

    def _make_client(self, model="llama3.1:8b"):
        from sara.rag.ollama_client import OllamaClient
        with patch("requests.get", return_value=mock_ollama_running()), \
             patch("requests.post", return_value=mock_ollama_chat_response()):
            return OllamaClient(model_id=model)

    def test_instantiation_succeeds(self):
        client = self._make_client()
        assert client.model_id == "llama3.1:8b"

    def test_raises_if_server_not_running(self):
        from sara.rag.ollama_client import OllamaClient
        import requests
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError), \
             pytest.raises(ConnectionError):
            OllamaClient("llama3.1:8b")

    def test_generate_calls_chat_api(self):
        from sara.rag.ollama_client import OllamaClient
        from sara.rag.pipeline import Document
        mock_resp = mock_ollama_chat_response("Answer citing [Doc-1].")
        with patch("requests.get", return_value=mock_ollama_running()), \
             patch("requests.post", return_value=mock_resp) as mock_post:
            client = OllamaClient("llama3.1:8b")
            docs   = [Document("Passage text.", "source.txt", 0)]
            result = client.generate("What is KD?", docs)
        mock_post.assert_called_once()
        assert "Answer" in result

    def test_update_system_changes_prompt(self):
        client = self._make_client()
        client.update_system("New system prompt.")
        assert client.system == "New system prompt."

    def test_default_system_prompt_set(self):
        client = self._make_client()
        assert len(client.system) > 10

    def test_model_id_stored(self):
        client = self._make_client("llama3.2:3b")
        assert client.model_id == "llama3.2:3b"

    def test_generate_returns_string(self):
        from sara.rag.ollama_client import OllamaClient
        from sara.rag.pipeline import Document
        with patch("requests.get", return_value=mock_ollama_running()), \
             patch("requests.post", return_value=mock_ollama_chat_response("Hello")):
            client = OllamaClient("llama3.1:8b")
            docs   = [Document("ctx", "src", 0)]
            result = client.generate("q", docs)
        assert isinstance(result, str)
        assert len(result) > 0


# ══════════════════════════════════════════════════════════════════════════════
# _OllamaMsgCompat (Anthropic SDK compatibility shim)
# ══════════════════════════════════════════════════════════════════════════════

class TestOllamaMsgCompat:

    def _make_compat(self):
        from sara.rag.ollama_client import OllamaClient
        with patch("requests.get", return_value=mock_ollama_running()), \
             patch("requests.post", return_value=mock_ollama_chat_response("Proposal text.")):
            client = OllamaClient("llama3.1:8b")
        return client._client

    def test_has_messages_attribute(self):
        compat = self._make_compat()
        assert hasattr(compat, "messages")

    def test_create_returns_response_with_content(self):
        compat = self._make_compat()
        with patch("requests.post", return_value=mock_ollama_chat_response("My proposal")):
            resp = compat.create(
                model="llama3.1:8b", max_tokens=80,
                messages=[{"role": "user", "content": "Fix this issue."}],
            )
        assert hasattr(resp, "content")
        assert len(resp.content) > 0
        assert resp.content[0].text == "My proposal"

    def test_create_without_messages(self):
        compat = self._make_compat()
        with patch("requests.post", return_value=mock_ollama_chat_response("OK")):
            resp = compat.create(model="llama3.1:8b")
        assert resp.content[0].text == "OK"


# ══════════════════════════════════════════════════════════════════════════════
# OllamaResponse
# ══════════════════════════════════════════════════════════════════════════════

class TestOllamaResponse:

    def test_content_list_with_text(self):
        from sara.rag.ollama_client import _OllamaResponse
        resp = _OllamaResponse("Hello world")
        assert resp.content[0].text == "Hello world"

    def test_empty_text(self):
        from sara.rag.ollama_client import _OllamaResponse
        resp = _OllamaResponse("")
        assert resp.content[0].text == ""


# ══════════════════════════════════════════════════════════════════════════════
# OllamaRAGPipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestOllamaRAGPipeline:

    def _make_pipeline(self, model="llama3.1:8b", mock_store=None, mock_chroma=None):
        from sara.rag.ollama_pipeline import OllamaRAGPipeline

        # Mock out RAGVectorStore
        if mock_store is None:
            mock_store = MagicMock()
            mock_store.count = 3
            mock_store.search.return_value = []
            mock_store.add_documents.return_value = 2

        with patch("requests.get", return_value=mock_ollama_running()), \
             patch("requests.post", return_value=mock_ollama_chat_response()), \
             patch("chromadb.PersistentClient"), \
             patch("chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"):
            return OllamaRAGPipeline(
                model_id=model, store=mock_store, auto_pull=False
            ), mock_store

    def test_instantiation(self):
        pipe, _ = self._make_pipeline()
        assert pipe.model_id == "llama3.1:8b"

    def test_query_returns_rag_response(self):
        from sara.rag.pipeline import Document
        pipe, store = self._make_pipeline()
        store.search.return_value = [Document("Passage", "doc.txt", 0)]
        with patch("requests.post", return_value=mock_ollama_chat_response("Answer [Doc-1].")):
            resp = pipe.query("What is KD?", return_context=True)
        assert resp.answer == "Answer [Doc-1]."
        assert resp.model_used == "llama3.1:8b"

    def test_query_no_results_returns_fallback(self):
        pipe, store = self._make_pipeline()
        store.search.return_value = []
        resp = pipe.query("Obscure question")
        assert "No relevant passages" in resp.answer

    def test_ingest_calls_store(self):
        pipe, store = self._make_pipeline()
        n = pipe.ingest({"doc.txt": "Some text about KD here."})
        store.add_documents.assert_called_once()

    def test_citations_extracted(self):
        from sara.rag.pipeline import Document
        pipe, store = self._make_pipeline()
        store.search.return_value = [Document("Passage", "doc.txt", 0)]
        with patch("requests.post",
                   return_value=mock_ollama_chat_response(
                       "See [Doc-1] and also [Doc-2] for details.")):
            resp = pipe.query("What is KD?")
        assert "[Doc-1]" in resp.citations
        assert "[Doc-2]" in resp.citations


# ══════════════════════════════════════════════════════════════════════════════
# OllamaTeacherSpec
# ══════════════════════════════════════════════════════════════════════════════

class TestOllamaTeacherSpec:

    def test_defaults(self):
        from sara.rag.ollama_kd_spar import OllamaTeacherSpec
        spec = OllamaTeacherSpec("citation_expert", "llama3.1:8b")
        assert spec.name == "citation_expert"
        assert spec.model_id == "llama3.1:8b"
        assert spec.weight == 1.0
        assert not spec.is_primary

    def test_custom_values(self):
        from sara.rag.ollama_kd_spar import OllamaTeacherSpec
        spec = OllamaTeacherSpec(
            "reasoning", "qwen2.5:7b",
            system_prompt="Reason step by step.",
            weight=2.0, is_primary=True,
        )
        assert spec.is_primary
        assert spec.weight == 2.0
        assert "Reason" in spec.system_prompt


# ══════════════════════════════════════════════════════════════════════════════
# Model config constants
# ══════════════════════════════════════════════════════════════════════════════

class TestModelConstants:

    def test_teacher_model_strings(self):
        from sara.rag.ollama_client import (
            OLLAMA_TEACHER_MODEL, OLLAMA_STUDENT_MODEL,
            OLLAMA_ALT_TEACHER, OLLAMA_ALT_STUDENT,
        )
        assert OLLAMA_TEACHER_MODEL == "llama3.1:8b"
        assert OLLAMA_STUDENT_MODEL == "llama3.2:3b"
        assert "qwen" in OLLAMA_ALT_TEACHER
        assert "3b" in OLLAMA_ALT_STUDENT or "qwen" in OLLAMA_ALT_STUDENT

    def test_default_url(self):
        from sara.rag.ollama_client import OLLAMA_DEFAULT_URL
        assert "11434" in OLLAMA_DEFAULT_URL

    def test_default_system_prompt_non_empty(self):
        from sara.rag.ollama_client import OLLAMA_DEFAULT_SYSTEM
        assert len(OLLAMA_DEFAULT_SYSTEM) > 20
        assert "context" in OLLAMA_DEFAULT_SYSTEM.lower()
