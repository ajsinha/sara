"""
tests/conftest.py
=================
Shared pytest fixtures for the knowledge distillation test suite.

All fixtures produce synthetic in-memory data so the tests run
without downloading datasets, model weights, or making API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── Reproducibility ────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def set_seed():
    """Fix random seed for reproducibility across all tests."""
    torch.manual_seed(42)


# ── Device ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def device() -> str:
    return "cpu"   # tests always run on CPU for CI compatibility


# ── Minimal classification models ─────────────────────────────────────────────

@pytest.fixture
def tiny_teacher(device) -> nn.Module:
    """A minimal 2-layer teacher network for unit tests."""
    return nn.Sequential(
        nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 5)
    ).to(device)


@pytest.fixture
def tiny_student(device) -> nn.Module:
    """A smaller 2-layer student network for unit tests."""
    return nn.Sequential(
        nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 5)
    ).to(device)


@pytest.fixture
def tiny_loader(device) -> tuple[DataLoader, DataLoader]:
    """
    Tiny synthetic (features, labels) data loaders.
    Train: 64 samples  |  Val: 16 samples
    Input dim: 16   |  Classes: 5
    """
    def _make(n):
        x = torch.randn(n, 16)
        y = torch.randint(0, 5, (n,))
        return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)

    return _make(64), _make(16)


# ── Minimal CNN models (for vision tests) ─────────────────────────────────────

@pytest.fixture
def tiny_cnn_teacher(device) -> nn.Module:
    """Minimal CNN teacher for feature-extraction tests."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 5),
    ).to(device)


@pytest.fixture
def tiny_cnn_student(device) -> nn.Module:
    """Minimal CNN student."""
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 5),
    ).to(device)


@pytest.fixture
def image_loader() -> tuple[DataLoader, DataLoader]:
    """Synthetic image data loaders: 3×8×8 images, 5 classes."""
    def _make(n):
        x = torch.randn(n, 3, 8, 8)
        y = torch.randint(0, 5, (n,))
        return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)
    return _make(32), _make(8)


# ── Mock Anthropic client ─────────────────────────────────────────────────────

@pytest.fixture
def mock_anthropic_response():
    """
    Returns a factory that creates a mock Anthropic API response object.

    Usage:
        response = mock_anthropic_response("My answer [Doc-1].")
    """
    def _factory(text: str = "Answer [Doc-1]."):
        resp = MagicMock()
        resp.content = [MagicMock(text=text)]
        return resp
    return _factory


@pytest.fixture
def mock_anthropic_client(mock_anthropic_response):
    """
    Patches anthropic.Anthropic so no real API calls are made.
    The mocked messages.create() returns a response with '[Doc-1]' in it.
    """
    with patch("anthropic.Anthropic") as mock_cls:
        instance = MagicMock()
        instance.messages.create.return_value = mock_anthropic_response()
        mock_cls.return_value = instance
        yield instance


# ── Mock ChromaDB ──────────────────────────────────────────────────────────────

@pytest.fixture
def mock_chroma(tmp_path):
    """
    Patches chromadb.PersistentClient with an in-memory mock so tests
    don't write to disk or require chromadb installed.
    """
    with patch("chromadb.PersistentClient") as mock_cls:
        # Mock collection
        col = MagicMock()
        col.count.return_value = 2
        col.query.return_value = {
            "documents": [["Passage about KD.", "Another passage."]],
            "metadatas": [
                [{"source": "doc1.txt", "chunk": 0},
                 {"source": "doc2.txt", "chunk": 0}],
            ],
        }
        # Mock client
        client = MagicMock()
        client.get_or_create_collection.return_value = col
        mock_cls.return_value = client
        yield col


# ── Mock SentenceTransformer embedding function ────────────────────────────────

@pytest.fixture(autouse=False)
def mock_sentence_transformer():
    """
    Patches chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction
    so sentence-transformers doesn't need to be installed for basic tests.
    """
    with patch(
        "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
    ) as mock_cls:
        mock_cls.return_value = MagicMock()
        yield mock_cls
