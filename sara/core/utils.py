# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.6.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
from __future__ import annotations
"""
sara.core.utils
=============
Shared utilities: model profiling, hyperparameter guidance, config loading.
"""


import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


# ── Model profiler ─────────────────────────────────────────────────────────────

@dataclass
class ProfileResult:
    """Results from :func:`profile_model`."""
    model_name:     str
    latency_ms:     float
    throughput_sps: float   # samples per second
    params_mb:      float   # parameter memory in megabytes
    params_count:   int

    def __str__(self) -> str:
        return (
            f"[{self.model_name}]  "
            f"latency={self.latency_ms:.2f}ms  "
            f"throughput={self.throughput_sps:.0f} sps  "
            f"params={self.params_mb:.1f}MB ({self.params_count:,})"
        )


def profile_model(
    model: nn.Module,
    dummy_input: torch.Tensor,
    n: int = 200,
    model_name: str = "model",
    warmup: int = 20,
) -> ProfileResult:
    """
    Measure inference latency, throughput, and parameter footprint.

    Parameters
    ----------
    model       : PyTorch model to profile (must accept `dummy_input`)
    dummy_input : Representative input tensor (single sample or batch)
    n           : Number of timed forward passes
    model_name  : Label used in printed output
    warmup      : Number of warm-up passes before timing

    Returns
    -------
    ProfileResult dataclass

    Examples
    --------
    >>> result = profile_model(model, torch.randn(1, 3, 224, 224))
    >>> print(result)
    """
    model.eval()
    device = dummy_input.device

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - t0) / n * 1000.0

    # Count trainable + non-trainable parameters
    params = sum(p.numel() for p in model.parameters())
    params_mb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / 1e6

    result = ProfileResult(
        model_name     = model_name,
        latency_ms     = round(elapsed_ms, 3),
        throughput_sps = round(1000.0 / elapsed_ms, 1),
        params_mb      = round(params_mb, 2),
        params_count   = params,
    )
    print(result)
    return result


def compare_profiles(
    teacher: ProfileResult,
    student: ProfileResult,
) -> dict[str, float]:
    """
    Print and return speedup / compression ratios.

    Parameters
    ----------
    teacher : ProfileResult from the teacher model
    student : ProfileResult from the student model

    Returns
    -------
    Dict with keys 'speedup' and 'compression'
    """
    speedup     = round(teacher.latency_ms / max(student.latency_ms, 1e-9), 2)
    compression = round(teacher.params_mb  / max(student.params_mb,  1e-9), 2)
    print(f"  Speedup:     {speedup}×")
    print(f"  Compression: {compression}×  "
          f"({teacher.params_mb:.1f}MB → {student.params_mb:.1f}MB)")
    return {"speedup": speedup, "compression": compression}


# ── Hyperparameter recommender ─────────────────────────────────────────────────

def recommend_hyperparams(
    dataset_size: int,
    capacity_ratio: float,
    data_available: bool = True,
    is_nlp: bool = False,
) -> dict[str, Any]:
    """
    Rule-based hyperparameter recommender for distillation.

    Parameters
    ----------
    dataset_size    : Number of training samples
    capacity_ratio  : teacher_params / student_params  (e.g. 50M/5M = 10.0)
    data_available  : False → data-free mode
    is_nlp          : True → NLP-tuned temperature range

    Returns
    -------
    Dict with keys 'T' (temperature), 'alpha', and 'note'

    Examples
    --------
    >>> rec = recommend_hyperparams(50000, 10.0, is_nlp=False)
    >>> print(rec)
    {'T': 4.0, 'alpha': 0.5, 'note': 'medium capacity gap: standard settings'}
    """
    if not data_available:
        return {"T": 5.0, "alpha": 0.9, "note": "data-free: lean heavily on teacher"}

    if capacity_ratio > 20:
        T, alpha = (6.0 if not is_nlp else 12.0), 0.7
        note = "wide capacity gap: high T, high alpha"
    elif capacity_ratio > 5:
        T, alpha = (4.0 if not is_nlp else 8.0), 0.5
        note = "medium capacity gap: standard settings"
    else:
        T, alpha = (2.5 if not is_nlp else 4.0), 0.4
        note = "close capacity: low T, lower alpha"

    if dataset_size < 5_000:
        alpha = min(alpha + 0.15, 0.85)
        note += " [boosted alpha for small dataset]"

    return {"T": T, "alpha": alpha, "note": note}


# ── YAML config loader ─────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Requires PyYAML (``pip install pyyaml``).  Returns an empty dict if
    PyYAML is not installed — callers should fall back to defaults.

    Parameters
    ----------
    path : Path to the YAML file

    Returns
    -------
    Dict of configuration values

    Raises
    ------
    FileNotFoundError if the file does not exist.

    Examples
    --------
    >>> cfg = load_config("configs/vision_cifar10.yaml")
    >>> epochs = cfg.get("epochs", 30)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    try:
        import yaml  # type: ignore
    except ImportError:
        import warnings
        warnings.warn(
            "PyYAML not installed — returning empty config. "
            "Install with: pip install pyyaml",
            stacklevel=2,
        )
        return {}

    with open(p) as fh:
        return yaml.safe_load(fh) or {}


# ══════════════════════════════════════════════════════════════════════════
# Shared RAG scoring utilities
# Canonical definitions — imported by evaluation, migration, kd_spar, etc.
# ══════════════════════════════════════════════════════════════════════════

import re as _re

CITATION_RE = _re.compile(r"\[Doc-\d+\]")

HEDGE_WORDS: tuple[str, ...] = (
    "may", "might", "could", "possibly", "perhaps",
    "it appears", "it seems", "i'm not certain",
    "i cannot confirm", "unclear from the provided",
)

# Single canonical system prompt — used by both Anthropic and Ollama clients
DEFAULT_SYSTEM_PROMPT = (
    "You are a precise knowledge assistant. "
    "Answer questions using ONLY the provided context passages. "
    "Cite sources inline as [Doc-N] where N is the passage number. "
    "If the context does not contain the answer, say: "
    "'I cannot find this in the provided context.' "
    "Express uncertainty explicitly when evidence is partial."
)


def jaccard(a: str, b: str) -> float:
    """Token-overlap Jaccard similarity between two strings."""
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(len(sa | sb), 1)


def kd_score(student: str, teacher: str) -> float:
    """Composite KD alignment: 0.3·citation_match + 0.7·Jaccard."""
    t_cited = bool(CITATION_RE.search(teacher))
    s_cited = bool(CITATION_RE.search(student))
    cit     = 1.0 if (not t_cited) or s_cited else 0.0
    return round(0.3 * cit + 0.7 * jaccard(student, teacher), 4)


# ── BERTScore-based scoring (Enhancement 2) ──────────────────────────────

_BERT_MODEL = None

def _get_bert_model():
    """Lazy-load sentence-transformers model for BERTScore."""
    global _BERT_MODEL
    if _BERT_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _BERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            return None
    return _BERT_MODEL


def bert_similarity(a: str, b: str) -> float:
    """Cosine similarity using sentence-transformers embeddings."""
    model = _get_bert_model()
    if model is None:
        return jaccard(a, b)  # fallback
    import numpy as np
    embs = model.encode([a, b], normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


def kd_score_v2(student: str, teacher: str, use_bert: bool = True) -> float:
    """Enhanced KD score: 0.3·citation + 0.5·semantic + 0.2·jaccard.

    Falls back to kd_score() if sentence-transformers is not available.
    """
    t_cited = bool(CITATION_RE.search(teacher))
    s_cited = bool(CITATION_RE.search(student))
    cit     = 1.0 if (not t_cited) or s_cited else 0.0

    if use_bert and _get_bert_model() is not None:
        sem = bert_similarity(student, teacher)
        jac = jaccard(student, teacher)
        return round(0.3 * cit + 0.5 * sem + 0.2 * jac, 4)
    else:
        return kd_score(student, teacher)


def batch_kd_score(queries: list, teacher_responses: dict, pipeline) -> float:
    """Mean KD score across a list of queries against a RAG pipeline."""
    scores = []
    for q in queries:
        if q not in teacher_responses:
            continue
        try:
            scores.append(kd_score(
                pipeline.query(q, return_context=False).answer,
                teacher_responses[q],
            ))
        except Exception:
            scores.append(0.0)
    return sum(scores) / max(len(scores), 1)


def interpret_ab_gap(gap: float) -> str:
    """Return a human-readable interpretation of the A−B ablation gap."""
    if gap > 0.02:  return "strong — gap > 0.02 provides compelling evidence"
    if gap > 0.01:  return "moderate — gap 0.01–0.02 supports the claim"
    if gap > 0.005: return "suggestive — gap in the 0.005–0.01 range"
    if gap > 0.0:   return "marginal — positive but near noise floor"
    return "negative — external proposer matched or exceeded self-proposed"
