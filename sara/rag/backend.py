# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.2.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
"""
sara.rag.backend
================
Provider-agnostic backend factory.

This is the **single place** you change to switch between Anthropic,
Ollama, or any future provider.  All examples and experiments import
from here — nothing hard-codes a specific client or pipeline class.

Configuration priority (highest → lowest)
------------------------------------------
1. Environment variable  SARA_BACKEND=ollama | anthropic
2. Config file           configs/backend.yaml  (backend: ollama)
3. Default               ollama  (so FOSS works out-of-the-box)

Model configuration priority
-----------------------------
1. Explicit kwargs passed to get_pipeline() / get_client()
2. Environment variables  SARA_TEACHER_MODEL, SARA_STUDENT_MODEL
3. Config file            configs/backend.yaml
4. Built-in defaults per backend

Usage
-----
Anywhere in examples or experiments, replace direct imports with::

    from sara.rag.backend import get_pipeline, get_client, get_spar, cfg

    store    = RAGVectorStore()
    teacher  = get_pipeline("teacher", store=store)
    student  = get_pipeline("student", store=store)
    spar     = get_spar(store=store)

The rest of the code is identical regardless of backend.

Switching backends
------------------
Option A — environment variable (quickest, no file needed)::

    export SARA_BACKEND=ollama
    export SARA_TEACHER_MODEL=llama3.1:8b
    export SARA_STUDENT_MODEL=llama3.2:3b

Option B — config file  configs/backend.yaml::

    backend: ollama
    teacher_model: llama3.1:8b
    student_model: llama3.2:3b
    ollama_base_url: http://localhost:11434

Option C — pass kwargs directly::

    pipe = get_pipeline("teacher", model_id="qwen2.5:7b", store=store)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# ── Try to load yaml (optional; falls back to env/defaults if missing) ────────
try:
    import yaml as _yaml
    _HAVE_YAML = True
except ImportError:
    _HAVE_YAML = False

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "backend.yaml"


def _load_config() -> dict:
    """Load configs/backend.yaml if it exists, else return empty dict."""
    if _HAVE_YAML and _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return _yaml.safe_load(f) or {}
    return {}


# ── Resolved configuration (computed once at import time) ─────────────────────
_cfg = _load_config()


def _resolve(key: str, default: str, env_var: Optional[str] = None) -> str:
    """Priority: env_var > config file > default."""
    if env_var:
        v = os.environ.get(env_var, "").strip()
        if v:
            return v
    return str(_cfg.get(key, default))


# Public config object — read this anywhere with: from sara.rag.backend import cfg
cfg = {
    "backend":       _resolve("backend",       "ollama",          "SARA_BACKEND"),
    "teacher_model": _resolve("teacher_model", "llama3.1:8b",     "SARA_TEACHER_MODEL"),
    "student_model": _resolve("student_model", "llama3.2:3b",     "SARA_STUDENT_MODEL"),
    "ollama_url":    _resolve("ollama_base_url","http://localhost:11434", "OLLAMA_BASE_URL"),
    # Anthropic-specific (only used when backend=anthropic)
    "anthropic_teacher": _resolve("anthropic_teacher", "claude-3-5-sonnet-20241022", None),
    "anthropic_student": _resolve("anthropic_student", "claude-sonnet-4-5-20250929",  None),
}

# If backend=anthropic override teacher/student with anthropic defaults
# unless user explicitly set SARA_TEACHER_MODEL
if cfg["backend"] == "anthropic" and not os.environ.get("SARA_TEACHER_MODEL"):
    cfg["teacher_model"] = cfg["anthropic_teacher"]
if cfg["backend"] == "anthropic" and not os.environ.get("SARA_STUDENT_MODEL"):
    cfg["student_model"] = cfg["anthropic_student"]


def _check_anthropic() -> None:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set.\n"
            "Either set it:  export ANTHROPIC_API_KEY='sk-ant-...'\n"
            "Or switch to Ollama:  export SARA_BACKEND=ollama"
        )


def _check_ollama() -> None:
    from sara.rag.ollama_client import check_ollama_running
    if not check_ollama_running(cfg["ollama_url"]):
        raise ConnectionError(
            f"Ollama server not reachable at {cfg['ollama_url']}.\n"
            "Start it:   ollama serve\n"
            "Install:    curl -fsSL https://ollama.com/install.sh | sh"
        )


# ── Public factory functions ──────────────────────────────────────────────────

def get_pipeline(
    role: str = "teacher",
    model_id: Optional[str] = None,
    store=None,
    system_prompt: Optional[str] = None,
    auto_pull: bool = True,
):
    """
    Return a RAG pipeline for the given role using the configured backend.

    Parameters
    ----------
    role        : "teacher" or "student" (selects default model from cfg)
    model_id    : Override the model (e.g. "qwen2.5:7b")
    store       : RAGVectorStore to use (creates new one if None)
    system_prompt : Override system prompt
    auto_pull   : Auto-pull Ollama model if not present (Ollama backend only)

    Returns
    -------
    OllamaRAGPipeline or RAGPipeline depending on backend
    """
    from sara.rag.pipeline import RAGVectorStore

    if store is None:
        store = RAGVectorStore()

    if model_id is None:
        model_id = cfg["teacher_model"] if role == "teacher" else cfg["student_model"]

    backend = cfg["backend"].lower()

    if backend == "ollama":
        from sara.rag.ollama_pipeline import OllamaRAGPipeline
        _check_ollama()
        return OllamaRAGPipeline(
            model_id=model_id, store=store,
            system_prompt=system_prompt,
            base_url=cfg["ollama_url"],
            auto_pull=auto_pull,
        )
    elif backend == "anthropic":
        from sara.rag.pipeline import RAGPipeline
        _check_anthropic()
        return RAGPipeline(
            model_id=model_id, store=store,
            system_prompt=system_prompt,
        )
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            "Set SARA_BACKEND=ollama or SARA_BACKEND=anthropic"
        )


def get_client(
    role: str = "teacher",
    model_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
):
    """
    Return a chat client for the given role.

    Returns OllamaClient or AnthropicClient depending on backend.
    """
    if model_id is None:
        model_id = cfg["teacher_model"] if role == "teacher" else cfg["student_model"]

    backend = cfg["backend"].lower()

    if backend == "ollama":
        from sara.rag.ollama_client import OllamaClient
        _check_ollama()
        return OllamaClient(
            model_id=model_id,
            system_prompt=system_prompt,
            base_url=cfg["ollama_url"],
        )
    elif backend == "anthropic":
        from sara.rag.pipeline import AnthropicClient
        _check_anthropic()
        return AnthropicClient(
            model_id=model_id,
            system_prompt=system_prompt,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


def get_spar(
    store=None,
    teacher_model: Optional[str] = None,
    student_model: Optional[str] = None,
):
    """
    Return the appropriate KD-SPAR class instance for the configured backend.

    Returns OllamaKDSPAR or KDSPAR depending on backend.
    """
    from sara.rag.pipeline import RAGVectorStore

    if store is None:
        store = RAGVectorStore()

    t = teacher_model or cfg["teacher_model"]
    s = student_model or cfg["student_model"]
    backend = cfg["backend"].lower()

    if backend == "ollama":
        from sara.rag.ollama_kd_spar import OllamaKDSPAR
        _check_ollama()
        return OllamaKDSPAR(
            teacher_model=t, student_model=s,
            vector_store=store, base_url=cfg["ollama_url"],
        )
    elif backend == "anthropic":
        from sara.rag.kd_spar import KDSPAR
        _check_anthropic()
        return KDSPAR(
            teacher_model=t, student_model=s,
            vector_store=store,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


def describe() -> str:
    """Return a human-readable summary of the current backend configuration."""
    lines = [
        f"Backend   : {cfg['backend']}",
        f"Teacher   : {cfg['teacher_model']}",
        f"Student   : {cfg['student_model']}",
    ]
    if cfg["backend"] == "ollama":
        lines.append(f"Ollama URL: {cfg['ollama_url']}")
    return "\n".join(lines)
