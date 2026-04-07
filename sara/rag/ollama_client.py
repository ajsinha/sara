# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.7.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ajsinha/sara
from __future__ import annotations
from sara.core.utils import DEFAULT_SYSTEM_PROMPT  # noqa
"""
sara.rag.ollama_client
=====================
Drop-in replacement for :class:`sara.rag.pipeline.AnthropicClient` that calls
a locally-running Ollama server instead of the Anthropic API.

Zero API costs · No rate limits · Fully reproducible · Works offline.

Supported Ollama API endpoints
-------------------------------
  POST /api/chat            – multi-turn chat (used here)
  POST /api/generate        – single-turn generation (fallback)
  GET  /api/tags            – list available models
  POST /api/pull            – pull a model

Recommended local model pairs
-------------------------------
  Teacher : llama3.1:8b    (larger, higher-quality responses)
  Student : llama3.2:3b    (smaller, faster — the model we are improving)

  Teacher : qwen2.5:7b     (alternative teacher with strong instruction following)
  Student : llama3.2:3b

  Teacher : llama3.1:8b
  Student : qwen2.5:3b     (Qwen student variant)

Setup on OryxPro (Pop!_OS)
---------------------------
  # Install Ollama
  curl -fsSL https://ollama.com/install.sh | sh

  # Start the Ollama server (or it auto-starts as a systemd service)
  ollama serve &

  # Pull the models (one-time download)
  ollama pull llama3.1:8b    # ~4.7 GB
  ollama pull llama3.2:3b    # ~2.0 GB
  ollama pull qwen2.5:7b     # ~4.4 GB  (optional alternative teacher)

  # Verify
  ollama list

Requirements
------------
  pip install requests
  (no API key needed)
"""


import json
import time
from typing import Optional

import requests

from sara.rag.pipeline import Document

# ── Default model strings ──────────────────────────────────────────────────
OLLAMA_DEFAULT_URL     = "http://localhost:11434"
OLLAMA_TEACHER_MODEL   = "llama3.1:8b"
OLLAMA_STUDENT_MODEL   = "llama3.2:3b"
OLLAMA_ALT_TEACHER     = "qwen2.5:7b"          # alternative teacher
OLLAMA_ALT_STUDENT     = "qwen2.5:3b"          # alternative student

OLLAMA_DEFAULT_SYSTEM = (
    "You are a precise knowledge assistant. "
    "Answer questions using ONLY the provided context passages. "
    "Cite sources inline as [Doc-N] where N is the passage number. "
    "If the context does not contain the answer, say: "
    "'I cannot find this in the provided context.' "
    "Express uncertainty explicitly when evidence is partial."
)


# ── Connectivity helpers ───────────────────────────────────────────────────

def check_ollama_running(base_url: str = OLLAMA_DEFAULT_URL) -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def list_available_models(base_url: str = OLLAMA_DEFAULT_URL) -> list[str]:
    """Return list of model names currently pulled on the local server."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except requests.exceptions.RequestException as exc:
        raise ConnectionError(
            f"Cannot reach Ollama at {base_url}. "
            "Is the server running? Try: ollama serve"
        ) from exc


def pull_model(model: str, base_url: str = OLLAMA_DEFAULT_URL) -> None:
    """
    Pull a model from the Ollama registry (blocks until complete).
    Shows progress dots to stdout.
    """
    print(f"Pulling '{model}' from Ollama registry … (this may take several minutes)")
    resp = requests.post(
        f"{base_url}/api/pull",
        json={"name": model, "stream": True},
        stream=True, timeout=600,
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line:
            data = json.loads(line)
            status = data.get("status", "")
            if "pulling" in status or "verifying" in status:
                print(".", end="", flush=True)
            elif data.get("status") == "success":
                print(f"\n'{model}' pulled successfully.")
                return
    print(f"\n'{model}' pull complete.")


def ensure_model(model: str, base_url: str = OLLAMA_DEFAULT_URL) -> None:
    """Pull the model if it is not already available locally."""
    available = list_available_models(base_url)
    # Ollama may store model names with or without the :latest tag
    short = model.split(":")[0]
    if not any(m.startswith(short) or m == model for m in available):
        print(f"Model '{model}' not found locally. Pulling …")
        pull_model(model, base_url)
    else:
        print(f"Model '{model}' available locally. ✓")


# ── OllamaClient ──────────────────────────────────────────────────────────

class OllamaClient:
    """
    Ollama-backed LLM client with the same interface as
    :class:`sara.rag.pipeline.AnthropicClient`.

    All calls go to a locally-running ``ollama serve`` instance.

    Parameters
    ----------
    model_id      : Ollama model string (e.g. ``"llama3.1:8b"``)
    system_prompt : System instructions  (swappable via :meth:`update_system`)
    base_url      : Ollama server URL (default ``http://localhost:11434``)
    max_tokens    : Maximum tokens to generate
    temperature   : Sampling temperature (0 = deterministic)
    timeout       : Per-request HTTP timeout in seconds

    Examples
    --------
    >>> client = OllamaClient("llama3.1:8b")
    >>> answer = client.generate("What is KD?", context_docs=[...])
    """

    def __init__(
        self,
        model_id:     str = OLLAMA_TEACHER_MODEL,
        system_prompt: Optional[str] = None,
        base_url:     str = OLLAMA_DEFAULT_URL,
        max_tokens:   int = 1024,
        temperature:  float = 0.1,
        timeout:      int = 120,
    ) -> None:
        self.model_id    = model_id
        self.system      = system_prompt or OLLAMA_DEFAULT_SYSTEM
        self.base_url    = base_url.rstrip("/")
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.timeout     = timeout

        # Validate server is reachable at construction time
        if not check_ollama_running(self.base_url):
            raise ConnectionError(
                f"Ollama server not reachable at {self.base_url}.\n"
                "Start it with:  ollama serve\n"
                "Or install from: https://ollama.com/install.sh"
            )

    def generate(self, query: str, context_docs: list[Document]) -> str:
        """
        Generate a grounded answer from the query and retrieved passages.

        Parameters
        ----------
        query        : User question
        context_docs : Retrieved Document objects (numbered [Doc-1], …)

        Returns
        -------
        Model response string
        """
        context_block = "\n\n".join(
            f"[Doc-{i+1}] (source: {d.source})\n{d.content}"
            for i, d in enumerate(context_docs)
        )
        user_message = (
            f"Context passages:\n{context_block}\n\n"
            f"Question: {query}"
        )

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system",  "content": self.system},
                {"role": "user",    "content": user_message},
            ],
            "stream":  False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        }

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data    = resp.json()
        content = data.get("message", {}).get("content", "")
        if not content:
            # Fallback: try generate endpoint
            content = self._generate_fallback(user_message)
        return content

    def _generate_fallback(self, prompt: str) -> str:
        """Fallback to /api/generate if /api/chat returns empty."""
        full_prompt = f"System: {self.system}\n\nUser: {prompt}\n\nAssistant:"
        payload = {
            "model":  self.model_id,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        }
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload, timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def update_system(self, new_prompt: str) -> None:
        """Hot-swap the system prompt (used by KD-SPAR)."""
        self.system = new_prompt

    # ── Compatibility shim — mimics AnthropicClient._client.messages.create ──
    # This lets KD-SPAR's self-interview code call client._client.messages.create(...)
    @property
    def _client(self):
        return _OllamaMsgCompat(self)


class _OllamaMsgCompat:
    """
    Compatibility shim that exposes a .messages.create(...) interface
    matching the Anthropic SDK, so KD-SPAR's self-interview code works
    unchanged with OllamaClient.
    """

    def __init__(self, ollama_client: OllamaClient):
        self._oc = ollama_client
        self.messages = self

    def create(
        self,
        model: str,
        max_tokens: int = 200,
        messages: Optional[list] = None,
        system: Optional[str] = None,
        **kwargs,
    ):
        """Mimics anthropic.Anthropic().messages.create()."""
        messages = messages or []
        # Build prompt from messages list
        user_content = " ".join(
            m.get("content", "") for m in messages if m.get("role") == "user"
        )
        sys_content = system or self._oc.system

        payload = {
            "model": self._oc.model_id,
            "messages": [
                {"role": "system", "content": sys_content},
                {"role": "user",   "content": user_content},
            ],
            "stream":  False,
            "options": {
                "num_predict": max_tokens,
                "temperature": self._oc.temperature,
            },
        }
        resp = requests.post(
            f"{self._oc.base_url}/api/chat",
            json=payload, timeout=self._oc.timeout,
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        return _OllamaResponse(content)


class _OllamaResponse:
    """Mimics the Anthropic SDK response object (content[0].text)."""
    def __init__(self, text: str):
        self.content = [type("Block", (), {"text": text})()]
