# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.8.3
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ajsinha/sara
"""
experiments/kd_spar_ablation_ollama.py
========================================
Controlled KD-SPAR ablation using LOCAL Ollama models.

  ✓  Zero API cost
  ✓  No rate limits  →  full experiment in ~20 minutes
  ✓  Fully reproducible (fixed seed + temperature=0)
  ✓  Works completely offline after models are pulled

FOUR CONDITIONS (identical to kd_spar_ablation.py, different backend)
----------------------------------------------------------------------
A  KD-SPAR (student self-proposed)       — our method
B  Externally proposed (teacher proposes) — same KD signal, different proposer
C  Random instructions                    — no KD signal
D  No prompt tuning                       — vanilla baseline

MODEL CONFIGURATIONS
--------------------
Config 1: Teacher=llama3.1:8b  →  Student=llama3.2:3b   (same family)
Config 2: Teacher=qwen2.5:7b   →  Student=llama3.2:3b   (cross-family)
Config 3: Teacher=llama3.1:8b  →  Student=qwen2.5:3b    (cross-family, Qwen student)

Choose with --config 1|2|3  or  --teacher <model> --student <model>

SETUP ON ORYXPRO (Pop!_OS)
---------------------------
  # 1. Install Ollama
  curl -fsSL https://ollama.com/install.sh | sh

  # 2. Pull the models  (one-time download)
  ollama pull llama3.1:8b     # ~4.7 GB
  ollama pull llama3.2:3b     # ~2.0 GB
  ollama pull qwen2.5:7b      # ~4.4 GB  (for config 2)

  # 3. Start Ollama  (or it auto-starts as a systemd service)
  ollama serve &

  # 4. Run the ablation
  cd knowledge_distillation
  source .venv/bin/activate
  pip install -e ".[rag]"  requests

  # Config 1 (llama8b → llama3b) — recommended first run
  python experiments/kd_spar_ablation_ollama.py --config 1 --iterations 3

  # Config 2 (qwen7b → llama3b) — cross-family comparison
  python experiments/kd_spar_ablation_ollama.py --config 2 --iterations 3

  # Both configs, 3 seeds each — publication quality
  for cfg in 1 2; do
    for seed in 42 123 777; do
      python experiments/kd_spar_ablation_ollama.py --config $cfg --iterations 3 --seed $seed
    done
  done

EXPECTED RUNTIME
----------------
  llama3.1:8b teacher + llama3.2:3b student, 3 iterations, 40 train / 15 val:
    OryxPro (RTX 3070 Ti):  ~12–18 minutes
    OryxPro (CPU only):     ~45–90 minutes

GPU IS OPTIONAL. Set OLLAMA_NUM_GPU=0 to force CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Ensure project root on path ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sara.rag.pipeline import RAGVectorStore
from sara.rag.ollama_client import (
    OllamaClient,
    OLLAMA_DEFAULT_SYSTEM,
    check_ollama_running,
    ensure_model,
    list_available_models,
)
from sara.rag.ollama_pipeline import OllamaRAGPipeline
from sara.rag.ollama_kd_spar import OllamaKDSPAR
from sara.rag.kd_spar import _kd_score as _kd_score_jaccard, _classify_failure, _target_pattern, FAILURE_DESCRIPTIONS
from sara.core.utils import kd_score_v2 as _kd_score_v2

# Default: use BERTScore-based scoring for ALL conditions (fairer comparison)
# Pass --jaccard to revert to token-overlap scoring
_SCORING = {"use_bert": True}

def _kd_score(student: str, teacher: str) -> float:
    """Score function used by all conditions — BERTScore by default."""
    if _SCORING["use_bert"]:
        return _kd_score_v2(student, teacher, use_bert=True)
    return _kd_score_jaccard(student, teacher)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Corpus loader ──────────────────────────────────────────────────────────
CORPUS_DIR = Path(__file__).parent / "docs"

def load_corpus(domain: str = "rag") -> dict[str, str]:
    """Load corpus documents from experiments/corpus/<domain>/*.txt.

    Falls back to inline CORPUS/CODE_CORPUS dicts if files not found.
    """
    corpus_path = CORPUS_DIR / domain
    if corpus_path.exists():
        loaded = {}
        for p in sorted(corpus_path.glob("*.txt")):
            text = p.read_text().strip()
            if text:
                loaded[p.name] = text
        if loaded:
            print(f"  Corpus loaded from {corpus_path}: {len(loaded)} documents "
                  f"({sum(len(t) for t in loaded.values())} chars)")
            return loaded
    # Fallback to inline
    print(f"  Corpus: using inline {domain.upper()} documents (files not found at {corpus_path})")
    return _CORPUS_RAG if domain == "rag" else _CORPUS_CODE

# ── Model configs ──────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    # Primary experiments — constant teacher (llama3.1:8b), 7 student architectures
    1:  {"teacher": "llama3.1:8b",  "student": "llama3.2:3b",    "label": "llama8b→llama3b"},
    2:  {"teacher": "llama3.1:8b",  "student": "qwen2.5:3b",     "label": "llama8b→qwen3b"},
    3:  {"teacher": "llama3.1:8b",  "student": "gemma2:2b",      "label": "llama8b→gemma2b"},
    4:  {"teacher": "llama3.1:8b",  "student": "phi3:3.8b",      "label": "llama8b→phi3.8b"},
    5:  {"teacher": "llama3.1:8b",  "student": "smollm2:1.7b",   "label": "llama8b→smollm1.7b"},
    6:  {"teacher": "llama3.1:8b",  "student": "stablelm2:1.6b", "label": "llama8b→stablelm1.6b"},
    7:  {"teacher": "llama3.1:8b",  "student": "tinyllama:1.1b", "label": "llama8b→tinyllama1.1b"},
    # Cross-family teacher tests
    8:  {"teacher": "qwen2.5:7b",   "student": "llama3.2:3b",    "label": "qwen7b→llama3b"},
    9:  {"teacher": "qwen2.5:7b",   "student": "qwen2.5:3b",     "label": "qwen7b→qwen3b"},
    # Larger student (capacity ceiling test) — needs more VRAM
    10: {"teacher": "qwen2.5:14b",  "student": "llama3.1:8b",    "label": "qwen14b→llama8b"},
    11: {"teacher": "qwen2.5:14b",  "student": "qwen2.5:7b",     "label": "qwen14b→qwen7b"},
}

# ── Random instruction pool ────────────────────────────────────────────────
RANDOM_INSTRUCTION_POOL = [
    "Always use bullet points when listing multiple items.",
    "Begin every response with a one-sentence summary.",
    "Keep responses under 120 words whenever possible.",
    "Use the active voice in all sentences.",
    "End every response with a clarifying question.",
    "Include at least one concrete example in each response.",
    "Use the Oxford comma when listing three or more items.",
    "Include a brief disclaimer about response limitations.",
    "Use subheadings to organise responses longer than 100 words.",
    "Translate any jargon into plain language immediately after using it.",
    "Always acknowledge if there are multiple valid perspectives.",
    "Start responses with 'To answer your question directly:'",
    "Prefer shorter sentences with fewer than 20 words each.",
    "Use the phrase 'In summary' before concluding paragraphs.",
    "Include a confidence rating (high/medium/low) at the end.",
    "Reference the most relevant source document first.",
    "Avoid passive constructions wherever possible.",
    "State what you cannot confirm before what you can.",
    "Use numbered lists for procedural or sequential content.",
    "Always spell out acronyms on first use.",
]

EXTERNAL_PROPOSE_PROMPT = """\
You are a prompt engineering expert optimising a knowledge assistant's system prompt.

The assistant is failing with these issues:
{failure_modes}

The target response pattern shows:
{target_pattern}

Write exactly ONE instruction to add to the assistant's system prompt.
Return ONLY the instruction text. No preamble, no quotes, no explanation.
"""

# ── Corpus — loaded from experiments/docs/rag/*.txt ─────────────────────
# The file-based loader (load_corpus) is the primary path.
# These inline dicts are minimal fallbacks for environments without the files.
_CORPUS_RAG = {
    "kd_foundations.txt": "Knowledge distillation transfers knowledge from a large teacher model to a smaller student model using temperature-scaled softmax outputs as soft targets. The distillation loss combines KL divergence on softened distributions with cross-entropy on hard labels. The T-squared factor corrects gradient magnitudes.",
    "rag_systems.txt": "Retrieval-Augmented Generation grounds LLM responses in external knowledge through three phases: document ingestion, retrieval via semantic search, and generation with citations. Citation format uses [Doc-N] notation. Faithfulness measures whether claims are grounded in retrieved passages.",
    "kd_spar_method.txt": "KD-SPAR lets the student model diagnose its own failure modes and propose targeted amendments to its own system prompt. The four phases are diagnostic pass, self-interview, aggregation, and validate-and-commit. The self-knowledge hypothesis claims the student has privileged knowledge about what instructions will improve its own performance.",
    "local_models.txt": "Llama 3.1 8B offers strong instruction following at 4.7 GB in Q4 quantisation. Llama 3.2 3B targets edge deployment at 2 GB. Qwen 2.5 provides excellent multilingual capability. Running locally via Ollama requires no API key and provides full reproducibility.",
}
# ── Code domain corpus — loaded from experiments/docs/code/*.txt ─────────
_CORPUS_CODE = {
    "functions_and_closures.txt": "Python functions are first-class objects. Closures capture variables from enclosing scope. Decorators wrap functions to add behavior. Lambda functions enable inline anonymous functions. List comprehensions are preferred over map/filter for readability.",
    "data_structures.txt": "Python built-in types: list (ordered mutable), tuple (immutable), dict (O(1) lookup), set (unique elements). Collections module adds defaultdict, Counter, deque, namedtuple. Dataclasses reduce boilerplate for data containers.",
    "algorithms.txt": "Common patterns: two pointers, sliding window, binary search, BFS/DFS, dynamic programming, backtracking. Python sorted() uses Timsort O(n log n). bisect provides binary search on sorted data. heapq provides min-heap operations.",
    "testing_and_pytest.txt": "pytest test functions start with test_. Fixtures provide reusable setup. Parametrize runs tests with multiple inputs. Mock external dependencies with unittest.mock.patch. Coverage measured with pytest-cov. TDD: write failing test first.",
}

CODE_TRAIN_QUERIES = [
    "Write a Python function that computes the Fibonacci sequence using dynamic programming.",
    "Explain how Python's list comprehensions work and give three examples.",
    "Write a function to find all duplicates in a list using a set.",
    "Implement a binary search function that returns the index of a target in a sorted list.",
    "Explain the difference between @staticmethod and @classmethod in Python.",
    "Write a decorator that measures the execution time of a function.",
    "Implement a simple LRU cache using OrderedDict.",
    "Write a pytest test suite for a function that validates email addresses.",
    "Explain Python's GIL and its impact on multithreading.",
    "Write a generator function that yields prime numbers indefinitely.",
    "Implement a context manager using both the class-based and decorator approaches.",
    "Explain how Python's garbage collection works with reference counting and cycle detection.",
]

CODE_VAL_QUERIES = [
    "Write a function that merges two sorted lists into one sorted list.",
    "Explain the difference between deepcopy and shallow copy with examples.",
    "Write a class that implements an iterator for a binary tree (in-order traversal).",
    "Explain how to use functools.lru_cache and when it helps.",
    "Write a function that groups anagrams from a list of strings.",
    "Implement a simple publish-subscribe event system in Python.",
    "Explain Python's descriptor protocol (__get__, __set__, __delete__).",
    "Write a function that validates a balanced parentheses string.",
]

TRAIN_QUERIES = [
    "What is the role of temperature in knowledge distillation?",
    "How does the T-squared scaling factor work?",
    "What is dark knowledge and why does it matter?",
    "Explain the combined distillation loss formula.",
    "What is the difference between soft targets and hard labels?",
    "How does FitNets feature-based distillation work?",
    "What does the alpha parameter control in KD?",
    "How does attention transfer distillation work?",
    "How does ChromaDB support a RAG pipeline?",
    "What are the three phases of a RAG pipeline?",
    "Why should RAG responses use [Doc-N] citations?",
    "What failure modes does KD-SPAR target?",
    "What is the self-interview phase of KD-SPAR?",
    "What is the self-knowledge hypothesis in KD-SPAR?",
    "How does Llama 3.1 8B differ from Llama 3.2 3B?",
    "What is Qwen 2.5 and what are its strengths?",
    "What is relational knowledge distillation?",
    "How does progressive distillation work?",
    "What is the diagnostic pass in KD-SPAR?",
    "How does aggregation work in the KD-SPAR loop?",
    "When should you use a high temperature in KD?",
    "What is the capacity ratio in knowledge distillation?",
    "How does self-distillation work with early-exit networks?",
    "What is the validate-and-commit phase of KD-SPAR?",
    "How does Ollama enable local model deployment?",
    "What are the main failure modes KD-SPAR diagnoses?",
    "What makes KD-SPAR different from DSPy or OPRO?",
    "How does online mutual learning work?",
    "What is the hallucination proxy metric in RAG?",
    "What is citation fidelity and how is it measured?",
]

VAL_QUERIES = [
    "What is the primary contribution of KD-SPAR?",
    "Why does the student self-author its own prompt?",
    "What is the T-squared factor and why is it necessary?",
    "What does the missing_citation failure mode indicate?",
    "How does the calibration ratio equivalence check work?",
    "What is the difference between gap-mined and generated adversarial queries?",
    "How does the validate-and-commit phase prevent overfitting?",
    "What is dark knowledge in KD?",
    "How does KD-SPAR's self-knowledge differ from Constitutional AI?",
    "What makes Llama 3.2 3B suitable as a student model?",
    "How does the Transformer self-attention mechanism work?",
    "What is the difference between LoRA and full fine-tuning?",
    "How does BERTScore differ from BLEU as an evaluation metric?",
    "What is the purpose of chunking in a vector database?",
    "How does Constitutional AI use self-critique for alignment?",
    "What is quantisation-aware training and why is it useful?",
    "How does DeepSpeed ZeRO reduce memory requirements?",
    "What are the advantages of few-shot prompting over zero-shot?",
    "How does FAISS differ from ChromaDB for vector search?",
    "What is the lottery ticket hypothesis in model compression?",
    "How does the temperature parameter affect generation diversity?",
    "What is the difference between data parallelism and tensor parallelism?",
    "How does RLHF align models with human preferences?",
    "What is faithfulness grounding in RAG evaluation?",
    "How does prefix tuning differ from prompt tuning?",
]


# ── Metrics ────────────────────────────────────────────────────────────────
CITATION_RE = re.compile(r"\[Doc-\d+\]")
HEDGE_WORDS = ["may", "might", "could", "possibly", "perhaps", "appears", "seems",
               "uncertain", "unclear", "cannot confirm"]

def citation_fidelity(s: str, t: str) -> float:
    return 1.0 if (not CITATION_RE.search(t)) or CITATION_RE.search(s) else 0.0

def hedge_match(s: str, t: str) -> float:
    sh = sum(1 for w in HEDGE_WORDS if w in s.lower())
    th = sum(1 for w in HEDGE_WORDS if w in t.lower())
    return sh / max(th, 0.01)

def evaluate_prompt(
    prompt: str,
    queries: list[str],
    teacher_responses: dict[str, str],
    store: RAGVectorStore,
    student_model: str,
    label: str = "",
) -> dict:
    pipeline = OllamaRAGPipeline(
        student_model, store=store, system_prompt=prompt,
        base_url=OLLAMA_BASE_URL, auto_pull=False, temperature=0.0,
    )
    scores, cits, hedges = [], [], []
    per_query = []
    for q in queries:
        if q not in teacher_responses: continue
        try:
            s_resp = pipeline.query(q, return_context=False).answer
            t_resp = teacher_responses[q]
            kd  = _kd_score(s_resp, t_resp)
            cit = citation_fidelity(s_resp, t_resp)
            hed = hedge_match(s_resp, t_resp)
            scores.append(kd); cits.append(cit); hedges.append(hed)
            per_query.append({"q": q[:60], "kd": round(kd,4), "cit": round(cit,4)})
        except Exception as exc:
            print(f"    [eval] error: {exc}")

    result = {
        "label":             label,
        "n_queries":         len(scores),
        "mean_kd_score":     round(sum(scores) / max(len(scores), 1), 4),
        "citation_fidelity": round(sum(cits)   / max(len(cits),   1), 4),
        "hedge_match":       round(sum(hedges)  / max(len(hedges),  1), 4),
        "per_query":         per_query,
    }
    print(f"  [{label}]  kd={result['mean_kd_score']:.4f}  "
          f"cit={result['citation_fidelity']:.3f}  hedge={result['hedge_match']:.3f}")
    return result


def batch_kd(queries, teacher_resps, pipeline) -> float:
    scores = []
    for q in queries:
        if q not in teacher_resps: continue
        try:
            s = pipeline.query(q, return_context=False).answer
            scores.append(_kd_score(s, teacher_resps[q]))
        except Exception: scores.append(0.0)
    return sum(scores) / max(len(scores), 1)


# ── Condition builders ─────────────────────────────────────────────────────

def build_D(teacher_model: str, student_model: str) -> str:
    return OLLAMA_DEFAULT_SYSTEM

def build_C(teacher_model: str, student_model: str, n: int = 4) -> str:
    sampled = random.sample(RANDOM_INSTRUCTION_POOL, min(n, len(RANDOM_INSTRUCTION_POOL)))
    return OLLAMA_DEFAULT_SYSTEM + "\n\n# Random instructions:\n" + \
           "\n".join(f"- {s}" for s in sampled)

def build_B(
    teacher_model: str,
    student_model: str,
    train_queries: list[str],
    teacher_responses: dict[str, str],
    store: RAGVectorStore,
    iterations: int = 3,
) -> str:
    """Condition B: TEACHER externally proposes instructions."""
    from sara.core.progress import SaraLogger
    log = SaraLogger("Cond-B")

    current = OLLAMA_DEFAULT_SYSTEM
    teacher_proposer = OllamaClient(
        teacher_model, system_prompt=current,
        base_url=OLLAMA_BASE_URL, temperature=0.3,
    )
    student_pipe = OllamaRAGPipeline(
        student_model, store=store, system_prompt=current,
        base_url=OLLAMA_BASE_URL, auto_pull=False, temperature=0.0,
    )

    for it in range(iterations):
        log.info(f"  External iteration {it+1}/{iterations} — diagnosing")
        # Diagnose failures
        scored = []
        q_batch = train_queries[:8]
        for i, q in enumerate(q_batch, 1):
            if q not in teacher_responses: continue
            try:
                s = student_pipe.query(q, return_context=False).answer
                sc = _kd_score(s, teacher_responses[q])
                md = _classify_failure(s, teacher_responses[q])
                scored.append((q, s, teacher_responses[q], md, sc))
            except Exception:
                pass
        scored.sort(key=lambda x: x[4])

        if not scored:
            log.warn("  No scored queries — stopping B early")
            break

        # Teacher proposes (NOT student)
        log.info(f"  Generating external proposals from {teacher_model}")
        proposals = []
        for q, s_resp, t_resp, mode, _ in scored[:3]:
            t_pats = []
            if CITATION_RE.search(t_resp): t_pats.append("uses [Doc-N] citations")
            if any(w in t_resp.lower() for w in HEDGE_WORDS): t_pats.append("hedges uncertainty")
            if len(t_resp) > 200: t_pats.append("provides detailed multi-sentence answers")
            prompt = EXTERNAL_PROPOSE_PROMPT.format(
                failure_modes=mode,
                target_pattern="; ".join(t_pats) or "matches reference style",
            )
            try:
                resp = teacher_proposer._client.create(
                    model=teacher_model, max_tokens=80,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text.strip()
                if len(text) > 15 and not text.lower().startswith(("sure", "here", "i'll")):
                    proposals.append(text)
            except Exception as exc:
                log.warn(f"  External proposal error: {exc}")

        if not proposals:
            log.warn("  No proposals generated — stopping B")
            break

        # Same commit gate as KD-SPAR (fair comparison)
        old_sc = batch_kd(VAL_QUERIES[:5], teacher_responses, student_pipe)
        candidate = (current + "\n\n# External proposals:\n"
                     + "\n".join(f"- {p}" for p in proposals[:3]))
        student_pipe.client.update_system(candidate)
        new_sc = batch_kd(VAL_QUERIES[:5], teacher_responses, student_pipe)

        if new_sc > old_sc + 0.003:
            current = candidate
            log.result("B", new_sc, new_sc - old_sc, 0.0, accepted=True)
            print(f"[SARA_ITER] condition=B iter={it+1}/{iterations} ACCEPTED delta={new_sc - old_sc:+.4f}", flush=True)
        else:
            student_pipe.client.update_system(current)
            log.result("B", new_sc, new_sc - old_sc, 0.0, accepted=False)
            print(f"[SARA_ITER] condition=B iter={it+1}/{iterations} REVERTED delta={new_sc - old_sc:+.4f}", flush=True)

    return current

def build_A(
    teacher_model: str,
    student_model: str,
    train_queries: list[str],
    val_queries: list[str],
    teacher_responses: dict[str, str],
    store: RAGVectorStore,
    iterations: int = 3,
) -> str:
    """Condition A: KD-SPAR student self-proposed."""
    spar = OllamaKDSPAR(
        teacher_model=teacher_model, student_model=student_model,
        vector_store=store, base_url=OLLAMA_BASE_URL, auto_pull=False,
        temperature=0.3,  # allow some variation for proposal diversity
    )
    final_prompt, _ = spar.run(
        train_queries=train_queries, val_queries=val_queries,
        teacher_responses=teacher_responses,
        iterations=iterations, threshold=0.003, n_proposals=4, top_k=3,
    )
    return final_prompt


def build_E(
    teacher_model: str,
    student_model: str,
    train_queries: list[str],
    val_queries: list[str],
    teacher_responses: dict[str, str],
    store: RAGVectorStore,
    iterations: int = 3,
) -> str:
    """Condition E: MetaKDSPAR — metaprompting-enhanced self-proposed."""
    from sara.rag.kd_spar_meta import MetaKDSPAR
    meta = MetaKDSPAR(
        student_model=student_model, vector_store=store,
        base_url=OLLAMA_BASE_URL, temperature=0.3,
    )
    final_prompt, _ = meta.run(
        train_queries=train_queries, val_queries=val_queries,
        teacher_responses=teacher_responses,
        iterations=iterations, threshold=0.003,
        n_proposals=3, top_k_diag=3, top_k_instr=3,
    )
    return final_prompt


def build_F(
    teacher_model: str,
    student_model: str,
    train_queries: list[str],
    val_queries: list[str],
    teacher_responses: dict[str, str],
    store: RAGVectorStore,
    iterations: int = 3,
) -> str:
    """Condition F: Enhanced KD-SPAR — all 8 improvements."""
    from sara.rag.kd_spar_enhanced import EnhancedKDSPAR, EnhancedConfig
    cfg = EnhancedConfig(
        use_bert_score=True,
        use_hybrid_proposer=True,
        use_contrastive=True,
        warm_start_from_b=True,
        iterations=max(iterations, 5),
        soft_gate=True,
        teacher_guided=True,
        threshold=0.002,
    )
    enhanced = EnhancedKDSPAR(
        teacher_model=teacher_model, student_model=student_model,
        vector_store=store, config=cfg,
        base_url=OLLAMA_BASE_URL, temperature=0.3,
    )
    final_prompt, _ = enhanced.run(
        train_queries=train_queries, val_queries=val_queries,
        teacher_responses=teacher_responses,
    )
    return final_prompt


# ── Main ablation ──────────────────────────────────────────────────────────

@dataclass
class AblationResult:
    condition: str
    description: str
    final_prompt: str
    val_metrics: dict
    build_time_sec: float


def evaluate_prompt(
    prompt: str,
    queries: list[str],
    teacher_responses: dict[str, str],
    store: RAGVectorStore,
    student_model: str,
    label: str = "",
    log=None,
) -> dict:
    """Evaluate a prompt against val queries, with per-query progress."""
    from sara.core.progress import SaraLogger
    _log = log or SaraLogger("eval")

    pipeline = OllamaRAGPipeline(
        student_model, store=store, system_prompt=prompt,
        base_url=OLLAMA_BASE_URL, auto_pull=False, temperature=0.0,
    )
    eligible = [q for q in queries if q in teacher_responses]
    _log.step(f"Evaluating {len(eligible)} val queries", total=len(eligible))

    scores, cits, hedges, per_query = [], [], [], []
    for i, q in enumerate(eligible, 1):
        try:
            s_resp = pipeline.query(q, return_context=False).answer
            t_resp = teacher_responses[q]
            kd  = _kd_score(s_resp, t_resp)
            cit = 1.0 if (not CITATION_RE.search(t_resp)) or CITATION_RE.search(s_resp) else 0.0
            hed = sum(1 for w in HEDGE_WORDS if w in s_resp.lower()) / max(
                    sum(1 for w in HEDGE_WORDS if w in t_resp.lower()), 0.01)
            scores.append(kd); cits.append(cit); hedges.append(hed)
            per_query.append({"q": q[:60], "kd": round(kd, 4), "cit": round(cit, 4)})
        except Exception as exc:
            _log.warn(f"eval error on q{i}: {exc}")
        _log.tick(i)

    result = {
        "label":             label,
        "n_queries":         len(scores),
        "mean_kd_score":     round(sum(scores) / max(len(scores), 1), 4),
        "citation_fidelity": round(sum(cits)   / max(len(cits),   1), 4),
        "hedge_match":       round(sum(hedges)  / max(len(hedges),  1), 4),
        "per_query":         per_query,
    }
    _log.done(
        f"kd={result['mean_kd_score']:.4f}  "
        f"cit={result['citation_fidelity']:.3f}  "
        f"hedge={result['hedge_match']:.3f}"
    )
    return result


def run_ablation(
    teacher_model: str,
    student_model: str,
    config_label:  str,
    iterations:    int  = 3,
    seed:          int  = 42,
    quick_mode:    bool = False,
    corpus:        dict | None = None,
    train_queries: list[str] | None = None,
    val_queries:   list[str] | None = None,
) -> tuple[list[AblationResult], dict]:
    from sara.core.progress import SaraLogger, Heartbeat, _fmt_elapsed

    _corpus = corpus or _CORPUS_RAG
    _train  = train_queries or TRAIN_QUERIES
    _val    = val_queries or VAL_QUERIES

    log = SaraLogger("Sara Ablation")
    log.banner(
        f"Sara (सार)  KD-SPAR Ablation",
        f"Config   : {config_label}",
        f"Teacher  : {teacher_model}",
        f"Student  : {student_model}",
        f"Seed     : {seed}   Iterations: {iterations}   Quick: {quick_mode}",
    )

    random.seed(seed)
    train_q = _train[:10] if quick_mode else _train
    val_q   = _val[:5]    if quick_mode else _val
    log.info(f"Train queries: {len(train_q)}   Val queries: {len(val_q)}")

    # Start global heartbeat — prints if silent for 30s
    log.start_heartbeat(interval=30, message="Waiting for Ollama (GPU inference in progress)…")

    # ── Ingest corpus ────────────────────────────────────────────────────────
    log.section("Corpus ingestion")
    store_path   = str(RESULTS_DIR / f"ablation_ollama_{config_label}_chroma")
    store        = RAGVectorStore(persist_path=store_path)
    teacher_pipe = OllamaRAGPipeline(
        teacher_model, store=store, base_url=OLLAMA_BASE_URL,
        auto_pull=False, temperature=0.0,
    )
    n = teacher_pipe.ingest(_corpus)
    log.done(f"Indexed {n} chunks from corpus")

    # ── Harvest teacher responses ────────────────────────────────────────────
    log.section(f"Teacher harvest  ({teacher_model}, temperature=0)")
    all_q   = list(dict.fromkeys(train_q + val_q))
    t_resps: dict[str, str] = {}
    log.step(f"Querying teacher for {len(all_q)} responses", total=len(all_q))
    for i, q in enumerate(all_q, 1):
        try:
            t_resps[q] = teacher_pipe.query(q, return_context=False).answer
        except Exception as exc:
            log.warn(f"  q{i} failed: {exc}")
        log.tick(i)
    log.done(f"{len(t_resps)}/{len(all_q)} teacher responses collected")

    # ── Run six conditions ──────────────────────────────────────────────────
    results: list[AblationResult] = []
    cond_defs = [
        ("D", "Baseline (no tuning)",    build_D,
         (teacher_model, student_model)),
        ("C", "Random instructions",     build_C,
         (teacher_model, student_model, min(iterations * 2, 6))),
        ("B", "External-proposed",       build_B,
         (teacher_model, student_model, train_q, t_resps, store, iterations)),
        ("A", "KD-SPAR (self-proposed)", build_A,
         (teacher_model, student_model, train_q, val_q, t_resps, store, iterations)),
        ("E", "MetaKDSPAR (meta-prompted)", build_E,
         (teacher_model, student_model, train_q, val_q, t_resps, store, iterations)),
        ("F", "Enhanced KD-SPAR (all improvements)", build_F,
         (teacher_model, student_model, train_q, val_q, t_resps, store, iterations)),
    ]

    for ci, (cond_label, description, fn, args) in enumerate(cond_defs, 1):
        log.section(f"Condition {cond_label} — {description}")
        t0 = time.time()
        prompt  = fn(*args)
        metrics = evaluate_prompt(
            prompt, val_q, t_resps, store, student_model,
            label=f"{cond_label}_{description[:8]}",
            log=log,
        )
        elapsed = round(time.time() - t0, 1)
        results.append(AblationResult(cond_label, description, prompt, metrics, elapsed))
        log.metric(
            f"Condition {cond_label}",
            f"kd={metrics['mean_kd_score']:.4f}  "
            f"cit={metrics['citation_fidelity']:.3f}  "
            f"hedge={metrics['hedge_match']:.3f}",
            f"{_fmt_elapsed(elapsed)}"
        )
        # Structured progress line for master script parsing
        print(f"[SARA_EVAL] {ci}/6 condition={cond_label} "
              f"kd={metrics['mean_kd_score']:.4f} "
              f"citation={metrics['citation_fidelity']:.3f} "
              f"elapsed={elapsed:.0f}s", flush=True)

    log.stop_heartbeat()
    return results, t_resps


def print_report(results: list[AblationResult], config_label: str) -> str:
    sorted_r  = sorted(results, key=lambda r: r.val_metrics["mean_kd_score"], reverse=True)
    baseline  = next(r for r in results if r.condition == "D")
    d_kd      = baseline.val_metrics["mean_kd_score"]

    lines = []
    lines.append(f"\n{'='*65}")
    lines.append(f"OLLAMA KD-SPAR ABLATION RESULTS  [{config_label}]")
    lines.append(f"{'='*65}")
    lines.append(f"{'Rank':<5}{'Cond':<6}{'KD Score':<12}{'Δ vs D':<10}"
                 f"{'Cit Fid':<10}{'Hedge':<10}{'Time':>8}  Description")
    lines.append("-"*65)

    for rank, r in enumerate(sorted_r, 1):
        delta = r.val_metrics["mean_kd_score"] - d_kd
        lines.append(
            f"  {rank}     {r.condition}     "
            f"{r.val_metrics['mean_kd_score']:.4f}       "
            f"{delta:+.4f}    "
            f"{r.val_metrics['citation_fidelity']:.3f}      "
            f"{r.val_metrics['hedge_match']:.3f}    "
            f"{r.build_time_sec:>6.0f}s  "
            f"{r.description[:30]}"
        )

    a = next(r for r in results if r.condition == "A")
    b = next(r for r in results if r.condition == "B")
    c = next(r for r in results if r.condition == "C")
    e = next((r for r in results if r.condition == "E"), None)
    a_kd = a.val_metrics["mean_kd_score"]
    b_kd = b.val_metrics["mean_kd_score"]
    c_kd = c.val_metrics["mean_kd_score"]
    e_kd = e.val_metrics["mean_kd_score"] if e else 0.0
    gap  = a_kd - b_kd

    lines.append(f"\n{'='*65}")
    supported = a_kd > b_kd and a_kd > c_kd and a_kd > d_kd
    lines.append("KEY FINDING:")
    if supported:
        lines.append("  ✓ SELF-KNOWLEDGE HYPOTHESIS SUPPORTED")
        lines.append(f"  A={a_kd:.4f} > B={b_kd:.4f} > D={d_kd:.4f} > C={c_kd:.4f}")
    else:
        lines.append("  ✗ INCONCLUSIVE — see details below")
        lines.append(f"  A={a_kd:.4f}  B={b_kd:.4f}  D={d_kd:.4f}  C={c_kd:.4f}")

    if e:
        ea_gap = e_kd - a_kd
        lines.append(f"\nMetaKDSPAR (E) = {e_kd:.4f}   E−A gap = {ea_gap:+.4f}")
        if ea_gap > 0.005:
            lines.append("  → Meta-prompting improves over base KD-SPAR")
        elif ea_gap > -0.005:
            lines.append("  → Meta-prompting comparable to base KD-SPAR")
        else:
            lines.append("  → Meta-prompting overhead not justified for this config")

    f_r = next((r for r in results if r.condition == "F"), None)
    if f_r:
        f_kd = f_r.val_metrics["mean_kd_score"]
        fa_gap = f_kd - a_kd
        fb_gap = f_kd - b_kd
        lines.append(f"\nEnhanced (F) = {f_kd:.4f}   F−A gap = {fa_gap:+.4f}   F−B gap = {fb_gap:+.4f}")
        if fb_gap > 0.005:
            lines.append("  → Enhanced KD-SPAR outperforms external proposal")
        elif fa_gap > 0.005:
            lines.append("  → Enhancements improve over base but not external")
        else:
            lines.append("  → Enhancements did not improve in this config")

    lines.append(f"\nA−B gap (self-authorship value) = {gap:+.4f}")
    if   gap > 0.02: lines.append("  → STRONG evidence for self-knowledge claim")
    elif gap > 0.01: lines.append("  → MODERATE — supports claim; more iterations would strengthen it")
    elif gap > 0.005:lines.append("  → WEAK — suggestive; try more iterations or a larger query set")
    elif gap > 0.0:  lines.append("  → MARGINAL — positive but within noise; re-run with more iterations")
    else:            lines.append("  → NEGATIVE — external matching/beating self-proposed")

    lines.append(f"\nINTERPRETATION:")
    lines.append("  A > B  → self-authorship adds value beyond the KD signal alone")
    lines.append("  E > A  → meta-prompting improves over flat self-diagnosis")
    lines.append("  F > A  → enhancements improve over base KD-SPAR")
    lines.append("  F > B  → enhanced self-authorship beats external proposal")
    lines.append("  B > C  → KD-signal-guided proposals beat random augmentation")
    lines.append("  B > D  → any KD-guided proposal beats no tuning")
    lines.append("  C ≈ D  → random instructions provide no real signal (expected)")

    report = "\n".join(lines)
    print(report)
    return report


def save_results(results, config_label, report, seed,
                 teacher_model: str = "", student_model: str = "") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jp = RESULTS_DIR / f"ablation_ollama_{config_label}_seed{seed}_{ts}.json"
    tp = RESULTS_DIR / f"ablation_ollama_{config_label}_seed{seed}_{ts}_summary.txt"

    conds_data = [
        {
            "condition":      r.condition,
            "description":    r.description,
            "val_metrics":    r.val_metrics,
            "build_time_sec": r.build_time_sec,
            "final_prompt":   r.final_prompt,
        }
        for r in results
    ]
    with open(jp, "w") as f:
        json.dump({"timestamp": ts, "config": config_label, "seed": seed,
                   "teacher": teacher_model, "student": student_model,
                   "conditions": conds_data}, f, indent=2)
    with open(tp, "w") as f:
        f.write(report)
    print(f"\nSaved: {jp}")
    return jp


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KD-SPAR Ablation — Local Ollama Models"
    )
    parser.add_argument("--config", type=int, choices=list(range(1,12)), default=1,
                        help="Model config: 1=llama8b→llama3b, 2=qwen7b→llama3b, 3=llama8b→qwen3b")
    parser.add_argument("--teacher", type=str, default=None,
                        help="Override teacher model (e.g. llama3.1:8b)")
    parser.add_argument("--student", type=str, default=None,
                        help="Override student model (e.g. llama3.2:3b)")
    parser.add_argument("--iterations", type=int, default=3,
                        help="SPAR iterations per condition (default 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 10 train / 5 val queries")
    parser.add_argument("--jaccard", action="store_true",
                        help="Use Jaccard scoring instead of BERTScore for all conditions")
    parser.add_argument("--domain", type=str, choices=["rag", "code"], default="rag",
                        help="Evaluation domain: 'rag' (default) or 'code' (cross-domain test)")
    args = parser.parse_args()

    # Set scoring mode
    _SCORING["use_bert"] = not args.jaccard
    print(f"Scoring: {'BERTScore (kd_score_v2)' if _SCORING['use_bert'] else 'Jaccard (kd_score v1)'}")

    # Check Ollama is running
    if not check_ollama_running(OLLAMA_BASE_URL):
        print(f"ERROR: Ollama server not running at {OLLAMA_BASE_URL}")
        print("Start it:   ollama serve")
        print("Install:    curl -fsSL https://ollama.com/install.sh | sh")
        sys.exit(1)

    # Resolve model config
    if args.teacher and args.student:
        teacher_m = args.teacher
        student_m = args.student
        label     = f"custom_{teacher_m.replace(':','_')}_{student_m.replace(':','_')}"
    else:
        cfg       = MODEL_CONFIGS[args.config]
        teacher_m = cfg["teacher"]
        student_m = cfg["student"]
        label     = cfg["label"]

    print(f"Available models: {list_available_models(OLLAMA_BASE_URL)}")

    # Pull if needed
    ensure_model(teacher_m, OLLAMA_BASE_URL)
    ensure_model(student_m, OLLAMA_BASE_URL)

    # Run
    # Select domain
    if args.domain == "code":
        corpus_d = load_corpus("code")
        train_d  = CODE_TRAIN_QUERIES
        val_d    = CODE_VAL_QUERIES
        label    = f"code_{label}"
        print(f"Domain: CODE GENERATION ({len(train_d)} train, {len(val_d)} val)")
    else:
        corpus_d = load_corpus("rag")
        train_d  = TRAIN_QUERIES
        val_d    = VAL_QUERIES
        print(f"Domain: RAG QA ({len(train_d)} train, {len(val_d)} val)")

    t_run_start = time.time()
    results, teacher_responses = run_ablation(
        teacher_model=teacher_m, student_model=student_m,
        config_label=label,
        iterations=args.iterations, seed=args.seed,
        quick_mode=args.quick,
        corpus=corpus_d, train_queries=train_d, val_queries=val_d,
    )
    report   = print_report(results, label)
    out_path = save_results(results, label, report, args.seed,
                            teacher_model=teacher_m, student_model=student_m)

    from sara.core.progress import SaraLogger, _fmt_elapsed
    log = SaraLogger("Sara")
    log.summary(time.time() - t_run_start)
    print(f"Results file: {out_path}")
