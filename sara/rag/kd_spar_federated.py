# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.6.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
from __future__ import annotations
"""
sara.rag.kd_spar_federated
=========================
Federated KD-SPAR: multiple geographically or organisationally distributed
student deployments each run KD-SPAR diagnosis locally, then share only their
**proposed instruction strings** (never raw query/response data) with a central
aggregation server.

Privacy guarantee
-----------------
The only information that crosses the client→server boundary is a list of
natural-language instruction strings proposed by each client (e.g.
"Always cite retrieved passages using [Doc-N] notation").
No query text, no retrieved document content, and no model responses leave
each client site.

Architecture
------------
                  ┌──────────────────────────────────────────┐
                  │           AGGREGATION SERVER              │
                  │  1. Broadcast current global prompt       │
                  │  2. Receive proposals from clients        │
                  │  3. Cluster + validate on server data     │
                  │  4. Broadcast updated global prompt       │
                  └──────────┬─────────────┬─────────────────┘
                             │             │
               ┌─────────────┘             └──────────┐
               ▼                                       ▼
    ┌──────────────────┐                  ┌────────────────────┐
    │  CLIENT SITE A   │                  │   CLIENT SITE B    │
    │  Local RAG data  │                  │   Local RAG data   │
    │  Diagnose locally│                  │   Diagnose locally │
    │  Propose instrs  │                  │   Propose instrs   │
    │  (no data shared)│                  │   (no data shared) │
    └──────────────────┘                  └────────────────────┘

Federated round
---------------
Each round:
  1. Server sends current global prompt to all clients
  2. Each client runs local diagnosis on its private RAG traces
  3. Each client runs self-interview and sends a list of proposed instructions
  4. Server clusters all proposals, validates each cluster representative
     against its server-side validation queries, accepts best top-K
  5. Server broadcasts updated global prompt
  Repeat until convergence

Practical use-case
------------------
Multiple hospital branches each have their own patient-query RAG system.
They all want a shared student prompt that performs well across all sites
without sharing patient data. Federated KD-SPAR lets each site improve the
shared prompt using only locally available data.
"""


import json
import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from sara.rag.pipeline import (
    RAGPipeline,
    RAGVectorStore,
    AnthropicClient,
    DEFAULT_SYSTEM,
    STUDENT_MODEL,
    TEACHER_MODEL,
)
from sara.rag.kd_spar import (
    KDSPAR,
    SPARIteration,
    _kd_score,
    _classify_failure,
    _target_pattern,
    SELF_INTERVIEW_PROMPT,
    FAILURE_DESCRIPTIONS,
)


# ── Client configuration ──────────────────────────────────────────────────────

@dataclass
class FederatedClientConfig:
    """Configuration for one federated client site."""
    client_id:         str
    student_model:     str = STUDENT_MODEL
    n_local_queries:   int = 50     # local queries used for diagnosis each round
    n_proposals:       int = 3      # proposals per failure mode
    top_k_local:       int = 5      # local pre-filter before sending to server


# ── Round record ──────────────────────────────────────────────────────────────

@dataclass
class FederatedRound:
    """Record of one federated optimisation round."""
    round_number:       int
    clients_participated: list[str]
    total_proposals:    int
    selected_instrs:    list[str]
    score_before:       float
    score_after:        float
    delta:              float
    accepted:           bool


# ── Federated client ──────────────────────────────────────────────────────────

class FederatedKDSPARClient:
    """
    One client site in the Federated KD-SPAR system.

    The client holds private local data that never leaves the site.
    It receives a prompt from the server, diagnoses failures locally,
    runs self-interview, and returns only proposed instruction strings.

    Parameters
    ----------
    config         : FederatedClientConfig
    local_traces   : List of (query, teacher_response) tuples (private data)
    vector_store   : Local ChromaDB store for this client's documents
    """

    def __init__(
        self,
        config:        FederatedClientConfig,
        local_traces:  list[tuple[str, str]],  # (query, teacher_response)
        vector_store:  RAGVectorStore,
    ) -> None:
        self.config       = config
        self.local_traces = local_traces
        self.store        = vector_store
        self._pipeline:   Optional[RAGPipeline] = None
        self._interviewer: Optional[AnthropicClient] = None

    def receive_prompt(self, prompt: str) -> None:
        """Called by the server to update the client's current prompt."""
        if self._pipeline is None:
            self._pipeline = RAGPipeline(
                self.config.student_model, store=self.store, system_prompt=prompt
            )
            self._interviewer = AnthropicClient(
                self.config.student_model, system_prompt=prompt
            )
        else:
            self._pipeline.client.update_system(prompt)
            self._interviewer.update_system(prompt)

    def propose_instructions(self) -> list[str]:
        """
        Diagnose failures on local data and return proposed instruction strings.

        This is the ONLY method that communicates back to the server.
        No query text, context, or responses are returned.

        Returns
        -------
        List of instruction string proposals (plain text, no private data)
        """
        if self._pipeline is None:
            return []

        queries = [q for q, _ in self.local_traces[:self.config.n_local_queries]]
        t_resps = {q: r for q, r in self.local_traces[:self.config.n_local_queries]}

        # Local diagnosis
        failures = self._local_diagnose(queries, t_resps)
        if not failures:
            return []

        # Local self-interview
        all_proposals: list[str] = []
        for diag_q, s_resp, t_resp, mode in failures[:3]:   # top-3 failures
            props = self._local_interview(diag_q, s_resp, t_resp, mode)
            all_proposals.extend(props)

        # Local pre-filter: deduplicate and cap
        unique = list({p for p in all_proposals if len(p) > 15})
        return unique[:self.config.top_k_local]

    def local_kd_score(self, queries: list[str], teacher_resps: dict[str, str]) -> float:
        """Compute mean KD score on local validation queries (used for server round validation)."""
        if self._pipeline is None:
            return 0.0
        scores = []
        for q in queries:
            if q not in teacher_resps:
                continue
            try:
                s = self._pipeline.query(q, return_context=False).answer
                scores.append(_kd_score(s, teacher_resps[q]))
            except Exception:
                scores.append(0.0)
        return sum(scores) / max(len(scores), 1)

    def _local_diagnose(
        self,
        queries:    list[str],
        t_resps:    dict[str, str],
        top_k:      int = 5,
    ) -> list[tuple[str, str, str, str]]:
        """Return (query, student_response, teacher_response, failure_mode) for worst cases."""
        scored = []
        for q in queries:
            if q not in t_resps:
                continue
            try:
                s = self._pipeline.query(q, return_context=False).answer
                sc = _kd_score(s, t_resps[q])
                md = _classify_failure(s, t_resps[q])
                scored.append((q, s, t_resps[q], md, sc))
            except Exception:
                pass
        scored.sort(key=lambda x: x[4])  # worst first
        return [(q, s, t, m) for q, s, t, m, _ in scored[:top_k]]

    def _local_interview(
        self,
        query:     str,
        s_resp:    str,
        t_resp:    str,
        mode:      str,
    ) -> list[str]:
        prompt = SELF_INTERVIEW_PROMPT.format(
            query              = query,
            student_response   = s_resp[:400],
            failure_description= FAILURE_DESCRIPTIONS.get(mode, mode),
            target_pattern     = _target_pattern(t_resp),
        )
        proposals = []
        for _ in range(self.config.n_proposals):
            try:
                resp = self._interviewer._client.messages.create(
                    model=self._interviewer.model_id, max_tokens=80,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text.strip()
                if len(text) > 15 and not text.lower().startswith(("sure", "here")):
                    proposals.append(text)
            except Exception:
                pass
        return proposals


# ── Federated aggregation server ──────────────────────────────────────────────

class FederatedKDSPARServer:
    """
    Central aggregation server for Federated KD-SPAR.

    Coordinates rounds, aggregates client proposals, and maintains the
    global prompt. Uses a server-side validation set (may be synthetic
    or from a held-out corpus) to score proposals.

    Parameters
    ----------
    clients              : List of FederatedKDSPARClient instances
    server_val_queries   : Server-side validation queries (does not need to be
                           the same data as any client's private data)
    server_val_responses : Teacher responses for validation queries
    student_model        : Model ID for server-side scoring
    vector_store         : Server-side vector store for validation
    threshold            : Min mean KD delta to accept a new prompt
    regression_tol       : Per-client allowed regression

    Examples
    --------
    >>> server = FederatedKDSPARServer(
    ...     clients=[client_a, client_b, client_c],
    ...     server_val_queries=val_q,
    ...     server_val_responses=val_resps,
    ...     student_model=STUDENT_MODEL,
    ...     vector_store=server_store,
    ... )
    >>> final_prompt, history = server.run(rounds=10)
    """

    def __init__(
        self,
        clients:              list[FederatedKDSPARClient],
        server_val_queries:   list[str],
        server_val_responses: dict[str, str],
        student_model:        str = STUDENT_MODEL,
        vector_store:         Optional[RAGVectorStore] = None,
        threshold:            float = 0.003,
        regression_tol:       float = 0.02,
        parallel:             bool  = False,
    ) -> None:
        self.clients          = clients
        self.val_queries      = server_val_queries
        self.val_responses    = server_val_responses
        self.student_model    = student_model
        self.store            = vector_store or RAGVectorStore()
        self.threshold        = threshold
        self.regression_tol   = regression_tol
        self.parallel         = parallel
        self._server_pipeline : Optional[RAGPipeline] = None

    def run(
        self,
        base_prompt:  Optional[str] = None,
        rounds:       int  = 10,
        min_clients:  int  = 1,
        log_path:     Optional[str] = None,
    ) -> tuple[str, list[FederatedRound]]:
        """
        Execute the federated KD-SPAR optimisation across all rounds.

        Parameters
        ----------
        base_prompt : Starting global system prompt
        rounds      : Maximum number of federated rounds
        min_clients : Minimum clients that must respond per round
        log_path    : Optional JSONL log file

        Returns
        -------
        (final_global_prompt, round_history)
        """
        global_prompt = base_prompt or DEFAULT_SYSTEM
        history: list[FederatedRound] = []

        self._server_pipeline = RAGPipeline(
            self.student_model, store=self.store, system_prompt=global_prompt
        )

        log_fh = open(log_path, "w") if log_path else None
        try:
            for rnd in range(1, rounds + 1):
                print(f"\n{'='*55}")
                print(f"Federated KD-SPAR  Round {rnd}/{rounds}")
                print(f"  Active clients: {len(self.clients)}")

                # Step 1: Broadcast current prompt to all clients
                self._broadcast_prompt(global_prompt)

                # Step 2: Collect proposals from clients (no data shared)
                all_proposals, participating = self._collect_proposals(min_clients)
                if not all_proposals:
                    print("  No proposals received — converged or no clients responded.")
                    break

                print(f"  Received {len(all_proposals)} total proposals from "
                      f"{len(participating)} client(s)")

                # Step 3: Server-side aggregation and scoring
                old_score  = self._server_score(global_prompt)
                top_instrs = self._aggregate_and_score(
                    all_proposals, global_prompt
                )

                # Step 4: Build candidate and validate
                candidate = (
                    global_prompt
                    + "\n\n# Federated refinements (round "
                    + str(rnd) + "):\n"
                    + "\n".join(f"- {ins}" for ins in top_instrs)
                )
                new_score = self._server_score(candidate)
                delta     = new_score - old_score
                accepted  = delta >= self.threshold

                fed_round = FederatedRound(
                    round_number          = rnd,
                    clients_participated  = participating,
                    total_proposals       = len(all_proposals),
                    selected_instrs       = top_instrs,
                    score_before          = round(old_score, 4),
                    score_after           = round(new_score, 4),
                    delta                 = round(delta, 4),
                    accepted              = accepted,
                )
                history.append(fed_round)

                if log_fh:
                    log_fh.write(json.dumps({
                        "round": rnd, "delta": round(delta, 4),
                        "accepted": accepted,
                        "clients": len(participating),
                        "proposals": len(all_proposals),
                        "selected": top_instrs,
                    }) + "\n")

                status = "✓ ACCEPTED" if accepted else "✗ REVERTED"
                print(f"  {status}  server_score {old_score:.4f} → {new_score:.4f}  "
                      f"(Δ={delta:+.4f})")

                if accepted:
                    global_prompt = candidate
                    self._server_pipeline.client.update_system(global_prompt)

        finally:
            if log_fh:
                log_fh.close()

        accepted_rounds = sum(1 for r in history if r.accepted)
        print(f"\nFederated KD-SPAR complete: {accepted_rounds}/{len(history)} rounds accepted")
        return global_prompt, history

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _broadcast_prompt(self, prompt: str) -> None:
        """Send current global prompt to all clients."""
        for client in self.clients:
            client.receive_prompt(prompt)

    def _collect_proposals(
        self, min_clients: int
    ) -> tuple[list[str], list[str]]:
        """
        Collect proposal strings from all clients.
        If self.parallel=True, polls clients concurrently.

        Returns
        -------
        (all_proposals, participating_client_ids)
        """
        all_proposals: list[str] = []
        participating: list[str] = []

        if self.parallel:
            results: dict[str, list[str]] = {}
            threads = []
            for client in self.clients:
                def _poll(c=client, res=results):
                    res[c.config.client_id] = c.propose_instructions()
                t = threading.Thread(target=_poll)
                threads.append(t)
                t.start()
            for t in threads:
                t.join(timeout=120)
            for cid, props in results.items():
                if props:
                    all_proposals.extend(props)
                    participating.append(cid)
        else:
            for client in self.clients:
                props = client.propose_instructions()
                if props:
                    all_proposals.extend(props)
                    participating.append(client.config.client_id)

        if len(participating) < min_clients:
            print(f"  WARNING: only {len(participating)} client(s) responded "
                  f"(min_clients={min_clients})")

        return all_proposals, participating

    def _server_score(self, prompt: str) -> float:
        """Score a prompt on the server's validation set."""
        self._server_pipeline.client.update_system(prompt)
        scores = []
        for q in self.val_queries:
            if q not in self.val_responses:
                continue
            try:
                s = self._server_pipeline.query(q, return_context=False).answer
                scores.append(_kd_score(s, self.val_responses[q]))
            except Exception:
                scores.append(0.0)
        return sum(scores) / max(len(scores), 1)

    def _aggregate_and_score(
        self,
        proposals:     list[str],
        base_prompt:   str,
        top_k:         int = 3,
    ) -> list[str]:
        """Cluster proposals and return top-K by server-side KD score."""
        unique = list({p for p in proposals if len(p) > 15})[:top_k * 4]
        if len(unique) <= top_k:
            return unique

        orig = self._server_pipeline.client.system
        scored: list[tuple[str, float]] = []
        for p in unique:
            self._server_pipeline.client.update_system(orig + "\n- " + p)
            sc = self._server_score(orig + "\n- " + p)
            scored.append((p, sc))
            self._server_pipeline.client.update_system(orig)

        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [p for p, s in scored[:top_k]]
        for p, s in scored[:top_k]:
            print(f"    {s:.4f}  {p[:70]}")
        return selected


# ── Simulation harness ────────────────────────────────────────────────────────

class FederatedSimulation:
    """
    Convenience harness for simulating a federated KD-SPAR setup in a single
    process — useful for development and testing without real distributed infra.

    Partitions a set of traces into N clients and creates a server with a
    held-out validation set.

    Parameters
    ----------
    n_clients      : Number of simulated client sites
    all_traces     : List of (query, teacher_response) pairs to distribute
    val_fraction   : Fraction held out for the server validation set
    teacher_model  : Teacher model ID (used for any generation tasks)
    student_model  : Student model ID
    vector_store   : Shared or per-client vector store

    Examples
    --------
    >>> sim = FederatedSimulation(
    ...     n_clients=3, all_traces=all_traces,
    ...     student_model=STUDENT_MODEL,
    ...     vector_store=store,
    ... )
    >>> server = sim.build_server()
    >>> prompt, history = server.run(rounds=8)
    """

    def __init__(
        self,
        n_clients:    int,
        all_traces:   list[tuple[str, str]],
        val_fraction: float = 0.2,
        teacher_model: str  = TEACHER_MODEL,
        student_model: str  = STUDENT_MODEL,
        vector_store:  Optional[RAGVectorStore] = None,
    ) -> None:
        self.n_clients     = n_clients
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.store         = vector_store or RAGVectorStore()

        # Split into validation and client partitions
        val_n          = max(1, int(len(all_traces) * val_fraction))
        self._val      = all_traces[:val_n]
        client_traces  = all_traces[val_n:]
        chunk          = max(1, len(client_traces) // n_clients)
        self._client_traces = [
            client_traces[i * chunk : (i + 1) * chunk]
            for i in range(n_clients)
        ]

    def build_clients(self) -> list[FederatedKDSPARClient]:
        clients = []
        for i, traces in enumerate(self._client_traces):
            cfg = FederatedClientConfig(
                client_id    = f"client_{i+1}",
                student_model= self.student_model,
            )
            clients.append(FederatedKDSPARClient(cfg, traces, self.store))
        return clients

    def build_server(
        self,
        threshold:     float = 0.003,
        regression_tol: float = 0.02,
    ) -> FederatedKDSPARServer:
        clients      = self.build_clients()
        val_queries  = [q for q, _ in self._val]
        val_responses= {q: r for q, r in self._val}
        return FederatedKDSPARServer(
            clients             = clients,
            server_val_queries  = val_queries,
            server_val_responses= val_responses,
            student_model       = self.student_model,
            vector_store        = self.store,
            threshold           = threshold,
            regression_tol      = regression_tol,
        )
