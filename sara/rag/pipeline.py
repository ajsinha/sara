# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.1.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
from __future__ import annotations
from sara.core.utils import DEFAULT_SYSTEM_PROMPT  # noqa
"""
sara.rag.pipeline
===============
Production-ready RAG pipeline built on ChromaDB and the Anthropic API.

Key classes
-----------
RAGVectorStore   Persistent ChromaDB collection with sentence-transformer embeddings
AnthropicClient  Thin Anthropic Messages API wrapper with swappable system prompt
RAGPipeline      End-to-end ingest → retrieve → generate pipeline
Document         Typed document chunk dataclass
RAGResponse      Typed response dataclass
"""


import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Optional

# ── Models ────────────────────────────────────────────────────────────────────
TEACHER_MODEL    = "claude-3-5-sonnet-20241022"
STUDENT_MODEL    = "claude-sonnet-4-5-20250929"
EMBEDDING_MODEL  = "sentence-transformers/all-mpnet-base-v2"
CHROMA_PATH      = os.environ.get("CHROMA_PERSIST_PATH", "./chroma_db")
COLLECTION_NAME  = "kd_knowledge_base"
TOP_K            = 5
MAX_TOKENS       = 1024

DEFAULT_SYSTEM = (
    "You are a precise knowledge assistant. "
    "Answer questions using ONLY the provided context passages. "
    "Cite sources inline as [Doc-N] where N is the passage number. "
    "If the context does not contain the answer, say: "
    "'I cannot find this in the provided context.' "
    "Express uncertainty explicitly when evidence is partial."
)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class Document:
    """A single text chunk ready for indexing."""
    content:     str
    source:      str
    chunk_index: int  = 0
    metadata:    dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        """Stable MD5-based identifier derived from content."""
        return hashlib.md5(self.content.encode()).hexdigest()[:16]


@dataclass
class RAGResponse:
    """Full response from a RAGPipeline.query() call."""
    query:          str
    answer:         str
    model_used:     str
    retrieved_docs: list[Document] = field(default_factory=list)
    citations:      list[str]      = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query":      self.query,
            "answer":     self.answer,
            "model_used": self.model_used,
            "citations":  self.citations,
        }


# ── Text chunker ───────────────────────────────────────────────────────────────

def chunk_text(
    text:       str,
    source:     str,
    chunk_size: int = 400,
    overlap:    int = 80,
) -> list[Document]:
    """
    Split text into overlapping word-based chunks.

    Parameters
    ----------
    text       : Full document text
    source     : Source label (filename, URL, …)
    chunk_size : Target chunk size in characters (~80 words)
    overlap    : Character overlap between adjacent chunks (~16 words)

    Returns
    -------
    List of Document objects
    """
    words            = text.split()
    wpc              = max(1, chunk_size  // 5)   # words per chunk
    wpo              = max(0, overlap     // 5)   # words per overlap
    chunks: list[Document] = []
    start  = 0
    idx    = 0

    while start < len(words):
        end   = min(start + wpc, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(Document(content=chunk, source=source, chunk_index=idx))
        start += wpc - wpo
        idx   += 1

    return chunks


# ── Vector store ──────────────────────────────────────────────────────────────

class RAGVectorStore:
    """
    Persistent ChromaDB vector store backed by sentence-transformer embeddings.

    Parameters
    ----------
    persist_path    : Local directory for ChromaDB data files
    collection_name : Name of the ChromaDB collection
    embedding_model : sentence-transformers model ID

    Examples
    --------
    >>> store = RAGVectorStore()
    >>> store.add_documents(chunks)
    >>> results = store.search("What is knowledge distillation?", top_k=5)
    """

    def __init__(
        self,
        persist_path:    str = CHROMA_PATH,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        import chromadb  # type: ignore
        from chromadb.utils import embedding_functions  # type: ignore

        self._client = chromadb.PersistentClient(path=persist_path)
        embed_fn     = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, docs: list[Document], batch_size: int = 64) -> int:
        """
        Upsert documents into the collection.

        Parameters
        ----------
        docs       : List of Document objects
        batch_size : ChromaDB upsert batch size

        Returns
        -------
        Number of documents upserted
        """
        added = 0
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            self._col.upsert(
                ids       = [d.doc_id for d in batch],
                documents = [d.content for d in batch],
                metadatas = [{"source": d.source, "chunk": d.chunk_index,
                              **d.metadata} for d in batch],
            )
            added += len(batch)
        return added

    def search(
        self,
        query:      str,
        top_k:      int = TOP_K,
        where:      Optional[dict] = None,
    ) -> list[Document]:
        """
        Semantic search over the collection.

        Parameters
        ----------
        query : Natural-language query
        top_k : Number of results to return
        where : Optional ChromaDB metadata filter dict

        Returns
        -------
        List of Document objects sorted by relevance (best first)
        """
        kwargs: dict = {"query_texts": [query], "n_results": min(top_k, self.count or 1)}
        if where:
            kwargs["where"] = where

        results = self._col.query(**kwargs)
        return [
            Document(content=c, source=m.get("source", ""), chunk_index=m.get("chunk", 0))
            for c, m in zip(results["documents"][0], results["metadatas"][0])
        ]

    @property
    def count(self) -> int:
        """Number of documents currently in the collection."""
        return self._col.count()

    def clear(self) -> None:
        """Delete all documents from the collection (non-destructive to schema)."""
        ids = self._col.get()["ids"]
        if ids:
            self._col.delete(ids=ids)


# ── Anthropic client ──────────────────────────────────────────────────────────

class AnthropicClient:
    """
    Thin wrapper around the Anthropic Messages API.

    Parameters
    ----------
    model_id     : Anthropic model string
    max_tokens   : Maximum output tokens
    system_prompt: Base system instructions (swappable via :meth:`update_system`)

    Examples
    --------
    >>> client = AnthropicClient(model_id=TEACHER_MODEL)
    >>> text = client.generate("What is KD?", context_docs=[...])
    """

    def __init__(
        self,
        model_id:      str = TEACHER_MODEL,
        max_tokens:    int = MAX_TOKENS,
        system_prompt: Optional[str] = None,
    ) -> None:
        import anthropic as _anthropic  # type: ignore

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Export it: export ANTHROPIC_API_KEY='sk-ant-...'"
            )
        self._client    = _anthropic.Anthropic(api_key=api_key)
        self.model_id   = model_id
        self.max_tokens = max_tokens
        self.system     = system_prompt or DEFAULT_SYSTEM

    def generate(self, query: str, context_docs: list[Document]) -> str:
        """
        Generate a grounded answer from the query and retrieved passages.

        Parameters
        ----------
        query        : User question
        context_docs : Retrieved Document objects (will be numbered [Doc-1], …)

        Returns
        -------
        Model response string
        """
        context_block = "\n\n".join(
            f"[Doc-{i+1}] (source: {d.source})\n{d.content}"
            for i, d in enumerate(context_docs)
        )
        message = f"Context passages:\n{context_block}\n\nQuestion: {query}"

        response = self._client.messages.create(
            model      = self.model_id,
            max_tokens = self.max_tokens,
            system     = self.system,
            messages   = [{"role": "user", "content": message}],
        )
        return response.content[0].text

    def update_system(self, new_prompt: str) -> None:
        """Hot-swap the system prompt (used by KD-SPAR and prompt optimisers)."""
        self.system = new_prompt


# ── Full RAG pipeline ─────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG: ingest → retrieve → generate.

    Parameters
    ----------
    model_id      : Anthropic model for generation
    store         : Optional pre-built RAGVectorStore (creates a new one if None)
    top_k         : Number of passages to retrieve per query
    system_prompt : Override the default system prompt

    Examples
    --------
    >>> pipeline = RAGPipeline(model_id=TEACHER_MODEL)
    >>> pipeline.ingest({"intro.txt": "Knowledge distillation is ..."})
    >>> response = pipeline.query("What is knowledge distillation?")
    >>> print(response.answer)
    """

    def __init__(
        self,
        model_id:      str = TEACHER_MODEL,
        store:         Optional[RAGVectorStore] = None,
        top_k:         int = TOP_K,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.store    = store or RAGVectorStore()
        self.client   = AnthropicClient(model_id=model_id, system_prompt=system_prompt)
        self.top_k    = top_k
        self.model_id = model_id

    def ingest(
        self,
        texts:      dict[str, str],
        chunk_size: int = 400,
        overlap:    int = 80,
    ) -> int:
        """
        Chunk and index a dict of ``{source_name: full_text}``.

        Returns
        -------
        Total number of chunks indexed
        """
        all_chunks: list[Document] = []
        for source, text in texts.items():
            all_chunks.extend(chunk_text(text, source, chunk_size, overlap))
        return self.store.add_documents(all_chunks)

    def query(
        self,
        question:       str,
        where:          Optional[dict] = None,
        return_context: bool = True,
    ) -> RAGResponse:
        """
        Retrieve relevant passages and generate a grounded answer.

        Parameters
        ----------
        question       : User query
        where          : Optional metadata filter for retrieval
        return_context : If False, retrieved_docs is empty in the response

        Returns
        -------
        RAGResponse
        """
        retrieved = self.store.search(question, self.top_k, where)
        if not retrieved:
            return RAGResponse(
                query="",
                answer="No relevant passages found in the knowledge base.",
                model_used=self.model_id,
            )

        answer    = self.client.generate(question, retrieved)
        citations = sorted(set(re.findall(r"\[Doc-\d+\]", answer)))

        return RAGResponse(
            query          = question,
            answer         = answer,
            model_used     = self.model_id,
            retrieved_docs = retrieved if return_context else [],
            citations      = citations,
        )
