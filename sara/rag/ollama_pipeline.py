from __future__ import annotations
"""
kd.rag.ollama_pipeline
======================
Ollama-backed RAGPipeline — identical interface to :class:`kd.rag.pipeline.RAGPipeline`
but uses a local Ollama model for generation instead of the Anthropic API.

Drop-in replacement: just swap the import and the model ID strings.

    # Anthropic version
    from sara.rag.pipeline import RAGPipeline
    pipe = RAGPipeline(model_id="claude-3-5-sonnet-20241022", store=store)

    # Ollama version (zero cost, no rate limits)
    from sara.rag.ollama_pipeline import OllamaRAGPipeline
    pipe = OllamaRAGPipeline(model_id="llama3.1:8b", store=store)

Both expose the same .query(), .ingest(), and .client.update_system() interface
so all KD-SPAR code works unchanged.
"""


import re
from typing import Optional

from sara.rag.pipeline import (
    RAGVectorStore,
    RAGResponse,
    Document,
    chunk_text,
    TOP_K,
)
from sara.rag.ollama_client import (
    OllamaClient,
    OLLAMA_TEACHER_MODEL,
    OLLAMA_STUDENT_MODEL,
    OLLAMA_DEFAULT_URL,
    OLLAMA_DEFAULT_SYSTEM,
    ensure_model,
)


class OllamaRAGPipeline:
    """
    End-to-end RAG pipeline using a local Ollama model for generation
    and ChromaDB + sentence-transformers for retrieval.

    Parameters
    ----------
    model_id      : Ollama model string (e.g. ``"llama3.1:8b"``)
    store         : Existing :class:`RAGVectorStore` (creates new one if None)
    top_k         : Number of passages to retrieve per query
    system_prompt : Override the default system prompt
    base_url      : Ollama server URL
    auto_pull     : If True, pull the model automatically if not found locally
    temperature   : Generation temperature (0 = fully deterministic)

    Examples
    --------
    >>> store    = RAGVectorStore()
    >>> pipeline = OllamaRAGPipeline("llama3.1:8b", store=store)
    >>> pipeline.ingest({"doc.txt": "Knowledge distillation ..."})
    >>> resp = pipeline.query("What is KD?")
    >>> print(resp.answer)
    """

    def __init__(
        self,
        model_id:      str = OLLAMA_TEACHER_MODEL,
        store:         Optional[RAGVectorStore] = None,
        top_k:         int = TOP_K,
        system_prompt: Optional[str] = None,
        base_url:      str = OLLAMA_DEFAULT_URL,
        auto_pull:     bool = True,
        temperature:   float = 0.1,
    ) -> None:
        self.store    = store or RAGVectorStore()
        self.top_k    = top_k
        self.model_id = model_id

        if auto_pull:
            ensure_model(model_id, base_url)

        self.client = OllamaClient(
            model_id      = model_id,
            system_prompt = system_prompt or OLLAMA_DEFAULT_SYSTEM,
            base_url      = base_url,
            temperature   = temperature,
        )

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
        :class:`kd.rag.pipeline.RAGResponse`
        """
        retrieved = self.store.search(question, self.top_k, where)
        if not retrieved:
            return RAGResponse(
                query      = question,
                answer     = "No relevant passages found in the knowledge base.",
                model_used = self.model_id,
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
