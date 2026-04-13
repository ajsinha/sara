# Evaluation Corpus

**Location:** `experiments/docs/`

These documents are the primary evidence base for the KD-SPAR experiments. They should be
reviewed, version-controlled, and cited in any publication. To modify the experiment's
knowledge base, edit or add files here — no code changes needed.

## Design Rationale

Three design decisions are critical to experimental validity:

1. **Document size: 3–5 KB each (~500–1,000 words).** Production RAG corpora contain
   substantial documents. Documents smaller than ~500 bytes make retrieval trivial (the
   model sees 20-30% of the entire corpus per query) and eliminate the selection pressure
   that makes RAG challenging. Early experiments with ~100-word documents produced a ceiling
   effect where all advanced conditions converged to identical scores.

2. **Total corpus size: 56 KB (RAG), 48 KB (Code).** At 512-token chunks with 10% overlap,
   the RAG corpus produces ~100–120 chunks. With top-5 retrieval, the model sees ~4–5% of
   the corpus per query — matching the selectivity of production RAG systems.

3. **Topical diversity with overlap.** Documents cover related but distinct topics (e.g.,
   'knowledge distillation' and 'model compression' overlap but are not identical). This
   creates realistic cross-document reasoning opportunities.

## RAG Domain (`rag/`) — 12 documents, 56 KB

| Document | Topic | Size |
|----------|-------|------|
| kd_foundations.txt | KD theory, loss formula, dark knowledge, variants | 5.0 KB |
| rag_systems.txt | RAG pipeline, citation, faithfulness, failure modes | 4.3 KB |
| kd_spar_method.txt | KD-SPAR algorithm, four phases, self-knowledge | 4.5 KB |
| local_models.txt | Llama 3.1/3.2, Qwen 2.5, Gemma 2, Phi-3, Ollama | 4.3 KB |
| transformer_architecture.txt | Self-attention, RoPE, GQA, SwiGLU, FFN | 4.3 KB |
| prompt_engineering.txt | System prompts, CoT, ReAct, structured output | 4.7 KB |
| evaluation_metrics.txt | BLEU, ROUGE, BERTScore, faithfulness, human eval | 5.0 KB |
| fine_tuning_methods.txt | LoRA, QLoRA, prefix tuning, RLHF, DPO | 4.2 KB |
| vector_databases.txt | ChromaDB, FAISS, Pinecone, HNSW, chunking | 5.0 KB |
| model_compression.txt | Quantisation, pruning, lottery ticket hypothesis | 4.8 KB |
| llm_safety.txt | Alignment, Constitutional AI, red teaming, guardrails | 4.5 KB |
| distributed_training.txt | Data/tensor/pipeline parallelism, DeepSpeed, FSDP | 5.3 KB |

## Code Domain (`code/`) — 8 documents, 48 KB

| Document | Topic | Size |
|----------|-------|------|
| functions_and_closures.txt | First-class functions, closures, decorators, lambdas | 5.8 KB |
| data_structures.txt | Lists, dicts, sets, collections, dataclasses | 4.5 KB |
| algorithms.txt | Sorting, searching, two pointers, sliding window, DP, BFS | 5.6 KB |
| testing_and_pytest.txt | Fixtures, parametrize, mocking, TDD, coverage | 5.3 KB |
| error_handling.txt | Exceptions, context managers, defensive programming | 6.0 KB |
| concurrency.txt | GIL, threading, asyncio, multiprocessing, rate limiting | 6.7 KB |
| file_io.txt | Path operations, JSON, CSV, atomic writes, archives | 6.5 KB |
| oop_patterns.txt | ABC, inheritance, composition, strategy pattern | 7.7 KB |

## Usage

```bash
# The ablation script loads corpus automatically:
python experiments/kd_spar_ablation_ollama.py --domain rag   # loads docs/rag/*.txt
python experiments/kd_spar_ablation_ollama.py --domain code  # loads docs/code/*.txt

# Or load programmatically:
from pathlib import Path
corpus = {p.name: p.read_text() for p in sorted(Path("experiments/docs/rag").glob("*.txt"))}
```

## Adding New Documents

Drop a `.txt` file into the appropriate directory. The loader picks up all `.txt` files
automatically. Rerun the experiment to include the new document.

*Sara v1.8.3 · Ashutosh Sinha · ajsinha@gmail.com · AGPL-3.0*
