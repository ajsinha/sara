# Backend Configuration Guide
**Sara v1.7.0** · Copyright (C) 2025 Ashutosh Sinha · AGPL-3.0

Sara supports any LLM provider through a single configuration point.
All examples and experiments read from `configs/backend.yaml` automatically.

---

## The config file

**`configs/backend.yaml`** controls everything:

```yaml
backend: ollama                        # ollama | anthropic
teacher_model: llama3.1:8b            # any valid model string
student_model: llama3.2:3b
ollama_base_url: http://localhost:11434
```

Change `backend:` and the model names — nothing else needs to change.

---

## Supported backends

### Ollama (FOSS — default)

No API key. Runs 100% locally. Works offline after models are pulled.

```yaml
backend: ollama
teacher_model: llama3.1:8b
student_model: llama3.2:3b
ollama_base_url: http://localhost:11434
```

**Install:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
ollama pull llama3.2:3b
ollama serve &
```

**Tested model pairs:**

| Config | Teacher | Student | Use case |
|--------|---------|---------|----------|
| 1 (recommended) | llama3.1:8b | llama3.2:3b | Same family — controlled test |
| 2 (cross-family) | qwen2.5:7b | llama3.2:3b | Cross-family — stronger evidence |
| 3 (Qwen student) | llama3.1:8b | qwen2.5:3b | Alternative student arch |

Any model on [ollama.com/library](https://ollama.com/library) works — just `ollama pull <name>`.

---

### Anthropic API

Requires `ANTHROPIC_API_KEY`. Highest quality responses.

```yaml
backend: anthropic
teacher_model: claude-3-5-sonnet-20241022
student_model: claude-sonnet-4-5-20250929
```

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Environment variable overrides

These always take priority over `configs/backend.yaml`:

| Variable | Example | Effect |
|----------|---------|--------|
| `SARA_BACKEND` | `ollama` | Switch backend |
| `SARA_TEACHER_MODEL` | `qwen2.5:7b` | Override teacher model |
| `SARA_STUDENT_MODEL` | `llama3.2:3b` | Override student model |
| `OLLAMA_BASE_URL` | `http://10.0.0.5:11434` | Remote Ollama server |
| `ANTHROPIC_API_KEY` | `sk-ant-...` | Anthropic key |

---

## Using the backend factory in code

```python
from sara.rag.backend import get_pipeline, get_client, get_spar, cfg, describe

# Print current configuration
print(describe())
# Backend   : ollama
# Teacher   : llama3.1:8b
# Student   : llama3.2:3b

# Get a pipeline (reads backend from config automatically)
store    = RAGVectorStore()
teacher  = get_pipeline("teacher", store=store)
student  = get_pipeline("student", store=store)

# Get a SPAR instance
spar = get_spar(store=store)
final_prompt, history = spar.run(train_q, val_q, teacher_responses)

# Override model per-call (ignores config for that call only)
pipe = get_pipeline("teacher", model_id="qwen2.5:7b", store=store)
```

---

## Adding a new provider

Extend `sara/rag/backend.py` — add a new `elif backend == "yourprovider":` branch
to `get_pipeline()`, `get_client()`, and `get_spar()`. No other files need changing.

---

*Sara v1.7.0 · Ashutosh Sinha · ajsinha@gmail.com · AGPL-3.0*
