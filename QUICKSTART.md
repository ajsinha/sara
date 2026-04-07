# Quickstart Guide

The full quickstart guide is at **[docs/QUICKSTART.md](docs/QUICKSTART.md)**.

For a quick start:

```bash
cd sara
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[rag]"

# Install Ollama (FOSS, default backend)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b && ollama pull llama3.2:3b

# Run KD-SPAR
python examples/04_kd_spar.py
```

See also: [README.md](README.md) · [docs/guides/](docs/guides/)

---

*Sara (सार) v1.7.0 · Ashutosh Sinha · ajsinha@gmail.com · AGPL-3.0*
