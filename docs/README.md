# Sara (सार) Documentation
**Version 1.4.0** · Copyright (C) 2025 Ashutosh Sinha · AGPL-3.0

---

## Documents in this folder

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | Installation, first run, Ollama setup — start here |
| [LEGAL.md](LEGAL.md) | Copyright, AGPL license, third-party notices |
| [paper/Sara_Knowledge_Distillation.pdf](paper/Sara_Knowledge_Distillation.pdf) | Full 43-page research paper |
| [guides/backend_configuration.md](guides/backend_configuration.md) | Switching between Ollama, Anthropic, and future providers |
| [guides/kd_spar_variants.md](guides/kd_spar_variants.md) | All four KD-SPAR variants with code examples |
| [guides/experiments.md](guides/experiments.md) | Running the ablation, interpreting A−B gap, patching the paper |
| [api/README.md](api/README.md) | API reference and common patterns |

---

## Paper structure (43 pages)

| Part | Sections | Content |
|------|----------|---------|
| I — Foundations | 1–3 | Theory, soft targets, taxonomy |
| II — Implementation | 4–5 | Code, advanced techniques |
| III — Applications | 6–7 | Benchmarks, real-world cases |
| IV — RAG & Migration | 8–9 | Pipeline, prompt KD |
| V — KD-SPAR Variants | 10–13 | Base, Multi-Teacher, Adversarial, Federated |
| VI — Practitioner Guide | 14–17 | Hyperparams, eval, frameworks |
| VII — Frontier | 18–18e | Future directions, Related Work, Algorithm 1, Limitations, Conclusion |
| VIII — Novelty & Research | 19–21 | Novelty assessment, ablation design, local model setup |

---

## Rebuild the paper

```bash
cd docs/paper
cat sara_helpers.py sara_story.py > _build_final.py
python _build_final.py
# → Sara_Knowledge_Distillation.pdf
```

Or after running experiments, patch with real numbers:
```bash
python experiments/collect_results.py
python experiments/patch_paper.py --output docs/paper/Sara_Knowledge_Distillation.pdf
```

---

*Sara (सार) — Sanskrit for "quintessence": the refined essence extracted from a larger whole.*  
*v1.4.0 · Ashutosh Sinha · ajsinha@gmail.com · AGPL-3.0*
