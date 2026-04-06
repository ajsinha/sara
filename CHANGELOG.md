# Changelog

## v1.4.0 (April 2025)

### New Features

- **MetaKDSPAR** — new metaprompting-enhanced KD-SPAR variant (`sara/rag/kd_spar_meta.py`).
  Uses a conductor + 4 specialist architecture (citation, calibration, completeness, format)
  for multi-perspective diagnosis. Each specialist independently analyses student failures,
  a conductor synthesises the top diagnoses, and each specialist proposes domain-specific
  fixes. Extends the ablation to 5 conditions (A-E) for direct comparison.

- **Literature Survey** — new §3b in the paper: comprehensive survey of knowledge
  distillation, prompt optimisation (OPRO, APE, DSPy, EvoPrompt), self-refinement
  (Self-Refine, Constitutional AI), federated learning (FedAvg), and metaprompting
  (Suzgun & Kalai, 2024). §18b Related Work now cross-references §3b.

- **MetaKDSPAR paper section** — new §13b covering architecture, algorithm differences
  from base KD-SPAR, testable hypothesis (E−A gap), implementation, and trade-offs.

### Experiment Changes

- **Condition E** added to ablation — `build_E()` runs MetaKDSPAR alongside the
  existing 4 conditions. `collect_results.py`, `results_analysis.py`, and
  `patch_paper.py` all handle 5 conditions (A-E).

- **E−A gap commentary** — `patch_paper.py` generates adaptive expert commentary
  on whether multi-perspective diagnosis outperforms flat diagnosis, with honest
  analysis for positive, neutral, and negative outcomes.

### Documentation

- Example `09_meta_kd_spar.py` added.
- README, API docs, KD-SPAR variants guide updated for MetaKDSPAR.
- Glossary updated with MetaKDSPAR terms (conductor, specialist, E−A gap, Condition E).
- References updated with Suzgun & Kalai (2024) [23].
- §19.2 Prior Art table includes MetaKDSPAR row.
- §18.3 Extensions includes MetaKDSPAR-specific future directions.

## v1.3.0 (April 2025)

### Paper — Publication-Quality Section 20

- **Synthetic results removed** — Section 20 in `sara_story.py` is now a clear
  placeholder that says "This section is automatically replaced with real results."
  No fabricated numbers remain in the codebase.

- **`patch_paper.py` completely rewritten** — generates a comprehensive, publication-ready
  Section 20 from `aggregated_results.json` with six subsections:
  - **§20.1 Experimental Methodology** — hardware, models, iterations, evaluation
    protocol, query counts, temperature setting, and four-condition design rationale.
  - **§20.2 Main Results** — measured KD scores, deltas, and citation fidelity with
    mean ± std across seeds.
  - **§20.3 A−B Gap Analysis** — per-configuration teacher→student gap table with
    hypothesis support status.
  - **§20.4 Statistical Analysis** — computed t-statistic, p-value, Cohen's d effect
    size, 95% CI, and formal H₀ rejection decision. All values are calculated from
    actual data, not hardcoded.
  - **§20.5 Discussion** — adaptive expert commentary that changes based on whether
    the A−B gap is strong/moderate/marginal/negative. Includes baseline ladder analysis
    and honest limitations (Jaccard vs BERTScore, human evaluation needs).
  - **§20.6 Reproducing These Results** — exact shell commands.

- **Honest commentary** — if the A−B gap is negative, the paper says so and offers
  concrete explanations (model capacity, iteration count, metric choice). No spin.

## v1.2.0 (April 2025)

### Improvements

- **Paper patching is now automatic** — `setup_and_run.sh` installs `reportlab` and `pypdf`
  during Step 3 (project setup) and always runs `patch_paper.py` in Step 7. The PDF at
  `docs/paper/Sara_Knowledge_Distillation.pdf` is rebuilt with real experimental numbers
  without any manual intervention.

### Bug Fixes

- **`patch_paper.py` path error** — all path constants pointed at the old root `paper/`
  directory (removed in v1.1.0). Now correctly uses `docs/paper/` for `PAPER_DIR`,
  `STORY_SRC`, `HELPERS_SRC`, `FINAL_SCRIPT`, and `DEFAULT_PDF`.

- **`results_analysis.py` KeyError: 'teacher'** — the Ollama ablation script saved
  `"config"` but not `"teacher"` / `"student"` as top-level JSON keys. `results_analysis.py`
  now uses `.get()` with graceful fallback to `"config"`. Also handles both `"build_time_s"`
  and `"build_time_sec"` key names.

- **`kd_spar_ablation_ollama.py` key inconsistencies** — JSON output now writes
  `"build_time_sec"` (was `"build_time_s"`, mismatching everything else) and includes
  `"teacher"` and `"student"` as top-level keys for downstream tooling.

- **`collect_results.py` deprecation warning** — `datetime.utcnow()` replaced with
  `datetime.now(timezone.utc)`. Teacher/student extraction also fixed to parse the
  `"config"` label (e.g. `"llama8b→llama3b"`) when dedicated keys are absent.

## v1.1.0 (April 2025)

### Bug Fixes

- **License inconsistency resolved** — `pyproject.toml` declared MIT while `LICENSE`,
  all SPDX headers (54 files), and `README.md` declared AGPL-3.0-or-later.
  `pyproject.toml` now correctly uses `AGPL-3.0-or-later` with the proper OSI classifier.

- **Stale `kd.` namespace purged** — the package was renamed from `kd` to `sara` in v1.0.0
  but 35+ docstrings and comments still referenced the old `kd.core`, `kd.rag`, `kd.vision`
  paths. All now read `sara.core`, `sara.rag`, `sara.vision`, etc.

- **Scoring function deduplication** — `_jaccard()`, `_kd_score()`, `CITATION_RE`, and
  `HEDGE_WORDS` were defined independently in both `sara/core/utils.py` (canonical) and
  `sara/rag/kd_spar.py` (local copies with slightly different behaviour). `kd_spar.py` now
  imports from `sara.core.utils` and re-exports under private aliases for backward
  compatibility with variant modules. The `HEDGE_WORDS` divergence (single words in
  `kd_spar.py` vs multi-word phrases in `core/utils.py`) is resolved — `_classify_failure()`
  now uses substring matching consistent with the canonical tuple.

- **`_classify_failure()` hedge-counting asymmetry** — teacher hedging used `in` (substring
  match) while student hedging used `split()` (word-only match). Both now use the same
  substring-match logic against the canonical `HEDGE_WORDS` tuple.

### New Exports

- **`sara.core.progress`** — `SaraLogger`, `Heartbeat`, `ProgressBar`, and `phase()` are
  now exported through `sara.core.__init__` and `sara.__init__`, making them available as
  `from sara import SaraLogger` or `from sara.core import phase`.

### Paper

- **Section 19 restructured** — renamed from "Is KD-SPAR Publishable or Patentable?" to
  "Novelties of KD-SPAR". Sections 19.3 (Publishability Assessment) and 19.4 (Patent
  Analysis) removed. Former §19.5 (Summary Verdict) renumbered to §19.3 (Summary
  Assessment) with table rows refocused on novelty dimensions rather than venue/patent
  analysis. Author's callout updated accordingly.

- **Part VIII renamed** — from "Publication & Patent Analysis" to "Novelty & Research
  Analysis" throughout (banner, document structure table, abstract, docs).

- **PDF rebuilt** — 43 pages (down from 44). All page-count references updated.

### Documentation

- **Root `paper/` directory removed** — the duplicate `paper/sara_helpers.py` and
  `paper/sara_story.py` at project root have been removed. The canonical location is
  `docs/paper/` which also contains the PDF.

- **Root `QUICKSTART.md` replaced** — the outdated root quickstart (referencing the old
  `knowledge_distillation` directory name, `from kd.core` imports, and "70+ tests") is
  replaced with a concise redirect to `docs/QUICKSTART.md`.

- **`README.md` updated** — module table now includes `sara.rag.backend` and progress
  logging; install section corrected (base `pip install -e .` installs no heavy deps;
  `[vision]` extra noted separately); project structure tree updated.

- **`docs/api/README.md` updated** — module map now includes `progress.py` with usage
  examples for `SaraLogger` and `phase()`.

### Internal

- Version bumped to 1.1.0 across all 96 files: SPDX headers, `pyproject.toml`,
  `__init__.py`, `LICENSE`, `README.md`, all docs, all examples, all experiments,
  all test files, and shell scripts.

- Stale `sara.egg-info/` removed from distribution.

---

*Sara (सार) v1.4.0 · Ashutosh Sinha · ajsinha@gmail.com · AGPL-3.0*
