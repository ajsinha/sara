# Changelog

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

*Sara (सार) v1.1.0 · Ashutosh Sinha · ajsinha@gmail.com · AGPL-3.0*
