# Changelog

## v1.7.0 (April 2025)

### Addressing Limitations — Six Targeted Improvements

1. **BERTScore throughout** — ALL ablation conditions now use `kd_score_v2`
   (0.3×citation + 0.5×BERTScore + 0.2×Jaccard) by default. Pass `--jaccard`
   to revert. This is the single highest-impact change — fairer comparison
   across conditions since Jaccard was penalising good paraphrases.

2. **Human evaluation toolkit** — `experiments/human_eval.py` generates blind
   evaluation CSV sheets with answer keys and rater instructions. Computes
   Cohen's κ (2 raters) or Fleiss' κ (3+ raters) for inter-rater agreement.
   Per-condition mean human scores reported alongside automatic metrics.

3. **Code generation domain** — new `CODE_CORPUS` (4 Python docs),
   `CODE_TRAIN_QUERIES` (12), `CODE_VAL_QUERIES` (8). Run with
   `--domain code` for cross-domain evidence. Same 6-condition ablation
   on coding tasks tests generalisation beyond RAG QA.

4. **Larger student configs** — Config 4 (qwen2.5:14b → llama3.1:8b) and
   Config 5 (qwen2.5:14b → qwen2.5:7b) test whether the base A−B gap
   becomes positive when the student has more meta-cognitive capacity (7-8B).

5. **Convergence theorem** — formal Theorem 1 in paper §18d.1 proves
   monotone convergence of the hard-gate SPAR loop. Proof sketch for
   soft-gate convergence via Metropolis-Hastings / simulated annealing argument.

6. **Active learning** — `uncertainty_sample()` and `run_active_learning()`
   methods added to `AdversarialKDSPAR`. Generates multiple responses per query
   at elevated temperature, measures KD score variance, selects the most
   uncertain queries for the next SPAR iteration. Dual-objective validation
   ensures adversarial improvement without standard regression.

### Paper

- §18d.1 "Convergence Analysis" with Theorem 1 (monotone convergence, proof
  sketch, soft-gate discussion)
- §18c.1 "Prompt Templates" — Self-Interview template in structured format

## v1.6.0 (April 2025)

### New Features

- **Tree of Thought proposal generation** (Enhancement 8) — replaces flat single-shot
  proposal generation with a structured search tree:
  - *Branch*: generate K root-cause hypotheses about *why* the failure occurred
  - *Evaluate*: score each hypothesis for explanatory power using the student as evaluator
  - *Expand*: best hypotheses generate targeted instructions from different angles
  - Optional depth > 1 recursion refines the best instruction further
  - Configurable via `use_tree_of_thought`, `tot_branches`, `tot_expansions`, `tot_depth`
    in `EnhancedConfig`

### Paper

- **§3b Literature Survey** — new "Tree of Thought Reasoning" subsection covering
  Yao et al. (2023) and how ToT integrates into KD-SPAR's Phase 2.
- **§13c** renamed to "Eight Algorithmic Improvements" — Enhancement 8 row in table,
  full description of branch→evaluate→expand mechanics, cost analysis (~15 calls/failure).
- **Reference [24]** added: Yao et al. (2023) "Tree of Thoughts: Deliberate Problem
  Solving with Large Language Models." NeurIPS.
- **Glossary** updated with 5 ToT entries (ToT, Branch, Evaluate, Expand).
- All "seven" → "eight" throughout paper, docs, and code.

### Documentation

- Variants guide updated with ToT configuration example.
- All docs reflect 8 enhancements.

## v1.5.0 (April 2025)

### New Features

- **Enhanced KD-SPAR** — new variant (`sara/rag/kd_spar_enhanced.py`) with seven
  algorithmic improvements addressing diagnosed root causes of base KD-SPAR
  underperformance:
  1. **Hybrid proposer** — teacher diagnoses failures, student proposes fixes
  2. **BERTScore metric** — `kd_score_v2()` in `core/utils.py`: 0.3×citation +
     0.5×BERTScore (MiniLM-L6) + 0.2×Jaccard, replacing pure token overlap
  3. **Contrastive interview** — good/bad query pair reasoning for more specific proposals
  4. **Warm-start from B** — one external-proposal iteration bootstraps the prompt
  5. **Increased iterations** — default 5 (was 3)
  6. **Soft commit gate** — probabilistic acceptance via simulated annealing
  7. **Teacher-guided interview** — shows actual teacher response text, not just labels

- **Condition F** added to ablation — runs Enhanced KD-SPAR with all improvements.
  F−A gap isolates enhancement value; F−B gap tests whether enhanced self-authorship
  beats external proposal.

### Paper

- **§13c Enhanced KD-SPAR** — full section covering root cause analysis, 7-enhancement
  table with mechanisms, detailed descriptions, implementation code, and ablation design.
- **§20 methodology** updated for 6 conditions (A-F) with Condition F description.
- **Prior Art table** (§19.2) now includes Enhanced KD-SPAR row.
- **Glossary** updated with Enhanced KD-SPAR terms (hybrid proposer, contrastive
  interview, warm-start, soft commit gate, F−A gap, F−B gap, Condition F).

### Bug Fixes

- **FreeSerif font crash** — `registerFontFamily()` now called after font registration
  so `<font name="FreeSerif">` works in ReportLab Paragraphs. All hardcoded FreeSerif
  references in `sara_story.py` replaced with `_DEVA_FONT` variable for graceful fallback.

### Documentation

- Example `10_enhanced_kd_spar.py` added.
- README, API docs, KD-SPAR variants guide updated for Enhanced KD-SPAR.
- `patch_paper.py` generates F−A and F−B gap commentary with adaptive analysis.

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

*Sara (सार) v1.7.0 · Ashutosh Sinha · ajsinha@gmail.com · AGPL-3.0*
