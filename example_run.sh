#!/usr/bin/env bash
# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.2.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
#
# example_run.sh — Quick reference for common ways to run Sara experiments
#
# ╔══════════════════════════════════════════════════════════════╗
# ║  REFERENCE FILE — do not run this script top-to-bottom.     ║
# ║  Copy and paste the single command that fits your situation. ║
# ╚══════════════════════════════════════════════════════════════╝
#
# Running ./example_run.sh prints this help and exits safely.

cat << 'HELP'

Sara (सार) — Common experiment commands
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

First, activate your virtual environment:
  cd ~/PycharmProjects/sara
  source .venv/bin/activate

─── FULL PUBLICATION RUN (default — 5 seeds × 2 configs, ~3 hours) ──────────
  bash setup_and_run.sh

─── QUICK FIRST RUN (~20 min, 3 seeds, Config 1 only) ───────────────────────
  bash setup_and_run.sh --quick

─── SKIP SETUP (Ollama + venv already installed) ────────────────────────────
  bash setup_and_run.sh --skip-setup

─── CUSTOM CONFIG ────────────────────────────────────────────────────────────
  # Config 2 only (qwen teacher), 3 iterations, specific seeds
  bash setup_and_run.sh --config 2 --iterations 3 --seeds "42 123 456"

  # Single seed quick check
  bash setup_and_run.sh --config 1 --iterations 3 --seeds "42"

─── INDIVIDUAL STEPS (manual, once setup is done) ───────────────────────────
  # Run one ablation manually
  python experiments/kd_spar_ablation_ollama.py --config 1 --iterations 3 --seed 42

  # Aggregate results after running
  python experiments/collect_results.py

  # Patch paper with real numbers
  python experiments/patch_paper.py --output docs/paper/Sara_Knowledge_Distillation.pdf

  # View results summary
  python experiments/results_analysis.py

─── SANITY CHECK (verify Ollama is working, ~2 min) ─────────────────────────
  python examples/08_ollama_kd_spar.py --sanity-only

─── HELP ─────────────────────────────────────────────────────────────────────
  bash setup_and_run.sh --help

HELP
