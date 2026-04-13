#!/usr/bin/env bash
# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.8.3
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ajsinha/sara
#
# ============================================================
# Sara (सार) — Local Experimentation Setup & Ablation Script
# ============================================================
#
# Usage:
#   bash setup_and_run.sh               # full publication run
#   bash setup_and_run.sh --quick       # 3 seeds × 1 config (~20 min)
#   bash setup_and_run.sh --skip-setup  # skip Ollama + pip (already done)
#   bash setup_and_run.sh --config 2    # only run Config 2 (qwen teacher)
#   bash setup_and_run.sh --iterations 5 --seeds "42 123 456 789 101 200 300"
#
# After completion, results are in experiments/results/
# The paper is patched automatically at docs/paper/Sara_Knowledge_Distillation.pdf
#
# Requirements: Pop!_OS / Ubuntu, Python 3.10+, ~12 GB free disk

set -euo pipefail   # exit on error, unset var, or pipe failure

# ── Defaults (override via flags) ───────────────────────────────────────────
SARA_DIR="${HOME}/PycharmProjects/sara"
QUICK_MODE=false
SKIP_SETUP=false
ITERATIONS=3
CONFIGS="1 2"
SEEDS="42 123 456 789 101"

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)        QUICK_MODE=true;  CONFIGS="1";    SEEDS="42 123 456"; shift ;;
        --skip-setup)   SKIP_SETUP=true;  shift ;;
        --config)       CONFIGS="$2";     shift 2 ;;
        --iterations)   ITERATIONS="$2";  shift 2 ;;
        --seeds)        SEEDS="$2";       shift 2 ;;
        --dir)          SARA_DIR="$2";    shift 2 ;;
        -h|--help)
            echo "Usage: bash setup_and_run.sh [--quick] [--skip-setup]"
            echo "       [--config 1|2|'1 2'] [--iterations N]"
            echo "       [--seeds '42 123 456'] [--dir /path/to/sara]"
            exit 0 ;;
        *)  echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[sara]${NC} $*"; }
warn() { echo -e "${YELLOW}[sara]${NC} $*"; }
fail() { echo -e "${RED}[sara] ERROR:${NC} $*"; exit 1; }

log "Sara (सार) — Local Experimentation Script v1.8.3"
log "Project dir : ${SARA_DIR}"
log "Configs     : ${CONFIGS}"
log "Seeds       : ${SEEDS}"
log "Iterations  : ${ITERATIONS}"
log "Quick mode  : ${QUICK_MODE}"
echo ""

# ── STEP 1: Install Ollama (one-time) ────────────────────────────────────────
if [[ "${SKIP_SETUP}" == "false" ]]; then
    log "STEP 1 — Checking Ollama installation"
    if ! command -v ollama &>/dev/null; then
        log "Ollama not found — installing..."
        curl -fsSL https://ollama.com/install.sh | sh
        log "Ollama installed."
    else
        log "Ollama already installed: $(ollama --version 2>/dev/null || echo 'version unknown')"
    fi

    # Start Ollama server if not already running
    if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
        log "Starting Ollama server..."
        ollama serve &>/tmp/ollama_sara.log &
        OLLAMA_PID=$!
        # Wait up to 15 seconds for server to be ready
        for i in $(seq 1 15); do
            if curl -sf http://localhost:11434/api/tags &>/dev/null; then
                log "Ollama server ready (pid ${OLLAMA_PID})"
                break
            fi
            sleep 1
            if [[ $i -eq 15 ]]; then
                fail "Ollama server did not start. Check: cat /tmp/ollama_sara.log"
            fi
        done
    else
        log "Ollama server already running."
    fi
    echo ""

    # ── STEP 2: Pull models (one-time, uses disk not GPU) ────────────────────
    log "STEP 2 — Pulling Ollama models"
    log "  Disk required: ~11 GB total (only downloads what is missing)"
    echo ""

    pull_if_missing() {
        local model="$1"
        local size="$2"
        if ollama list 2>/dev/null | grep -q "^${model%:*}"; then
            log "  ${model} already present, skipping."
        else
            log "  Pulling ${model}  (${size}) ..."
            ollama pull "${model}"
            log "  ${model} ready."
        fi
    }

    pull_if_missing "llama3.1:8b"  "4.7 GB — teacher for Config 1 & 3"
    pull_if_missing "llama3.2:3b"  "2.0 GB — student for Config 1 & 2"
    pull_if_missing "qwen2.5:7b"   "4.4 GB — teacher for Config 2"
    pull_if_missing "qwen2.5:3b"   "2.1 GB — student for Config 3 (optional)"
    echo ""

    # ── STEP 3: Project setup ────────────────────────────────────────────────
    log "STEP 3 — Project setup"

    [[ -d "${SARA_DIR}" ]] || fail "Sara directory not found: ${SARA_DIR}\nSet --dir /path/to/sara"
    cd "${SARA_DIR}"
    log "  Working directory: $(pwd)"

    if [[ ! -d ".venv" ]]; then
        log "  Creating virtual environment..."
        python3 -m venv .venv
    else
        log "  Virtual environment already exists."
    fi

    # shellcheck source=/dev/null
    source .venv/bin/activate
    log "  Python: $(python --version)"

    log "  Installing Sara RAG dependencies + paper tools..."
    pip install -e ".[rag]" reportlab pypdf -q
    log "  Dependencies installed (including reportlab + pypdf for paper rebuild)."
    echo ""

    # ── STEP 4: Sanity check ─────────────────────────────────────────────────
    log "STEP 4 — Sanity check (single query, ~1 min)"
    python examples/08_ollama_kd_spar.py \
        --teacher llama3.1:8b \
        --student llama3.2:3b \
        --sanity-only
    log "Sanity check passed."
    echo ""

else
    # --skip-setup: just activate venv
    [[ -d "${SARA_DIR}" ]] || fail "Sara directory not found: ${SARA_DIR}"
    cd "${SARA_DIR}"
    source .venv/bin/activate
    log "Setup skipped — using existing environment in $(pwd)"
    echo ""
fi

# ── STEP 5: Run ablation ─────────────────────────────────────────────────────
log "STEP 5 — Running KD-SPAR ablation"
if [[ "${QUICK_MODE}" == "true" ]]; then
    log "  Mode: QUICK (Config 1, 3 seeds, ~20 min)"
else
    log "  Mode: FULL PUBLICATION (${CONFIGS} configs × $(echo ${SEEDS} | wc -w) seeds)"
fi
echo ""

# Count total runs for progress display
total_runs=0
for _cfg in ${CONFIGS}; do
    for _seed in ${SEEDS}; do
        ((total_runs++)) || true
    done
done
run_num=0

for cfg in ${CONFIGS}; do
    for seed in ${SEEDS}; do
        ((run_num++)) || true
        log "  [${run_num}/${total_runs}] config=${cfg}  seed=${seed}  iterations=${ITERATIONS}"
        python experiments/kd_spar_ablation_ollama.py \
            --config "${cfg}" \
            --iterations "${ITERATIONS}" \
            --seed "${seed}"
        echo ""
    done
done

# ── STEP 6: Aggregate results ─────────────────────────────────────────────────
log "STEP 6 — Aggregating results"
python experiments/collect_results.py
echo ""

# ── STEP 7: Patch paper with real numbers ─────────────────────────────────────
log "STEP 7 — Patching paper with real experimental results"
# Ensure reportlab + pypdf are available (in case --skip-setup was used)
pip install reportlab pypdf -q 2>/dev/null || true
python experiments/patch_paper.py \
    --output docs/paper/Sara_Knowledge_Distillation.pdf \
    && log "Paper updated: docs/paper/Sara_Knowledge_Distillation.pdf" \
    || warn "Paper patch failed — run manually: python experiments/patch_paper.py"
echo ""

# ── STEP 8: View results summary ──────────────────────────────────────────────
log "STEP 8 — Results summary"
python experiments/results_analysis.py
echo ""

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}=================================================${NC}"
echo -e "${GREEN}  Sara ablation complete!${NC}"
echo ""
echo "  Results    : experiments/results/"
echo "  Paper      : docs/paper/Sara_Knowledge_Distillation.pdf"
echo "  Key metric : A-B gap (see summary above)"
echo ""
echo "  A-B gap > 0.02  →  strong self-knowledge evidence"
echo "  A-B gap > 0.01  →  moderate support"
echo "  A-B gap < 0.01  →  run more seeds or iterations"
echo -e "${GREEN}=================================================${NC}"
