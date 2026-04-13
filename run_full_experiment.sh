#!/usr/bin/env bash
# Sara (सार) — Master Experiment Script  v1.8.3
# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ajsinha/sara
#
# Comprehensive ablation across 7 student architectures (1.1B–3.8B),
# constant teacher (llama3.1:8b), 2 domains, 5 seeds.
#
# Student lineup (capacity curve):
#   tinyllama:1.1b   — 1.1B  — capacity floor
#   stablelm2:1.6b   — 1.6B  — alternative training methodology
#   smollm2:1.7b     — 1.7B  — sub-2B test (HuggingFace)
#   gemma2:2b        — 2.0B  — Google, small but well-trained
#   llama3.2:3b      — 3.0B  — Meta, same-family baseline
#   qwen2.5:3b       — 3.0B  — Alibaba, cross-family at 3B
#   phi3:3.8b        — 3.8B  — Microsoft, distillation-born model
#
# Total: 7 students × 2 domains × 5 seeds = 70 ablation runs
# Each run tests 6 conditions (D,C,B,A,E,F) → 420 condition evaluations
# Estimated time: 12-16 hours on OryxPro (RTX 3070 Ti)
#
# Usage:
#   bash run_full_experiment.sh                  # full run (70 runs, ~17 hours)
#   bash run_full_experiment.sh --fast           # thermal-safe (3 seeds, 3 iters, 60s cooldown, RAG only)
#   bash run_full_experiment.sh --quick          # quick test (3B models, 3 seeds, RAG only)
#   bash run_full_experiment.sh --rag-only       # RAG domain only (35 runs)
#   bash run_full_experiment.sh --code-only      # Code domain only (35 runs)
#   bash run_full_experiment.sh --3b-only        # Only 3B+ models (3 students)
#   bash run_full_experiment.sh --small-only     # Only sub-2B models (3 students)
#   bash run_full_experiment.sh --light-teacher  # Use phi3:3.8b teacher (half VRAM, 2x faster)
#   bash run_full_experiment.sh --cooldown=60    # 60s GPU cooling pause between runs
#   bash run_full_experiment.sh --resume         # Skip runs that already have results
#   bash run_full_experiment.sh --iterations=3   # Custom iteration count
#
# Recommended for OryxPro (RTX 3070 Ti, thermal-safe):
#   bash run_full_experiment.sh --fast                          # ~6 hours, RAG only
#   bash run_full_experiment.sh --fast --code-only              # ~6 hours, Code only (run next day)
#   bash run_full_experiment.sh --fast --resume                 # resume after crash
#   bash run_full_experiment.sh --light-teacher --fast          # fastest: phi3 teacher, ~4 hours

set -e

# ── Configuration ──────────────────────────────────────────────────────────

TEACHER="llama3.1:8b"

# All 7 student models — ordered by parameter count (capacity curve)
ALL_STUDENTS=(
    "tinyllama:1.1b"    # 1.1B — TinyLlama (community)
    "stablelm2:1.6b"    # 1.6B — StableLM 2 (Stability AI)
    "smollm2:1.7b"      # 1.7B — SmolLM 2 (HuggingFace)
    "gemma2:2b"         # 2.0B — Gemma 2 (Google)
    "llama3.2:3b"       # 3.0B — Llama 3.2 (Meta) — primary baseline
    "qwen2.5:3b"        # 3.0B — Qwen 2.5 (Alibaba)
    "phi3:3.8b"         # 3.8B — Phi-3 (Microsoft) — distillation-born
)

# Subsets for targeted runs
SMALL_STUDENTS=("tinyllama:1.1b" "stablelm2:1.6b" "smollm2:1.7b")
MEDIUM_STUDENTS=("gemma2:2b" "llama3.2:3b" "qwen2.5:3b" "phi3:3.8b")
THREE_B_STUDENTS=("llama3.2:3b" "qwen2.5:3b" "phi3:3.8b")

SEEDS=(42 123 456 789 101)
ITERATIONS=5
DOMAINS=("rag" "code")
COOLDOWN=0        # seconds to pause between runs (GPU cooling)
RESUME=false      # skip already-completed runs
RESULTS_DIR="experiments/results"

# ── Parse arguments ────────────────────────────────────────────────────────

STUDENTS=("${ALL_STUDENTS[@]}")  # default: all 7

for arg in "$@"; do
    case $arg in
        --quick)
            STUDENTS=("${THREE_B_STUDENTS[@]}")
            SEEDS=(42 123 456)
            DOMAINS=("rag")
            ITERATIONS=3
            ;;
        --fast)
            # Thermal-safe: 3 seeds, 3 iterations, cooldown, RAG only
            SEEDS=(42 123 456)
            ITERATIONS=3
            COOLDOWN=60
            DOMAINS=("rag")
            ;;
        --light-teacher)
            # Use phi3:3.8b as teacher — half VRAM, 2x faster inference
            TEACHER="phi3:3.8b"
            ;;
        --teacher=*)    TEACHER="${arg#*=}" ;;
        --rag-only)     DOMAINS=("rag") ;;
        --code-only)    DOMAINS=("code") ;;
        --3b-only)      STUDENTS=("${THREE_B_STUDENTS[@]}") ;;
        --small-only)   STUDENTS=("${SMALL_STUDENTS[@]}") ;;
        --medium-only)  STUDENTS=("${MEDIUM_STUDENTS[@]}") ;;
        --all)          STUDENTS=("${ALL_STUDENTS[@]}") ;;
        --iterations=*) ITERATIONS="${arg#*=}" ;;
        --seeds=*)      IFS=',' read -ra SEEDS <<< "${arg#*=}" ;;
        --cooldown=*)   COOLDOWN="${arg#*=}" ;;
        --resume)       RESUME=true ;;
    esac
done

# Remove teacher from student list if using light teacher
if [ "$TEACHER" != "llama3.1:8b" ]; then
    FILTERED_STUDENTS=()
    for s in "${STUDENTS[@]}"; do
        if [ "$s" != "$TEACHER" ]; then
            FILTERED_STUDENTS+=("$s")
        fi
    done
    STUDENTS=("${FILTERED_STUDENTS[@]}")
fi

TOTAL=$((${#STUDENTS[@]} * ${#DOMAINS[@]} * ${#SEEDS[@]}))

echo "════════════════════════════════════════════════════════════"
echo "  Sara (सार) — Master Experiment  v1.8.3"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Teacher:    $TEACHER"
echo "  Students:   ${#STUDENTS[@]} models:"
for s in "${STUDENTS[@]}"; do
    echo "                $s"
done
echo "  Seeds:      ${SEEDS[*]}"
echo "  Domains:    ${DOMAINS[*]}"
echo "  Iterations: $ITERATIONS"
echo "  Cooldown:   ${COOLDOWN}s between runs"
echo "  Resume:     $RESUME"
echo "  Conditions: D, C, B, A, E (MetaKDSPAR), F (Enhanced)"
echo ""
echo "  Total runs: $TOTAL  (× 6 conditions = $((TOTAL * 6)) evaluations)"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

# ── Setup ──────────────────────────────────────────────────────────────────

cd "$(dirname "$0")"
echo "[1/4] Setting up environment..."

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "  Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

# Install dependencies
pip install -e ".[rag]" reportlab pypdf -q 2>/dev/null || true
echo "  Dependencies installed."

# ── Check Ollama ───────────────────────────────────────────────────────────

echo ""
echo "[2/4] Checking Ollama..."

if ! command -v ollama &> /dev/null; then
    echo "  ERROR: Ollama not installed."
    echo "  Install: curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

# Start Ollama if not running
if ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  Starting Ollama server..."
    ollama serve &> /tmp/ollama_master.log &
    sleep 5
fi
echo "  Ollama running."

# ── Pull all models ────────────────────────────────────────────────────────

echo ""
echo "[3/4] Pulling models (one-time downloads)..."
echo ""

# Teacher
echo "  Pulling teacher: $TEACHER"
ollama pull "$TEACHER" 2>/dev/null || echo "    ⚠ Failed to pull $TEACHER"

# All student models
for student in "${STUDENTS[@]}"; do
    echo "  Pulling student: $student"
    ollama pull "$student" 2>/dev/null || echo "    ⚠ Failed to pull $student"
done

echo ""
echo "  Available models:"
ollama list 2>/dev/null | grep -E "llama|qwen|gemma|phi|smollm|stablelm|tinyllama" || true
echo ""

# ── Run experiments ────────────────────────────────────────────────────────

echo "[4/4] Running experiments..."
echo ""

RUN_COUNT=0
FAIL_COUNT=0
SUCCESS_COUNT=0
TOTAL_ACCEPTED=0
TOTAL_REVERTED=0
TOTAL_EVALS_DONE=0
TOTAL_EVALS=$((TOTAL * 6))
START_TIME=$(date +%s)

# Progress log file
PROGRESS_LOG=$(mktemp /tmp/sara_progress_XXXXX.log)

print_progress() {
    # Args: $1=eval_num $2=total_evals $3=domain $4=student $5=seed $6=cond_info
    local EVAL_NUM=$1 EVAL_TOTAL=$2 DOMAIN=$3 STUDENT=$4 SEED=$5 COND=$6
    local ELAPSED=$(($(date +%s) - START_TIME))
    local ELAPSED_H=$((ELAPSED / 3600))
    local ELAPSED_M=$(( (ELAPSED % 3600) / 60 ))
    local ELAPSED_S=$((ELAPSED % 60))
    local PCT=0
    local ETA_STR="calculating..."

    if [ "$EVAL_NUM" -gt 0 ]; then
        PCT=$(( EVAL_NUM * 100 / EVAL_TOTAL ))
    fi

    if [ "$EVAL_NUM" -gt 1 ] && [ "$ELAPSED" -gt 0 ]; then
        local AVG=$(( ELAPSED * 1000 / EVAL_NUM ))  # ms per eval
        local REMAINING=$(( AVG * (EVAL_TOTAL - EVAL_NUM) / 1000 ))
        local ETA_H=$((REMAINING / 3600))
        local ETA_M=$(( (REMAINING % 3600) / 60 ))
        ETA_STR="${ETA_H}h ${ETA_M}m"
    fi

    # Build progress bar (50 chars wide)
    local BAR_DONE=$(( PCT / 2 ))
    local BAR_LEFT=$(( 50 - BAR_DONE ))
    local BAR=""
    for ((i=0; i<BAR_DONE; i++)); do BAR="${BAR}█"; done
    for ((i=0; i<BAR_LEFT; i++)); do BAR="${BAR}░"; done

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                  ║"
    printf "║   ▶  EVALUATION  %-4s  OF  %-4s   (Run %s/%s)                  ║\n" "$EVAL_NUM" "$EVAL_TOTAL" "$RUN_COUNT" "$TOTAL"
    echo "║                                                                  ║"
    printf "║   %s  %3d%%       ║\n" "$BAR" "$PCT"
    echo "║                                                                  ║"
    printf "║   Domain:    %-52s ║\n" "$DOMAIN"
    printf "║   Student:   %-52s ║\n" "$STUDENT"
    printf "║   Seed:      %-52s ║\n" "$SEED"
    printf "║   Condition: %-52s ║\n" "$COND"
    echo "║                                                                  ║"
    printf "║   Elapsed: %dh %02dm %02ds  │  ETA: %-28s   ║\n" "$ELAPSED_H" "$ELAPSED_M" "$ELAPSED_S" "$ETA_STR"
    printf "║   Runs:  ✓ %d  ✗ %d  │  Remaining: %-29s ║\n" "$SUCCESS_COUNT" "$FAIL_COUNT" "$((TOTAL - RUN_COUNT)) runs, $((EVAL_TOTAL - EVAL_NUM)) evals"
    echo "║                                                                  ║"
    printf "║   Cumulative:  ACCEPTED=%-5d  REVERTED=%-5d  EVALS=%-5d       ║\n" "$TOTAL_ACCEPTED" "$TOTAL_REVERTED" "$EVAL_NUM"
    echo "║                                                                  ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
}

for domain in "${DOMAINS[@]}"; do
    DOMAIN_FLAG=""
    DOMAIN_LABEL="RAG QA"
    if [ "$domain" = "code" ]; then
        DOMAIN_FLAG="--domain code"
        DOMAIN_LABEL="Code Generation"
    fi

    for student in "${STUDENTS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            RUN_COUNT=$((RUN_COUNT + 1))

            # ── Resume: skip if results already exist ─────────────────────
            if $RESUME; then
                STUDENT_SAFE="${student//:/_}"
                TEACHER_SAFE="${TEACHER//:/_}"
                DOMAIN_PREFIX=""
                if [ "$domain" = "code" ]; then DOMAIN_PREFIX="code_"; fi
                RESULT_PATTERN="${RESULTS_DIR}/ablation_ollama_${DOMAIN_PREFIX}*${STUDENT_SAFE}*seed${seed}*.json"
                EXISTING=$(ls $RESULT_PATTERN 2>/dev/null | wc -l)
                if [ "$EXISTING" -gt 0 ]; then
                    echo ""
                    echo "  ⏭  Skipping run $RUN_COUNT/$TOTAL (results exist for $student seed=$seed domain=$domain)"
                    TOTAL_EVALS_DONE=$((TOTAL_EVALS_DONE + 6))
                    echo "$TOTAL_EVALS_DONE" > /tmp/sara_eval_count.tmp
                    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                    continue
                fi
            fi

            # Print banner at start of run
            print_progress "$TOTAL_EVALS_DONE" "$TOTAL_EVALS" "$DOMAIN_LABEL" "$student" "$seed" "starting run..."

            # ── Cooldown between runs (GPU thermal management) ────────────
            if [ "$COOLDOWN" -gt 0 ] && [ "$RUN_COUNT" -gt 1 ]; then
                echo ""
                echo "  ❄  GPU cooldown: ${COOLDOWN}s pause..."
                sleep "$COOLDOWN"
            fi

            # Run and capture output line-by-line for real-time parsing
            RUN_OK=true
            python experiments/kd_spar_ablation_ollama.py \
                --teacher "$TEACHER" \
                --student "$student" \
                --iterations "$ITERATIONS" \
                --seed "$seed" \
                $DOMAIN_FLAG 2>&1 | while IFS= read -r line; do

                echo "$line"

                # Parse evaluation completions
                if echo "$line" | grep -q '^\[SARA_EVAL\]'; then
                    COND_INFO=$(echo "$line" | sed 's/\[SARA_EVAL\] //')
                    TOTAL_EVALS_DONE=$((TOTAL_EVALS_DONE + 1))

                    # Write to temp files so parent shell can read
                    echo "$TOTAL_EVALS_DONE" > /tmp/sara_eval_count.tmp
                    echo "$TOTAL_ACCEPTED" > /tmp/sara_accepted.tmp
                    echo "$TOTAL_REVERTED" > /tmp/sara_reverted.tmp

                    # Recalculate progress
                    local_PCT=$(( TOTAL_EVALS_DONE * 100 / TOTAL_EVALS ))
                    local_ELAPSED=$(($(date +%s) - START_TIME))
                    local_EH=$((local_ELAPSED / 3600))
                    local_EM=$(( (local_ELAPSED % 3600) / 60 ))
                    local_ES=$((local_ELAPSED % 60))
                    local_ETA="calculating..."
                    if [ "$TOTAL_EVALS_DONE" -gt 1 ] && [ "$local_ELAPSED" -gt 0 ]; then
                        local_AVG=$(( local_ELAPSED * 1000 / TOTAL_EVALS_DONE ))
                        local_REM=$(( local_AVG * (TOTAL_EVALS - TOTAL_EVALS_DONE) / 1000 ))
                        local_ETA="$((local_REM / 3600))h $((local_REM % 3600 / 60))m"
                    fi
                    local_BAR=""
                    local_BD=$((local_PCT / 2))
                    local_BL=$((50 - local_BD))
                    for ((bi=0; bi<local_BD; bi++)); do local_BAR="${local_BAR}█"; done
                    for ((bi=0; bi<local_BL; bi++)); do local_BAR="${local_BAR}░"; done

                    echo ""
                    echo "╔══════════════════════════════════════════════════════════════════╗"
                    echo "║                                                                  ║"
                    printf "║   ▶  EVALUATION  %-4s  OF  %-4s   (Run %s/%s)                  ║\n" "$TOTAL_EVALS_DONE" "$TOTAL_EVALS" "$RUN_COUNT" "$TOTAL"
                    echo "║                                                                  ║"
                    printf "║   %s  %3d%%       ║\n" "$local_BAR" "$local_PCT"
                    echo "║                                                                  ║"
                    printf "║   Domain:    %-52s ║\n" "$DOMAIN_LABEL"
                    printf "║   Student:   %-52s ║\n" "$student"
                    printf "║   Seed:      %-52s ║\n" "$seed"
                    printf "║   Condition: %-52s ║\n" "$COND_INFO"
                    echo "║                                                                  ║"
                    printf "║   Elapsed: %dh %02dm %02ds  │  ETA: %-28s   ║\n" "$local_EH" "$local_EM" "$local_ES" "$local_ETA"
                    printf "║   Runs:  ✓ %d  ✗ %d  │  Remaining: %-29s ║\n" "$SUCCESS_COUNT" "$FAIL_COUNT" "$((TOTAL - RUN_COUNT)) runs, $((TOTAL_EVALS - TOTAL_EVALS_DONE)) evals"
                    echo "║                                                                  ║"
                    printf "║   Cumulative:  ACCEPTED=%-5d  REVERTED=%-5d  EVALS=%-5d       ║\n" "$TOTAL_ACCEPTED" "$TOTAL_REVERTED" "$TOTAL_EVALS_DONE"
                    echo "║                                                                  ║"
                    echo "╚══════════════════════════════════════════════════════════════════╝"
                fi

                # Parse iteration accept/revert
                if echo "$line" | grep -q '^\[SARA_ITER\].*ACCEPTED'; then
                    TOTAL_ACCEPTED=$((TOTAL_ACCEPTED + 1))
                    echo "$TOTAL_ACCEPTED" > /tmp/sara_accepted.tmp
                fi
                if echo "$line" | grep -q '^\[SARA_ITER\].*REVERTED'; then
                    TOTAL_REVERTED=$((TOTAL_REVERTED + 1))
                    echo "$TOTAL_REVERTED" > /tmp/sara_reverted.tmp
                fi

            done || RUN_OK=false

            # Read back counters from temp files (subshell can't modify parent vars)
            if [ -f /tmp/sara_eval_count.tmp ]; then
                TOTAL_EVALS_DONE=$(cat /tmp/sara_eval_count.tmp)
            else
                # Fallback: assume 6 evals per run
                TOTAL_EVALS_DONE=$((RUN_COUNT * 6))
            fi
            [ -f /tmp/sara_accepted.tmp ] && TOTAL_ACCEPTED=$(cat /tmp/sara_accepted.tmp)
            [ -f /tmp/sara_reverted.tmp ] && TOTAL_REVERTED=$(cat /tmp/sara_reverted.tmp)

            if [ "$?" -eq 0 ] && $RUN_OK; then
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
                echo ""
                echo "  ✓ Run $RUN_COUNT/$TOTAL COMPLETE │ Evals: $TOTAL_EVALS_DONE/$TOTAL_EVALS │ ACCEPTED=$TOTAL_ACCEPTED REVERTED=$TOTAL_REVERTED"
            else
                FAIL_COUNT=$((FAIL_COUNT + 1))
                # Still count 6 evals even on failure
                TOTAL_EVALS_DONE=$((RUN_COUNT * 6))
                echo "$TOTAL_EVALS_DONE" > /tmp/sara_eval_count.tmp
                echo ""
                echo "  ✗ Run $RUN_COUNT/$TOTAL FAILED"
            fi

            # Append to progress log
            echo "[$(date '+%H:%M:%S')] Run $RUN_COUNT/$TOTAL | Eval $TOTAL_EVALS_DONE/$TOTAL_EVALS | $student seed=$seed domain=$domain | OK=$SUCCESS_COUNT FAIL=$FAIL_COUNT | ACCEPTED=$TOTAL_ACCEPTED REVERTED=$TOTAL_REVERTED" >> "$PROGRESS_LOG"

        done
    done
done

# Cleanup temp files
rm -f /tmp/sara_eval_count.tmp /tmp/sara_accepted.tmp /tmp/sara_reverted.tmp

# ── Aggregate and patch paper ──────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Post-processing"
echo "════════════════════════════════════════════════════════════"
echo ""

echo "  Aggregating results..."
python experiments/collect_results.py || echo "  ⚠ Aggregation failed"

echo ""
echo "  Patching paper with results..."
python experiments/patch_paper.py \
    --output docs/paper/Sara_Knowledge_Distillation.pdf \
    2>/dev/null || echo "  ⚠ Paper patch failed — run manually"

echo ""
echo "  Generating analysis..."
python experiments/results_analysis.py 2>/dev/null || true

# ── Summary ────────────────────────────────────────────────────────────────

TOTAL_TIME=$(( $(date +%s) - START_TIME ))
HOURS=$((TOTAL_TIME / 3600))
MINS=$(( (TOTAL_TIME % 3600) / 60 ))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                                                              ║"
echo "║   Sara (सार) — Master Experiment COMPLETE                    ║"
echo "║                                                              ║"
echo "║   ██████████████████████████████████████████████████  100%    ║"
echo "║                                                              ║"
echo "║   Completed:  $SUCCESS_COUNT / $TOTAL successful                            ║"
if [ $FAIL_COUNT -gt 0 ]; then
echo "║   Failed:     $FAIL_COUNT runs                                          ║"
fi
echo "║   Time:       ${HOURS}h ${MINS}m                                        ║"
echo "║                                                              ║"
echo "║   ┌─────────────────────────────────────────────────┐        ║"
echo "║   │  CUMULATIVE RESULTS                             │        ║"
echo "║   │                                                 │        ║"
echo "║   │  Total Evaluations Done:     $TOTAL_EVALS_DONE / $TOTAL_EVALS            │        ║"
echo "║   │  Total ACCEPTED (improved):   $TOTAL_ACCEPTED               │        ║"
echo "║   │  Total REVERTED (no gain):    $TOTAL_REVERTED               │        ║"
if [ $((TOTAL_ACCEPTED + TOTAL_REVERTED)) -gt 0 ]; then
echo "║   │  Accept Rate:  $(( TOTAL_ACCEPTED * 100 / (TOTAL_ACCEPTED + TOTAL_REVERTED) ))%                              │        ║"
fi
echo "║   └─────────────────────────────────────────────────┘        ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Results:      experiments/results/"
echo "  Paper:        docs/paper/Sara_Knowledge_Distillation.pdf"
echo "  Progress log: $PROGRESS_LOG"
echo ""
echo "  Key output files:"
echo "    experiments/results/aggregated_results.json"
echo "    experiments/results/aggregated_results.txt"
echo ""
echo "  Post-experiment commands:"
echo "    python experiments/results_analysis.py        # detailed analysis"
echo "    python experiments/human_eval.py generate     # blind eval sheets"
echo ""
echo "  Student capacity curve (plot A−B gap vs model size):"
for s in "${STUDENTS[@]}"; do
    echo "    $s"
done
echo ""
