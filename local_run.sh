#!/usr/bin/env bash
# Sara (सार) — OryxPro Setup & Ablation Script
# Save as: setup_and_run.sh
# Run with: bash setup_and_run.sh
set -e
# exit on first error
# ── STEP 1: Install Ollama (one-time) ───────────────────────────────────
# ── STEP 2: Pull models (one-time, uses disk not GPU) ───────────────────
ollama pull llama3.1:8b
ollama pull llama3.2:3b
ollama pull qwen2.5:7b
ollama pull qwen2.5:3b
# ── STEP 3: Project setup ───────────────────────────────────────────────
cd ~/PycharmProjects/sara
# adjust path if different
# Create virtual environment (skip if already exists)
if [ ! -d '.venv' ]; then
	python3 -m venv .venv
fi
. .venv/bin/activate
# Install Sara RAG dependencies only (no PyTorch needed for ablation)
pip install -e '.[rag]'
# ── STEP 4: Sanity check ────────────────────────────────────────────────
python examples/08_ollama_kd_spar.py --teacher llama3.1:8b --student llama3.2:3b
# ── STEP 5: Run the ablation (publication run: 5 seeds x 2 configs) ─────
# Config 1: llama3.1:8b (teacher) -> llama3.2:3b (student) — same family
# Config 2: qwen2.5:7b (teacher) -> llama3.2:3b (student) — cross family
for cfg in 1 2; do
	for seed in 42 123 456 789 101; do
		echo "Running config=$cfg seed=$seed ..."
		python experiments/kd_spar_ablation_ollama.py --config $cfg --iterations 3 --seed $seed
	done
done
