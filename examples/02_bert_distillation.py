# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.6.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
"""
examples/02_bert_distillation.py
==================================
Distil BERT-base → DistilBERT on SST-2.

Run:
    python examples/02_bert_distillation.py

Requirements:
    pip install -e ".[nlp]"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sara.core.progress import SaraLogger
from sara.nlp.bert_distillation import BertDistillConfig, run_bert_distillation


def main():
    log = SaraLogger("BERT KD")
    log.banner(
        "Sara — BERT → DistilBERT Distillation",
        "Teacher : bert-base-uncased",
        "Student : DistilBERT-6L (auto-built)",
        "Dataset : GLUE / SST-2",
    )

    cfg = BertDistillConfig(
        alpha=0.5, beta=0.01, temperature=4.0,
        epochs=5, batch_size=32, lr=2e-5,
        output_dir="./output/bert_distil", fp16=True,
    )
    log.section("Configuration")
    log.info(f"  alpha={cfg.alpha}  beta={cfg.beta}  T={cfg.temperature}")
    log.info(f"  epochs={cfg.epochs}  batch={cfg.batch_size}  lr={cfg.lr}  fp16={cfg.fp16}")
    log.info(f"  output → {cfg.output_dir}")

    log.section("Loading data and models")
    log.info("  Downloading GLUE/SST-2 and loading teacher…")
    log.start_heartbeat(interval=30, message="Loading models/data from HuggingFace…")

    trainer = run_bert_distillation(
        teacher_id="bert-base-uncased", student_id=None,
        dataset_name="glue", dataset_config="sst2",
        text_col="sentence", label_col="label",
        max_length=128, config=cfg, num_labels=2,
    )
    log.stop_heartbeat()

    log.section("Evaluation")
    log.step("Running final evaluation on validation set")
    metrics = trainer.evaluate()
    log.done("Evaluation complete")

    log.section("Results")
    log.metric("Val accuracy", f"{metrics.get('eval_accuracy', 0):.4f}")
    log.metric("Val loss",     f"{metrics.get('eval_loss', 0):.4f}")
    log.metric("Saved to",     f"{cfg.output_dir}/final")
    log.summary()


if __name__ == "__main__":
    main()
