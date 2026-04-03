"""
examples/02_bert_distillation.py
==================================
Distil BERT-base → DistilBERT on SST-2.
Uses kd.nlp.bert_distillation — the proper package module.

Run:
    python examples/02_bert_distillation.py

Requirements:
    pip install torch transformers datasets evaluate accelerate
"""

from sara.nlp.bert_distillation import BertDistillConfig, run_bert_distillation


def main():
    print("BERT → DistilBERT distillation on SST-2")
    print("Teacher : bert-base-uncased  |  Student : DistilBERT-6L (auto-built)")

    cfg = BertDistillConfig(
        alpha=0.5, beta=0.01, temperature=4.0,
        epochs=5, batch_size=32, lr=2e-5,
        output_dir="./output/bert_distil", fp16=True,
    )

    trainer = run_bert_distillation(
        teacher_id="bert-base-uncased", student_id=None,
        dataset_name="glue", dataset_config="sst2",
        text_col="sentence", label_col="label",
        max_length=128, config=cfg, num_labels=2,
    )

    metrics = trainer.evaluate()
    print(f"\nFinal val accuracy: {metrics.get('eval_accuracy', 'N/A'):.4f}")
    print(f"Model saved to    : {cfg.output_dir}/final")


if __name__ == "__main__":
    main()
