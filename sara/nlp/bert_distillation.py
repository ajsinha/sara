# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.8.3
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ajsinha/sara
from __future__ import annotations
"""
sara.nlp.bert_distillation
=========================
BERT-family knowledge distillation using the Hugging Face Trainer API.

The :class:`BertDistillationTrainer` extends ``transformers.Trainer`` with a
three-term loss:

    L = α · T² · KL(student_soft ‖ teacher_soft)
      + β · MSE(student_CLS_hidden, teacher_CLS_hidden)
      + (1−α) · CE(student_logits, labels)

The :func:`run_bert_distillation` convenience function handles dataset loading,
tokenisation, model construction, and training in one call.

Requirements
------------
    pip install transformers datasets evaluate accelerate
"""


from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class BertDistillConfig:
    """
    Hyperparameters for BERT-family distillation.

    Parameters
    ----------
    alpha       : Weight for the KL soft-target term  [0, 1]
    beta        : Weight for the hidden-state MSE term
    temperature : Softening temperature T
    epochs      : Training epochs
    batch_size  : Per-device train batch size
    lr          : AdamW learning rate
    warmup_ratio: Fraction of steps used for linear warm-up
    output_dir  : Checkpoint and log directory
    fp16        : Use mixed-precision if CUDA is available
    """
    alpha:       float = 0.5
    beta:        float = 0.01
    temperature: float = 4.0
    epochs:      int   = 5
    batch_size:  int   = 32
    lr:          float = 2e-5
    warmup_ratio:float = 0.06
    output_dir:  str   = "./output/bert_distil"
    fp16:        bool  = True


# ── Custom Trainer ──────────────────────────────────────────────────────────────

class BertDistillationTrainer:
    """
    Hugging Face Trainer wrapper for BERT → DistilBERT distillation.

    Injects three-term KD loss (KL + hidden-state MSE + CE) via the
    ``compute_loss`` override.

    Parameters
    ----------
    teacher_id  : HuggingFace model ID for the teacher (e.g. ``"bert-base-uncased"``)
    student_id  : HuggingFace model ID for the student (e.g. ``"distilbert-base-uncased"``)
                  Pass ``None`` to use a freshly initialised DistilBERT-6-layer.
    config      : :class:`BertDistillConfig` with hyperparameters
    num_labels  : Number of classification labels

    Examples
    --------
    >>> cfg     = BertDistillConfig(alpha=0.5, temperature=4.0, epochs=5)
    >>> trainer = BertDistillationTrainer("bert-base-uncased", None, cfg)
    >>> trainer.train(train_dataset, val_dataset, tokenizer)
    """

    def __init__(
        self,
        teacher_id:  str,
        student_id:  Optional[str] = None,
        config:      Optional[BertDistillConfig] = None,
        num_labels:  int = 2,
    ) -> None:
        self.teacher_id = teacher_id
        self.student_id = student_id
        self.config     = config or BertDistillConfig()
        self.num_labels = num_labels
        self._trainer   = None

    def train(
        self,
        train_dataset: Dataset,
        val_dataset:   Dataset,
        tokenizer,
        compute_metrics=None,
    ):
        """
        Build teacher, student, and Trainer; run distillation.

        Parameters
        ----------
        train_dataset   : Tokenised training dataset
        val_dataset     : Tokenised validation dataset
        tokenizer       : HuggingFace tokeniser matching the teacher
        compute_metrics : Optional metrics function for ``Trainer``

        Returns
        -------
        ``transformers.Trainer`` instance (already trained)
        """
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            DistilBertConfig,
            DistilBertForSequenceClassification,
            Trainer,
            TrainingArguments,
            DataCollatorWithPadding,
        )

        cfg  = self.config
        dev  = "cuda" if torch.cuda.is_available() else "cpu"

        teacher = AutoModelForSequenceClassification.from_pretrained(
            self.teacher_id, num_labels=self.num_labels
        ).to(dev).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        if self.student_id:
            student = AutoModelForSequenceClassification.from_pretrained(
                self.student_id, num_labels=self.num_labels
            )
        else:
            # Default: 6-layer DistilBERT with same hidden dim as BERT-base
            dcfg    = DistilBertConfig(
                n_layers=6, dim=768, hidden_dim=3072,
                num_labels=self.num_labels,
            )
            student = DistilBertForSequenceClassification(dcfg)

        alpha, beta, T = cfg.alpha, cfg.beta, cfg.temperature
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        mse_loss = nn.MSELoss()

        class _DistillTrainer(Trainer):
            def compute_loss(self_, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                s_out  = model(**inputs, output_hidden_states=True)
                with torch.no_grad():
                    t_out = teacher(**inputs, output_hidden_states=True)

                kd = kl_loss(
                    F.log_softmax(s_out.logits / T, dim=-1),
                    F.softmax(t_out.logits.detach() / T, dim=-1),
                ) * T ** 2
                hs = mse_loss(
                    s_out.hidden_states[-1][:, 0, :],
                    t_out.hidden_states[-1][:, 0, :].detach(),
                )
                ce   = F.cross_entropy(s_out.logits, labels)
                loss = alpha * kd + beta * hs + (1.0 - alpha) * ce
                return (loss, s_out) if return_outputs else loss

        args = TrainingArguments(
            output_dir                  = cfg.output_dir,
            num_train_epochs            = cfg.epochs,
            per_device_train_batch_size = cfg.batch_size,
            per_device_eval_batch_size  = cfg.batch_size * 2,
            learning_rate               = cfg.lr,
            weight_decay                = 0.01,
            warmup_ratio                = cfg.warmup_ratio,
            lr_scheduler_type           = "cosine",
            evaluation_strategy         = "epoch",
            save_strategy               = "epoch",
            load_best_model_at_end      = True,
            fp16                        = cfg.fp16 and torch.cuda.is_available(),
            report_to                   = "none",
        )

        self._trainer = _DistillTrainer(
            model           = student,
            args            = args,
            train_dataset   = train_dataset,
            eval_dataset    = val_dataset,
            tokenizer       = tokenizer,
            data_collator   = DataCollatorWithPadding(tokenizer),
            compute_metrics = compute_metrics,
        )
        self._trainer.train()
        return self._trainer

    @property
    def trainer(self):
        """Access the underlying ``transformers.Trainer`` after ``train()``."""
        return self._trainer


# ── Convenience function ────────────────────────────────────────────────────────

def run_bert_distillation(
    teacher_id:  str   = "bert-base-uncased",
    student_id:  Optional[str] = None,
    dataset_name: str  = "glue",
    dataset_config: str= "sst2",
    text_col:    str   = "sentence",
    label_col:   str   = "label",
    max_length:  int   = 128,
    config:      Optional[BertDistillConfig] = None,
    num_labels:  int   = 2,
):
    """
    End-to-end BERT distillation: load dataset, tokenise, train, evaluate.

    Parameters
    ----------
    teacher_id      : HuggingFace model ID for the teacher
    student_id      : HuggingFace model ID for the student (None = default DistilBERT)
    dataset_name    : HuggingFace datasets name (e.g. ``"glue"``)
    dataset_config  : Dataset config / task (e.g. ``"sst2"``)
    text_col        : Column name for the text input
    label_col       : Column name for the label
    max_length      : Tokeniser max sequence length
    config          : :class:`BertDistillConfig`
    num_labels      : Number of classification labels

    Returns
    -------
    ``transformers.Trainer`` trained on the distillation objective

    Examples
    --------
    >>> trainer = run_bert_distillation(epochs=3)
    >>> metrics = trainer.evaluate()
    """
    from transformers import AutoTokenizer  # type: ignore
    from datasets import load_dataset       # type: ignore
    import evaluate as hf_evaluate          # type: ignore

    cfg       = config or BertDistillConfig()
    tokenizer = AutoTokenizer.from_pretrained(teacher_id)
    ds        = load_dataset(dataset_name, dataset_config)

    def _tokenize(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    ds = ds.map(_tokenize, batched=True, remove_columns=[text_col])
    if "idx" in ds.column_names.get("train", []):
        ds = ds.map(lambda x: x, remove_columns=["idx"])
    ds = ds.rename_column(label_col, "labels")
    ds.set_format("torch")

    metric = hf_evaluate.load(dataset_name, dataset_config)

    def compute_metrics(ep):
        preds = ep.predictions.argmax(-1)
        return metric.compute(predictions=preds, references=ep.label_ids)

    trainer_wrapper = BertDistillationTrainer(
        teacher_id=teacher_id, student_id=student_id,
        config=cfg, num_labels=num_labels,
    )
    return trainer_wrapper.train(
        ds["train"], ds["validation"], tokenizer, compute_metrics
    )
