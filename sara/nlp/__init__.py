# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.7.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ajsinha/sara
"""sara.nlp — NLP / BERT-family distillation."""
from sara.nlp.bert_distillation import BertDistillationTrainer, BertDistillConfig, run_bert_distillation
__all__ = ["BertDistillationTrainer","BertDistillConfig","run_bert_distillation"]
