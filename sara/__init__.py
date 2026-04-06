# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.6.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ashutosh-sinha/sara
"""
Sara (सार) — Knowledge Distillation Toolkit
=============================================
A complete, modular library for knowledge distillation across vision, NLP, and RAG.

Subpackages
-----------
sara.core      : Loss functions, utilities, and progress logging
sara.vision    : Vision-domain distillation (response-based, feature-based, AT)
sara.nlp       : NLP distillation (BERT-family)
sara.advanced  : Progressive, mutual, self-distillation, relation-based
sara.rag       : RAG-specific KD — pipeline, migration, KD-SPAR, prompt opt
"""

from sara.core.losses import (
    DistillationLoss,
    FeatureDistillationLoss,
    AttentionTransferLoss,
    RKDLoss,
)
from sara.core.utils import (
    profile_model,
    ProfileResult,
    recommend_hyperparams,
    load_config,
)
from sara.core.progress import SaraLogger, Heartbeat, ProgressBar, phase

__all__ = [
    "DistillationLoss",
    "FeatureDistillationLoss",
    "AttentionTransferLoss",
    "RKDLoss",
    "profile_model",
    "ProfileResult",
    "recommend_hyperparams",
    "load_config",
    "SaraLogger",
    "Heartbeat",
    "ProgressBar",
    "phase",
]

__version__ = "1.6.0"
__author__  = "Ashutosh Sinha"
