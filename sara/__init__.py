"""
Knowledge Distillation Toolkit
================================
A complete, modular library for knowledge distillation across vision, NLP, and RAG.

Subpackages
-----------
kd.core      : Loss functions and utilities shared by all modules
kd.vision    : Vision-domain distillation (response-based, feature-based, AT)
kd.nlp       : NLP distillation (BERT-family)
kd.advanced  : Progressive, mutual, self-distillation, relation-based
kd.rag       : RAG-specific KD — pipeline, migration, KD-SPAR, prompt opt
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

__all__ = [
    "DistillationLoss",
    "FeatureDistillationLoss",
    "AttentionTransferLoss",
    "RKDLoss",
    "profile_model",
    "ProfileResult",
    "recommend_hyperparams",
    "load_config",
]

__version__ = "1.0.0"
__author__  = "Ashutosh Sinha"
