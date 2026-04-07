# Copyright (C) 2025 Ashutosh Sinha (ajsinha@gmail.com)
# Sara (सार) — Knowledge Distillation and KD-SPAR Toolkit  v1.7.0
# SPDX-License-Identifier: AGPL-3.0-or-later
# https://github.com/ajsinha/sara
"""sara.vision — Vision distillation trainers (response-based, feature-based, attention transfer)."""
from sara.vision.response_based import ResponseBasedDistiller, VisionDistillConfig, build_cifar10_loaders, build_default_teacher_student
from sara.vision.feature_based import FeatureBasedDistiller, attach_feature_hooks
from sara.vision.attention_transfer import AttentionTransferDistiller
__all__ = ["ResponseBasedDistiller","VisionDistillConfig","build_cifar10_loaders","build_default_teacher_student","FeatureBasedDistiller","attach_feature_hooks","AttentionTransferDistiller"]
