"""kd.vision — Vision distillation trainers (response-based, feature-based, attention transfer)."""
from sara.vision.response_based import ResponseBasedDistiller, VisionDistillConfig, build_cifar10_loaders, build_default_teacher_student
from sara.vision.feature_based import FeatureBasedDistiller, attach_feature_hooks
from sara.vision.attention_transfer import AttentionTransferDistiller
__all__ = ["ResponseBasedDistiller","VisionDistillConfig","build_cifar10_loaders","build_default_teacher_student","FeatureBasedDistiller","attach_feature_hooks","AttentionTransferDistiller"]
