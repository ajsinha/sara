from __future__ import annotations
"""
kd.core.losses
==============
All knowledge distillation loss functions in one place.

Classes
-------
DistillationLoss       Response-based KD (soft-target + cross-entropy)
FeatureDistillationLoss  FitNets-style intermediate feature alignment
AttentionTransferLoss  Spatial attention-map transfer
RKDLoss                Relational KD (pairwise distance + angle)
SelfDistillationLoss   Multi-exit self-supervised loss
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ── 1. Response-Based Distillation Loss ───────────────────────────────────────

class DistillationLoss(nn.Module):
    """
    Standard Hinton-style response-based KD loss.

    Combines KL divergence on teacher soft targets with cross-entropy
    on hard labels:

        L = alpha * T^2 * KL(student_soft || teacher_soft)
          + (1 - alpha) * CE(student_logits, labels)

    The T^2 factor restores the gradient magnitude lost when softening
    the distribution by temperature T.

    Parameters
    ----------
    alpha : float
        Weight for the distillation term. Range [0, 1].
        0 = hard labels only; 1 = soft targets only.
    temperature : float
        Softening temperature T (> 0).  Higher T → flatter distribution
        → richer dark knowledge signal.

    Examples
    --------
    >>> loss_fn = DistillationLoss(alpha=0.6, temperature=4.0)
    >>> loss = loss_fn(student_logits, teacher_logits, labels)
    """

    def __init__(self, alpha: float = 0.5, temperature: float = 4.0) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.alpha = alpha
        self.T = temperature
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        student_logits : Tensor of shape (B, C)
        teacher_logits : Tensor of shape (B, C)  — will be detached internally
        labels         : Tensor of shape (B,)    — integer class indices

        Returns
        -------
        Scalar combined loss
        """
        s_log_soft = F.log_softmax(student_logits / self.T, dim=-1)
        t_soft     = F.softmax(teacher_logits.detach() / self.T, dim=-1)
        loss_kd    = self.kl(s_log_soft, t_soft) * (self.T ** 2)
        loss_ce    = self.ce(student_logits, labels)
        return self.alpha * loss_kd + (1.0 - self.alpha) * loss_ce

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, temperature={self.T}"


# ── 2. Feature-Based Distillation Loss (FitNets) ──────────────────────────────

class FeatureDistillationLoss(nn.Module):
    """
    FitNets-style intermediate feature alignment (Romero et al., 2015).

    A learnable 1×1 Conv adapter projects the student's feature map into
    the teacher's channel space, then MSE is computed.

    Parameters
    ----------
    student_channels : int  — channels in the student feature map
    teacher_channels : int  — channels in the teacher feature map

    Examples
    --------
    >>> loss_fn = FeatureDistillationLoss(student_channels=128, teacher_channels=512)
    >>> loss = loss_fn(student_feat, teacher_feat)
    """

    def __init__(self, student_channels: int, teacher_channels: int) -> None:
        super().__init__()
        self.adapter = nn.Conv2d(
            student_channels, teacher_channels, kernel_size=1, bias=False
        )
        self.mse = nn.MSELoss()

    def forward(
        self, student_feat: torch.Tensor, teacher_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        student_feat : (B, C_s, H, W) student feature map
        teacher_feat : (B, C_t, H, W) teacher feature map — detached internally

        Returns
        -------
        Scalar MSE loss after projection
        """
        projected = self.adapter(student_feat)
        return self.mse(projected, teacher_feat.detach())


# ── 3. Attention Transfer Loss ────────────────────────────────────────────────

class AttentionTransferLoss(nn.Module):
    """
    Attention transfer from teacher to student (Zagoruyko & Komodakis, 2017).

    Computes L2-normalised spatial attention maps (sum of squared activations
    across channels), then penalises the mean squared difference between
    teacher and student attention maps across paired layer groups.

    Parameters
    ----------
    layer_pairs : list of (student_layer_name, teacher_layer_name) — informational only;
                  actual tensors are passed at forward time.

    Examples
    --------
    >>> loss_fn = AttentionTransferLoss()
    >>> loss = loss_fn([s_feat1, s_feat2], [t_feat1, t_feat2])
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _attention_map(feature: torch.Tensor) -> torch.Tensor:
        """
        Compute L2-normalised spatial attention map.

        Parameters
        ----------
        feature : (B, C, H, W)

        Returns
        -------
        (B, H*W) normalised attention vector
        """
        attn = feature.pow(2).sum(dim=1, keepdim=True)          # (B, 1, H, W)
        return F.normalize(attn.view(attn.size(0), -1), p=2, dim=1)

    def forward(
        self,
        student_features: list[torch.Tensor],
        teacher_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        student_features : list of (B, C_s, H, W) feature tensors
        teacher_features : list of (B, C_t, H, W) feature tensors (same spatial dims)

        Returns
        -------
        Scalar average attention-transfer loss
        """
        if len(student_features) != len(teacher_features):
            raise ValueError("student and teacher feature lists must have equal length")
        if not student_features:
            raise ValueError("feature lists must not be empty")

        total = sum(
            (self._attention_map(s) - self._attention_map(t.detach())).pow(2).mean()
            for s, t in zip(student_features, teacher_features)
        )
        return total / len(student_features)


# ── 4. Relational KD Loss ─────────────────────────────────────────────────────

class RKDLoss(nn.Module):
    """
    Relational Knowledge Distillation (Park et al., CVPR 2019).

    Distils pairwise structural relationships between samples instead of
    individual activations — architecture-agnostic and effective across
    large capacity gaps.

    Two terms are computed:
      * Distance loss — pairwise L2 distances normalised by their mean
      * Angle loss    — cosine of angles in every triplet (i, j, k)

    Parameters
    ----------
    lambda_d : float   weight for distance loss term
    lambda_a : float   weight for angle loss term
    eps      : float   numerical stability constant

    Examples
    --------
    >>> loss_fn = RKDLoss(lambda_d=1.0, lambda_a=2.0)
    >>> losses = loss_fn(student_embed, teacher_embed)
    >>> losses["total"].backward()
    """

    def __init__(
        self,
        lambda_d: float = 1.0,
        lambda_a: float = 2.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.lambda_d = lambda_d
        self.lambda_a = lambda_a
        self.eps = eps

    def _pairwise_l2(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D) → (B, B) pairwise L2 distance matrix."""
        dot = x @ x.T
        sq  = dot.diag().unsqueeze(1) + dot.diag().unsqueeze(0) - 2.0 * dot
        return sq.clamp(min=0.0).sqrt()

    def distance_loss(
        self, s_embed: torch.Tensor, t_embed: torch.Tensor
    ) -> torch.Tensor:
        t_dist = self._pairwise_l2(t_embed.detach())
        s_dist = self._pairwise_l2(s_embed)
        t_norm = t_dist / (t_dist.mean().clamp(min=self.eps))
        s_norm = s_dist / (s_dist.mean().clamp(min=self.eps))
        return F.huber_loss(s_norm, t_norm, reduction="mean", delta=1.0)

    def angle_loss(
        self, s_embed: torch.Tensor, t_embed: torch.Tensor
    ) -> torch.Tensor:
        def _angles(e: torch.Tensor) -> torch.Tensor:
            td = F.normalize(e.unsqueeze(0) - e.unsqueeze(1), p=2, dim=-1)  # (B,B,D)
            return torch.bmm(td, td.permute(0, 2, 1))                        # (B,B,B)

        return F.huber_loss(
            _angles(s_embed), _angles(t_embed.detach()), reduction="mean", delta=1.0
        )

    def forward(
        self,
        student_embed: torch.Tensor,
        teacher_embed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        student_embed : (B, D_s) student embedding vectors
        teacher_embed : (B, D_t) teacher embedding vectors

        Returns
        -------
        Dict with keys 'total', 'distance', 'angle'
        """
        d = self.distance_loss(student_embed, teacher_embed)
        a = self.angle_loss(student_embed, teacher_embed)
        return {
            "total":    self.lambda_d * d + self.lambda_a * a,
            "distance": d,
            "angle":    a,
        }


# ── 5. Self-Distillation Loss (multi-exit) ────────────────────────────────────

class SelfDistillationLoss(nn.Module):
    """
    Self-distillation loss for multi-exit networks.

    The deepest exit acts as the teacher, providing soft targets to
    all shallower exits. No separate teacher model required.

    Parameters
    ----------
    temperature : float   Softening temperature for the deepest-exit soft targets
    exit_weights : tuple  Per-exit loss weights — last entry is for the final exit.
                          Must have length == number of exits.

    Examples
    --------
    >>> loss_fn = SelfDistillationLoss(temperature=3.0, exit_weights=(0.3, 0.4, 1.0))
    >>> loss, details = loss_fn([l1, l2, l3], labels)
    """

    def __init__(
        self,
        temperature: float = 3.0,
        exit_weights: tuple[float, ...] = (0.3, 0.4, 1.0),
    ) -> None:
        super().__init__()
        self.T       = temperature
        self.weights = exit_weights
        self.kl      = nn.KLDivLoss(reduction="batchmean")
        self.ce      = nn.CrossEntropyLoss()

    def forward(
        self,
        exit_logits: list[torch.Tensor],
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Parameters
        ----------
        exit_logits : list of (B, C) logit tensors, ordered shallow → deep
        labels      : (B,) integer class indices

        Returns
        -------
        (total_loss, {per-exit loss values})
        """
        if len(exit_logits) != len(self.weights):
            raise ValueError(
                f"Got {len(exit_logits)} exit logits but "
                f"{len(self.weights)} weights"
            )

        # Deepest exit is the teacher — hard-label CE
        final_logits = exit_logits[-1]
        final_loss   = self.ce(final_logits, labels)
        t_soft       = F.softmax(final_logits.detach() / self.T, dim=-1)

        total     = self.weights[-1] * final_loss
        details   = {f"exit_{len(exit_logits)}": final_loss.item()}

        for i, (logits, w) in enumerate(
            zip(exit_logits[:-1], self.weights[:-1]), start=1
        ):
            kl  = self.kl(
                F.log_softmax(logits / self.T, dim=-1), t_soft
            ) * self.T ** 2
            ce  = self.ce(logits, labels)
            loss = 0.5 * kl + 0.5 * ce
            total += w * loss
            details[f"exit_{i}"] = loss.item()

        return total, details
