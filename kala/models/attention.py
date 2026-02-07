"""
Fractal attention mechanism with depth, group, and angular weighting.

Attention flows through geometric branches with three weighting factors:
    1. Depth proximity: weight = 1 / (phi ^ |depth_q - depth_k|)
    2. Aettir/Aicmi group similarity: 1.0 if same group, 0.3 otherwise
    3. Angular proximity: cos(angle_diff) for geometric closeness

Combined via multiplication and softmax normalization, this provides
O(N log log N) sub-quadratic complexity when used with fractal clustering
(cf. GraphFractalNet, ICLR 2026).

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under the Apache License, Version 2.0
"""

import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError(
        "PyTorch is required for kala.models.attention. "
        "Install with: pip install torch"
    )

from kala.fractal.geometry import PHI


class FractalAttentionBias(nn.Module):
    """Compute attention bias from fractal geometry features.

    This module takes depth, group, and angle information and produces
    an additive attention bias matrix encoding the three geometric
    weighting factors described in the fractal memory specification.
    """

    def __init__(self, group_match_weight: float = 1.0, group_mismatch_weight: float = 0.3):
        super().__init__()
        self.group_match = group_match_weight
        self.group_mismatch = group_mismatch_weight

    def forward(
        self,
        depths: torch.LongTensor,
        groups: torch.LongTensor,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        """Compute fractal attention bias matrix.

        Args:
            depths: [batch, seq_len] depth of each cell.
            groups: [batch, seq_len] group index of each cell.
            angles: [batch, seq_len] stemline angle in degrees.

        Returns:
            [batch, seq_len, seq_len] additive attention bias.
        """
        # Depth proximity: 1 / (phi ^ |d_q - d_k|)
        depth_q = depths.unsqueeze(-1).float()  # [B, S, 1]
        depth_k = depths.unsqueeze(-2).float()  # [B, 1, S]
        depth_diff = torch.abs(depth_q - depth_k)
        depth_weight = 1.0 / (PHI ** depth_diff)

        # Group similarity: 1.0 if same, 0.3 if different
        group_q = groups.unsqueeze(-1)  # [B, S, 1]
        group_k = groups.unsqueeze(-2)  # [B, 1, S]
        group_match = (group_q == group_k).float()
        group_weight = group_match * self.group_match + (1 - group_match) * self.group_mismatch

        # Angular proximity: cos(angle_diff)
        angle_q = angles.unsqueeze(-1)  # [B, S, 1]
        angle_k = angles.unsqueeze(-2)  # [B, 1, S]
        angle_diff_deg = torch.abs(angle_q - angle_k) % 360.0
        angle_diff_deg = torch.min(angle_diff_deg, 360.0 - angle_diff_deg)
        angle_weight = torch.cos(angle_diff_deg * math.pi / 180.0)

        # Combined bias (multiplicative, then log for additive attention)
        combined = depth_weight * group_weight * angle_weight
        # Clamp to avoid log(0)
        combined = torch.clamp(combined, min=1e-8)
        bias = torch.log(combined)

        return bias


class FractalAttentionLayer(nn.Module):
    """Transformer attention layer with fractal geometry bias.

    Wraps standard multi-head attention with an additive bias
    derived from fractal cell properties (depth, group, angle).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
        group_match_weight: float = 1.0,
        group_mismatch_weight: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.fractal_bias = FractalAttentionBias(group_match_weight, group_mismatch_weight)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

    def forward(
        self,
        x: torch.Tensor,
        depths: Optional[torch.LongTensor] = None,
        groups: Optional[torch.LongTensor] = None,
        angles: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional fractal geometry bias.

        Args:
            x: [batch, seq_len, d_model] input embeddings.
            depths: [batch, seq_len] depth indices (optional).
            groups: [batch, seq_len] group indices (optional).
            angles: [batch, seq_len] angles in degrees (optional).
            key_padding_mask: [batch, seq_len] True for padded positions.

        Returns:
            [batch, seq_len, d_model] output with residual connection.
        """
        B, S, D = x.shape
        residual = x

        # Project Q, K, V
        Q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Standard scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # Add fractal geometry bias if available
        if depths is not None and groups is not None and angles is not None:
            geo_bias = self.fractal_bias(depths, groups, angles)  # [B, S, S]
            attn_scores = attn_scores + geo_bias.unsqueeze(1)  # broadcast over heads

        # Apply padding mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute output
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        output = self.out_proj(attn_output)

        # Residual + LayerNorm
        return self.norm(residual + output)
