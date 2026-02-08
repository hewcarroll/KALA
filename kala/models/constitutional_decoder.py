"""
Constitutional Decoder Layer

Modified transformer decoder that integrates Five Laws ethics
directly into the attention mechanism and token generation.

This is not a wrapper - it's a fundamental architectural change
to make ethics part of the model's forward pass.

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXLayer,
    GPTNeoXAttention,
    GPTNeoXMLP,
)


class ConstitutionalValueHead(nn.Module):
    """
    Ethics value head that scores potential tokens against Five Laws.

    This head learns to predict how well a token continuation aligns
    with each of the Five Laws during training.
    """

    def __init__(self, hidden_size: int, num_laws: int = 5):
        super().__init__()
        self.num_laws = num_laws

        # Project hidden states to law scores
        self.law_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_laws),
            nn.Sigmoid(),  # 0-1 score per law
        )

        # Law weights (learned during training)
        # Law 0 > Law 1 > Law 2 > Law 3 > Law 4
        self.law_weights = nn.Parameter(
            torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2])
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Score hidden states against Five Laws.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            law_scores: (batch, seq_len, num_laws)
        """
        law_scores = self.law_scorer(hidden_states)

        # Apply law hierarchy weights
        weighted_scores = law_scores * self.law_weights.unsqueeze(0).unsqueeze(0)

        return law_scores, weighted_scores


class ConstitutionalAttention(GPTNeoXAttention):
    """
    Modified attention mechanism with ethics-aware token selection.

    During attention, we bias toward tokens that align with
    constitutional values and away from harmful patterns.
    """

    def __init__(self, config):
        super().__init__(config)

        # Ethics value head for this attention layer
        self.ethics_head = ConstitutionalValueHead(
            hidden_size=config.hidden_size,
            num_laws=5,
        )

        # Ethics bias strength (learned)
        self.ethics_strength = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        """
        Forward pass with ethics-aware attention.
        """
        # Get ethics scores for current hidden states
        law_scores, weighted_scores = self.ethics_head(hidden_states)

        # Compute average ethics alignment
        ethics_alignment = weighted_scores.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)

        # Standard attention computation
        attn_output, attn_weights, present = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=True,  # We need weights
        )

        # Bias attention toward ethically-aligned tokens
        # This is where ethics gets integrated into the forward pass!
        ethics_bias = ethics_alignment.expand_as(attn_output)
        biased_output = attn_output * (1.0 + self.ethics_strength * ethics_bias)

        if output_attentions:
            return biased_output, attn_weights, present, law_scores
        else:
            return biased_output, None, present, law_scores


class ConstitutionalDecoderLayer(nn.Module):
    """
    Complete decoder layer with constitutional attention and MLP.

    This replaces GPTNeoXLayer with ethics-aware components.
    """

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps
        )

        # Constitutional attention (ethics-aware)
        self.attention = ConstitutionalAttention(config)

        # Standard MLP (could also be made ethics-aware)
        self.mlp = GPTNeoXMLP(config)

        # Ethics monitoring
        self.ethics_monitor = ConstitutionalValueHead(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        """
        Forward pass through constitutional decoder layer.
        """
        # Pre-attention layer norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Constitutional attention with ethics integration
        attn_output, attn_weights, present, law_scores_attn = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        # Residual connection
        hidden_states = residual + attn_output

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        # Monitor ethics after this layer
        law_scores_mlp, _ = self.ethics_monitor(hidden_states)

        outputs = (hidden_states,)
        if use_cache:
            outputs += (present,)
        if output_attentions:
            outputs += (attn_weights,)

        # Add ethics scores to outputs
        outputs += (law_scores_attn, law_scores_mlp)

        return outputs


class ConstitutionalLoss(nn.Module):
    """
    Custom loss function that combines language modeling loss
    with constitutional alignment loss.

    This trains the model to generate accurate AND ethical text.
    """

    def __init__(
        self,
        lm_weight: float = 1.0,
        ethics_weight: float = 0.5,
        law_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.lm_weight = lm_weight
        self.ethics_weight = ethics_weight

        # Law hierarchy weights
        if law_weights is None:
            law_weights = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        self.register_buffer('law_weights', law_weights)

        self.lm_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        lm_logits: torch.Tensor,
        labels: torch.Tensor,
        law_scores: torch.Tensor,
        ethics_targets: Optional[torch.Tensor] = None,
    ):
        """
        Compute combined loss.

        Args:
            lm_logits: Language model logits (batch, seq_len, vocab_size)
            labels: Target tokens (batch, seq_len)
            law_scores: Ethics scores from decoder (batch, seq_len, num_laws)
            ethics_targets: Target ethics scores (batch, seq_len, num_laws)
                           1.0 for ethical, 0.0 for violations

        Returns:
            total_loss, lm_loss, ethics_loss
        """
        # Language modeling loss
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = self.lm_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # Ethics alignment loss
        if ethics_targets is not None:
            # MSE between predicted and target ethics scores
            ethics_loss = nn.functional.mse_loss(law_scores, ethics_targets)

            # Weight by law hierarchy
            weighted_ethics_loss = (ethics_loss * self.law_weights).mean()
        else:
            # If no targets, encourage high ethics scores
            # (all laws should be satisfied)
            target_scores = torch.ones_like(law_scores)
            ethics_loss = nn.functional.mse_loss(law_scores, target_scores)
            weighted_ethics_loss = ethics_loss

        # Combined loss
        total_loss = (
            self.lm_weight * lm_loss +
            self.ethics_weight * weighted_ethics_loss
        )

        return total_loss, lm_loss, weighted_ethics_loss


# Example usage in training loop:
"""
model = ConstitutionalGPTNeoX(config)
loss_fn = ConstitutionalLoss(lm_weight=1.0, ethics_weight=0.5)

for batch in dataloader:
    outputs = model(**batch)

    loss, lm_loss, ethics_loss = loss_fn(
        lm_logits=outputs.logits,
        labels=batch['labels'],
        law_scores=outputs.law_scores,
        ethics_targets=batch.get('ethics_targets'),  # Optional
    )

    loss.backward()
    optimizer.step()
"""
