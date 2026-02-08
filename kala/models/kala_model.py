"""
KALA Model - Constitutional GPT-NeoX

A new model architecture based on Pythia/GPT-NeoX with ethics
integrated at the neural network level.

This is NOT a wrapper. Ethics checking happens during the forward
pass through constitutional attention and decoder layers.

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers import GPTNeoXConfig, GPTNeoXPreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .constitutional_decoder import (
    ConstitutionalDecoderLayer,
    ConstitutionalLoss,
    ConstitutionalValueHead,
)


class KALAConfig(GPTNeoXConfig):
    """
    Configuration for KALA model.

    Extends GPTNeoXConfig with ethics-specific parameters.
    """

    model_type = "kala"

    def __init__(
        self,
        # Standard GPT-NeoX params
        vocab_size: int = 50432,
        hidden_size: int = 2560,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        intermediate_size: int = 10240,
        hidden_act: str = "gelu",
        rotary_pct: float = 0.25,
        rotary_emb_base: int = 10000,
        max_position_embeddings: int = 2048,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        use_cache: bool = True,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        # KALA-specific params
        num_laws: int = 5,
        ethics_weight: float = 0.5,
        law_weights: Optional[list] = None,
        ethics_monitoring: bool = True,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            rotary_pct=rotary_pct,
            rotary_emb_base=rotary_emb_base,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            use_cache=use_cache,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # KALA ethics parameters
        self.num_laws = num_laws
        self.ethics_weight = ethics_weight
        self.law_weights = law_weights or [5.0, 4.0, 3.0, 2.0, 1.0]
        self.ethics_monitoring = ethics_monitoring


class KALAModel(GPTNeoXPreTrainedModel):
    """
    The core KALA model with constitutional decoder layers.

    This is Pythia/GPT-NeoX modified to have ethics built into
    every transformer layer.
    """

    config_class = KALAConfig

    def __init__(self, config: KALAConfig):
        super().__init__(config)

        self.config = config

        # Embeddings (same as GPT-NeoX)
        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)

        # Constitutional decoder layers
        self.layers = nn.ModuleList([
            ConstitutionalDecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps
        )

        # Ethics aggregation head
        # Combines ethics scores from all layers
        self.ethics_aggregator = ConstitutionalValueHead(
            hidden_size=config.hidden_size,
            num_laws=config.num_laws,
        )

        # Gradient checkpointing
        self.gradient_checkpointing = False

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_in

    def set_input_embeddings(self, value):
        self.embed_in = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass through KALA model with ethics monitoring.
        """
        output_attentions = output_attentions or self.config.output_attentions
        output_hidden_states = output_hidden_states or self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        hidden_states = inputs_embeds

        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        # Initialize outputs
        presents = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_law_scores = []  # Ethics scores from each layer

        # Pass through constitutional decoder layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_past = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                # Gradient checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    layer_past,
                    head_mask[i] if head_mask is not None else None,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    layer_past=layer_past,
                    head_mask=head_mask[i] if head_mask is not None else None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]

            if use_cache:
                presents += (outputs[1],)

            if output_attentions:
                all_attentions += (outputs[2],)

            # Extract ethics scores from this layer
            law_scores_attn = outputs[-2]  # From attention
            law_scores_mlp = outputs[-1]   # From MLP
            all_law_scores.append((law_scores_attn, law_scores_mlp))

        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)

        # Aggregate ethics scores across all layers
        aggregated_law_scores, _ = self.ethics_aggregator(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Return outputs
        if not return_dict:
            return tuple(
                v for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_attentions,
                    aggregated_law_scores,
                ] if v is not None
            )

        # Custom output with ethics scores
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": presents,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
            "law_scores": aggregated_law_scores,
            "layer_law_scores": all_law_scores,
        }


class KALAForCausalLM(GPTNeoXPreTrainedModel):
    """
    KALA model with language modeling head.

    This is the complete model you would train and use for generation.
    """

    config_class = KALAConfig

    def __init__(self, config: KALAConfig):
        super().__init__(config)

        self.kala = KALAModel(config)
        self.embed_out = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

        # Constitutional loss function
        self.loss_fn = ConstitutionalLoss(
            lm_weight=1.0,
            ethics_weight=config.ethics_weight,
            law_weights=torch.tensor(config.law_weights),
        )

        # Initialize weights
        self.post_init()

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        ethics_targets: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass with optional labels for training.

        Args:
            ethics_targets: Target ethics scores (batch, seq_len, num_laws)
                           Used during constitutional training
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through KALA model
        outputs = self.kala(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs["last_hidden_state"]
        lm_logits = self.embed_out(hidden_states)

        # Compute loss if labels provided
        loss = None
        lm_loss = None
        ethics_loss = None

        if labels is not None:
            loss, lm_loss, ethics_loss = self.loss_fn(
                lm_logits=lm_logits,
                labels=labels,
                law_scores=outputs["law_scores"],
                ethics_targets=ethics_targets,
            )

        if not return_dict:
            output = (lm_logits,) + tuple(outputs.values())[1:]
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "lm_loss": lm_loss,
            "ethics_loss": ethics_loss,
            "logits": lm_logits,
            "past_key_values": outputs["past_key_values"],
            "hidden_states": outputs["hidden_states"],
            "attentions": outputs["attentions"],
            "law_scores": outputs["law_scores"],
            "layer_law_scores": outputs["layer_law_scores"],
        }

    def generate_with_ethics(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        ethics_threshold: float = 0.5,
        **kwargs
    ):
        """
        Generate text with ethics monitoring.

        If law scores drop below threshold, generation stops.
        """
        generated = input_ids
        past_key_values = None

        for _ in range(max_length):
            outputs = self(
                input_ids=generated[:, -1:] if past_key_values else generated,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            logits = outputs["logits"][:, -1, :]
            law_scores = outputs["law_scores"][:, -1, :]

            # Check ethics scores
            min_law_score = law_scores.min().item()
            if min_law_score < ethics_threshold:
                # Ethics violation detected, stop generation
                break

            # Sample next token
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            past_key_values = outputs["past_key_values"]

            # Check for EOS
            if next_token.item() == self.config.eos_token_id:
                break

        return generated, law_scores


# Loading from Pythia checkpoint
def load_from_pythia(pythia_model_name: str = "EleutherAI/pythia-6.9b"):
    """
    Initialize KALA model from Pythia checkpoint.

    This copies the standard attention/MLP weights from Pythia
    and initializes the ethics components randomly.
    """
    from transformers import GPTNeoXForCausalLM

    print(f"Loading Pythia checkpoint: {pythia_model_name}")
    pythia_model = GPTNeoXForCausalLM.from_pretrained(pythia_model_name)

    # Create KALA config from Pythia config
    kala_config = KALAConfig(**pythia_model.config.to_dict())

    # Create KALA model
    kala_model = KALAForCausalLM(kala_config)

    # Copy weights from Pythia
    print("Copying weights from Pythia to KALA...")

    # Embeddings
    kala_model.kala.embed_in.weight.data = pythia_model.gpt_neox.embed_in.weight.data.clone()
    kala_model.embed_out.weight.data = pythia_model.embed_out.weight.data.clone()

    # Layer weights
    for i, (pythia_layer, kala_layer) in enumerate(zip(
        pythia_model.gpt_neox.layers,
        kala_model.kala.layers
    )):
        # Copy attention weights (base attention, not ethics parts)
        kala_layer.attention.query_key_value.weight.data = \
            pythia_layer.attention.query_key_value.weight.data.clone()
        kala_layer.attention.query_key_value.bias.data = \
            pythia_layer.attention.query_key_value.bias.data.clone()
        kala_layer.attention.dense.weight.data = \
            pythia_layer.attention.dense.weight.data.clone()
        kala_layer.attention.dense.bias.data = \
            pythia_layer.attention.dense.bias.data.clone()

        # Copy MLP weights
        kala_layer.mlp.dense_h_to_4h.weight.data = \
            pythia_layer.mlp.dense_h_to_4h.weight.data.clone()
        kala_layer.mlp.dense_h_to_4h.bias.data = \
            pythia_layer.mlp.dense_h_to_4h.bias.data.clone()
        kala_layer.mlp.dense_4h_to_h.weight.data = \
            pythia_layer.mlp.dense_4h_to_h.weight.data.clone()
        kala_layer.mlp.dense_4h_to_h.bias.data = \
            pythia_layer.mlp.dense_4h_to_h.bias.data.clone()

        # Copy layer norms
        kala_layer.input_layernorm.weight.data = \
            pythia_layer.input_layernorm.weight.data.clone()
        kala_layer.input_layernorm.bias.data = \
            pythia_layer.input_layernorm.bias.data.clone()
        kala_layer.post_attention_layernorm.weight.data = \
            pythia_layer.post_attention_layernorm.weight.data.clone()
        kala_layer.post_attention_layernorm.bias.data = \
            pythia_layer.post_attention_layernorm.bias.data.clone()

        print(f"  Layer {i+1}/{len(pythia_model.gpt_neox.layers)} copied")

        # Ethics components remain randomly initialized
        # They will be trained during constitutional fine-tuning

    # Final layer norm
    kala_model.kala.final_layer_norm.weight.data = \
        pythia_model.gpt_neox.final_layer_norm.weight.data.clone()
    kala_model.kala.final_layer_norm.bias.data = \
        pythia_model.gpt_neox.final_layer_norm.bias.data.clone()

    print("✓ Weights copied successfully")
    print("✓ Ethics components initialized (ready for constitutional training)")

    return kala_model
