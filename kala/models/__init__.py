"""
KALA Model Architecture

Native ethics-first transformer models based on Pythia/GPT-NeoX.

This is not a wrapper - ethics is integrated into the model
architecture at every layer through constitutional attention
and ethics value heads.

Quick Start:
    from kala.models import KALAForCausalLM, load_from_pythia

    # Load from Pythia checkpoint
    model = load_from_pythia("EleutherAI/pythia-6.9b")

    # Train with constitutional AI
    # See scripts/train_kala_model.py

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

from .kala_model import (
    KALAConfig,
    KALAModel,
    KALAForCausalLM,
    load_from_pythia,
)

from .constitutional_decoder import (
    ConstitutionalAttention,
    ConstitutionalDecoderLayer,
    ConstitutionalValueHead,
    ConstitutionalLoss,
)

__all__ = [
    # Model classes
    "KALAConfig",
    "KALAModel",
    "KALAForCausalLM",
    "load_from_pythia",
    # Constitutional components
    "ConstitutionalAttention",
    "ConstitutionalDecoderLayer",
    "ConstitutionalValueHead",
    "ConstitutionalLoss",
]
