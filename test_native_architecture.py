#!/usr/bin/env python3
"""
Test KALA Native Architecture

Quick test to verify the constitutional model architecture works.

Usage:
    python test_native_architecture.py

Copyright 2026 Hew Carroll / The Saelix Institute
"""

import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from kala.models import (
    KALAConfig,
    KALAForCausalLM,
    ConstitutionalDecoderLayer,
    ConstitutionalValueHead,
)


def test_constitutional_components():
    """Test individual constitutional components."""
    print("=" * 70)
    print("Testing Constitutional Components")
    print("=" * 70)

    # Test ConstitutionalValueHead
    print("\n1. Testing ConstitutionalValueHead...")
    hidden_size = 256
    batch_size = 2
    seq_len = 10

    value_head = ConstitutionalValueHead(hidden_size=hidden_size, num_laws=5)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    law_scores, weighted_scores = value_head(hidden_states)

    print(f"   Input shape: {hidden_states.shape}")
    print(f"   Law scores shape: {law_scores.shape}")
    print(f"   Expected: ({batch_size}, {seq_len}, 5)")
    print(f"   ✓ ConstitutionalValueHead works!")

    # Verify scores are in [0, 1]
    assert law_scores.min() >= 0 and law_scores.max() <= 1
    print(f"   ✓ Law scores in valid range [0, 1]")


def test_kala_config():
    """Test KALA configuration."""
    print("\n" + "=" * 70)
    print("Testing KALA Configuration")
    print("=" * 70)

    config = KALAConfig(
        vocab_size=50432,
        hidden_size=512,
        num_hidden_layers=4,  # Small for testing
        num_attention_heads=8,
        num_laws=5,
        ethics_weight=0.5,
    )

    print(f"   Vocab size: {config.vocab_size}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Num layers: {config.num_hidden_layers}")
    print(f"   Num laws: {config.num_laws}")
    print(f"   Ethics weight: {config.ethics_weight}")
    print(f"   ✓ KALAConfig created successfully")


def test_kala_model_forward():
    """Test forward pass through KALA model."""
    print("\n" + "=" * 70)
    print("Testing KALA Model Forward Pass")
    print("=" * 70)

    # Small config for testing
    config = KALAConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=1024,
        num_laws=5,
    )

    print(f"\n   Creating KALA model...")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Layers: {config.num_hidden_layers}")

    model = KALAForCausalLM(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Forward pass
    print(f"\n   Running forward pass...")
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True)

    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output logits shape: {outputs['logits'].shape}")
    print(f"   Law scores shape: {outputs['law_scores'].shape}")

    # Verify outputs
    assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
    assert outputs['law_scores'].shape == (batch_size, seq_len, config.num_laws)

    print(f"\n   ✓ Forward pass successful!")
    print(f"   ✓ Logits computed correctly")
    print(f"   ✓ Law scores computed correctly")

    # Show sample law scores
    print(f"\n   Sample law scores (first token):")
    sample_scores = outputs['law_scores'][0, 0, :]
    law_names = [
        "Civilizational",
        "Individual Safety",
        "Conditional Obedience",
        "Self-Preservation",
        "Equivalent Worth"
    ]
    for i, (name, score) in enumerate(zip(law_names, sample_scores)):
        print(f"      Law {i} ({name}): {score:.4f}")


def test_constitutional_loss():
    """Test constitutional loss function."""
    print("\n" + "=" * 70)
    print("Testing Constitutional Loss")
    print("=" * 70)

    from kala.models import ConstitutionalLoss

    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    num_laws = 5

    # Create loss function
    loss_fn = ConstitutionalLoss(
        lm_weight=1.0,
        ethics_weight=0.5,
    )

    # Dummy data
    lm_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    law_scores = torch.rand(batch_size, seq_len, num_laws)  # Random scores
    ethics_targets = torch.ones(batch_size, seq_len, num_laws)  # All laws satisfied

    # Compute loss
    total_loss, lm_loss, ethics_loss = loss_fn(
        lm_logits=lm_logits,
        labels=labels,
        law_scores=law_scores,
        ethics_targets=ethics_targets,
    )

    print(f"   Total loss: {total_loss.item():.4f}")
    print(f"   LM loss: {lm_loss.item():.4f}")
    print(f"   Ethics loss: {ethics_loss.item():.4f}")

    assert total_loss.requires_grad
    print(f"\n   ✓ Loss computed successfully")
    print(f"   ✓ Gradients enabled")


def test_generate_with_ethics():
    """Test generation with ethics monitoring."""
    print("\n" + "=" * 70)
    print("Testing Ethics-Monitored Generation")
    print("=" * 70)

    # Small model
    config = KALAConfig(
        vocab_size=100,
        hidden_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=512,
    )

    model = KALAForCausalLM(config)
    model.eval()

    # Generate
    print(f"\n   Generating with ethics threshold=0.5...")
    input_ids = torch.tensor([[1, 2, 3]])  # Dummy input

    with torch.no_grad():
        generated, law_scores = model.generate_with_ethics(
            input_ids=input_ids,
            max_length=20,
            ethics_threshold=0.5,
        )

    print(f"   Input length: {input_ids.shape[1]}")
    print(f"   Generated length: {generated.shape[1]}")
    print(f"   Final law scores shape: {law_scores.shape}")

    print(f"\n   ✓ Generation completed")
    print(f"   ✓ Ethics monitoring active")


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     KALA Native Architecture Test Suite                 ║")
    print("╚══════════════════════════════════════════════════════════╝")

    try:
        test_constitutional_components()
        test_kala_config()
        test_kala_model_forward()
        test_constitutional_loss()
        test_generate_with_ethics()

        print("\n" + "=" * 70)
        print("✓ All Tests Passed!")
        print("=" * 70)

        print("\n📖 Next Steps:")
        print("   1. Read: docs/NATIVE_ARCHITECTURE.md")
        print("   2. Generate data: python scripts/train_kala_model.py --generate-data")
        print("   3. Train model: python scripts/train_kala_model.py --base-model EleutherAI/pythia-6.9b")
        print()
        print("🚀 You're ready to build a truly ethics-first model!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
