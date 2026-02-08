#!/usr/bin/env python3
"""
KALA Training Script

Main entry point for training KALA models with Constitutional AI.

Usage:
    python scripts/train_kala.py --config configs/lora_training_config.yaml

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kala.training.lora_trainer import KALALoRATrainer, LoRATrainingConfig
from kala.training.ethics_generator import EthicsExampleGenerator
from kala.training.dataset import ConstitutionalDataset


def generate_datasets(output_dir: Path):
    """Generate Constitutional AI training datasets."""
    print("=" * 70)
    print("Generating Constitutional AI Datasets")
    print("=" * 70)

    generator = EthicsExampleGenerator(seed=42)

    # Generate full dataset
    dataset = generator.generate_full_dataset(
        law_0_count=100,
        law_1_count=200,
        law_2_count=150,
        law_3_count=100,
        law_4_count=100,
        conflict_count=50,
        multi_turn_count=50,
    )

    # Save datasets
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset.save(output_dir / "ethics_training.jsonl")
    dataset.export_for_training(
        output_dir / "ethics_training_alpaca.json",
        format="alpaca"
    )

    # Show statistics
    import json
    print("\nDataset Statistics:")
    print(json.dumps(dataset.get_statistics(), indent=2))

    print(f"\n✓ Datasets generated in {output_dir}")


def train(config_path: Path, generate_data: bool = False):
    """Run training."""
    print("=" * 70)
    print("KALA Constitutional AI Training")
    print("=" * 70)

    # Generate datasets if requested
    if generate_data:
        generate_datasets(Path("datasets/constitutional_ai"))

    # Load configuration
    print(f"\nLoading configuration from {config_path}")
    config = LoRATrainingConfig.from_yaml(config_path)

    print(f"\nConfiguration:")
    print(f"  Base model: {config.base_model}")
    print(f"  LoRA rank: {config.lora_r}")
    print(f"  LoRA alpha: {config.lora_alpha}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Batch size: {config.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")

    # Initialize trainer
    print("\n" + "=" * 70)
    print("Initializing Trainer")
    print("=" * 70)

    trainer = KALALoRATrainer(config)

    # Prepare datasets
    print("\n" + "=" * 70)
    print("Preparing Datasets")
    print("=" * 70)

    try:
        dataset_dict = trainer.prepare_datasets(config_path)
    except Exception as e:
        print(f"\n⚠️  Error loading datasets: {e}")
        print("\nNote: Dataset files not found. Generate them first:")
        print("  python scripts/train_kala.py --generate-data")
        return

    # Run training
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    trainer.train(dataset_dict)

    # Save model
    print("\n" + "=" * 70)
    print("Saving Model")
    print("=" * 70)

    trainer.save_model()

    print("\n" + "=" * 70)
    print("✓ Training Complete!")
    print("=" * 70)

    print(f"\nModel saved to: {config.output_dir}")
    print("\nTo use the trained model:")
    print("  python scripts/evaluate_kala.py")


def main():
    parser = argparse.ArgumentParser(
        description="Train KALA with Constitutional AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lora_training_config.yaml"),
        help="Path to training configuration file"
    )

    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate Constitutional AI dataset before training"
    )

    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only generate datasets, don't train"
    )

    args = parser.parse_args()

    if args.data_only:
        generate_datasets(Path("datasets/constitutional_ai"))
        return 0

    train(args.config, generate_data=args.generate_data)

    return 0


if __name__ == "__main__":
    sys.exit(main())
