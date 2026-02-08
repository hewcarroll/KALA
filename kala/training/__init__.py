"""
KALA Training Pipeline

Fine-tuning infrastructure for creating ethics-aware, capable AI models.

Includes:
- Constitutional AI dataset generation
- LoRA fine-tuning pipeline
- Ethics-aware validation
- Checkpoint evaluation
- Multi-dataset training

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

__all__ = [
    "dataset",
    "lora_trainer",
    "evaluator",
    "ethics_validator",
]
