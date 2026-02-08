#!/usr/bin/env python3
"""
KALA Model Download Script

Downloads Pythia models from HuggingFace and prepares them for local deployment.
Supports both Pythia-6.9B and Pythia-12B with optional quantization.

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

import argparse
import sys
from pathlib import Path
from typing import Literal

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: Required packages not installed.")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)


# Model configurations
PYTHIA_MODELS = {
    "6.9b": "EleutherAI/pythia-6.9b",
    "12b": "EleutherAI/pythia-12b",
    "2.8b": "EleutherAI/pythia-2.8b",  # For testing on lower-end hardware
    "1b": "EleutherAI/pythia-1b",      # For KALA-Lite
}


def download_model(
    model_size: str,
    output_dir: Path,
    include_deduped: bool = False,
    tokenizer_only: bool = False,
) -> bool:
    """
    Download a Pythia model from HuggingFace.

    Args:
        model_size: Model size (6.9b, 12b, 2.8b, 1b)
        output_dir: Directory to save the model
        include_deduped: Download deduplicated version
        tokenizer_only: Only download tokenizer

    Returns:
        True if successful, False otherwise
    """
    if model_size not in PYTHIA_MODELS:
        print(f"Error: Invalid model size '{model_size}'")
        print(f"Available: {', '.join(PYTHIA_MODELS.keys())}")
        return False

    model_name = PYTHIA_MODELS[model_size]
    if include_deduped:
        model_name = model_name.replace("pythia", "pythia-deduped")

    model_dir = output_dir / model_size
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Downloading Pythia-{model_size}")
    print(f"Model: {model_name}")
    print(f"Target: {model_dir}")
    print("=" * 70)

    try:
        # Download tokenizer
        print("\n[1/2] Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_dir)
        print(f"✓ Tokenizer saved to {model_dir}")

        if tokenizer_only:
            print("\nTokenizer-only mode: Skipping model weights")
            return True

        # Download model weights
        print("\n[2/2] Downloading model weights...")
        print("⚠️  This may take a while depending on your connection")
        print(f"   Pythia-{model_size} size: ~{get_model_size(model_size)}")

        # Use snapshot_download for full model download
        snapshot_download(
            repo_id=model_name,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        print(f"✓ Model weights saved to {model_dir}")

        # Verify the download
        print("\n[3/3] Verifying download...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print(f"✓ Tokenizer loaded: {len(tokenizer)} tokens")

        print("\n" + "=" * 70)
        print("✓ Download complete!")
        print("=" * 70)
        print(f"\nModel location: {model_dir}")
        print(f"\nTo use this model:")
        print(f"  python -m kala.main --model-size {model_size}")

        return True

    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        return False


def get_model_size(model_size: str) -> str:
    """Get approximate download size for a model."""
    sizes = {
        "1b": "~2 GB",
        "2.8b": "~5.5 GB",
        "6.9b": "~13 GB",
        "12b": "~23 GB",
    }
    return sizes.get(model_size, "Unknown")


def list_models():
    """List available Pythia models."""
    print("\nAvailable Pythia models:")
    print("=" * 70)
    for size, repo in PYTHIA_MODELS.items():
        approx_size = get_model_size(size)
        print(f"  {size:6s} - {repo:40s} ({approx_size})")
    print("=" * 70)
    print("\nRecommended for KALA:")
    print("  • 6.9b: Best balance of capability and resource usage")
    print("  • 12b:  Higher quality, requires more VRAM")
    print("  • 2.8b: Testing and development")
    print("  • 1b:   KALA-Lite for resource-constrained environments")


def main():
    parser = argparse.ArgumentParser(
        description="Download Pythia models for KALA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "model_size",
        nargs="?",
        choices=list(PYTHIA_MODELS.keys()),
        help="Model size to download (6.9b, 12b, 2.8b, 1b)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Output directory (default: ./models)"
    )

    parser.add_argument(
        "--deduped",
        action="store_true",
        help="Download deduplicated version"
    )

    parser.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="Only download tokenizer (for testing)"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all recommended models (6.9b and 12b)"
    )

    args = parser.parse_args()

    # List models and exit
    if args.list:
        list_models()
        return 0

    # Download all models
    if args.all:
        print("Downloading all recommended models...")
        success = True
        for size in ["6.9b", "12b"]:
            if not download_model(size, args.output_dir, args.deduped, args.tokenizer_only):
                success = False
        return 0 if success else 1

    # Require model size if not listing or downloading all
    if not args.model_size:
        parser.print_help()
        print("\n")
        list_models()
        return 1

    # Download single model
    success = download_model(
        args.model_size,
        args.output_dir,
        args.deduped,
        args.tokenizer_only
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
