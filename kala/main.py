"""
KALA - Kognition Adaptive Learning Architecture
Main entry point for the KALA system.

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Main entry point for KALA."""
    parser = argparse.ArgumentParser(
        description="KALA - Kognition Adaptive Learning Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m kala.main                    # Start interactive session
  python -m kala.main --config custom.yaml  # Use custom config
  python -m kala.main --download-models     # Download required models
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download required models and exit"
    )

    parser.add_argument(
        "--model-size",
        type=str,
        choices=["6.9b", "12b"],
        default="6.9b",
        help="Pythia model size to use"
    )

    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "4bit", "8bit"],
        default="8bit",
        help="Quantization mode for model loading"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Handle model download
    if args.download_models:
        print("Model download functionality will be implemented in scripts/download_models.py")
        return 0

    # Main KALA startup
    print("=" * 60)
    print("KALA - Kognition Adaptive Learning Architecture")
    print("Copyright 2026 Hew Carroll / The Saelix Institute")
    print("=" * 60)
    print(f"\nConfig: {args.config}")
    print(f"Model: Pythia-{args.model_size}")
    print(f"Quantization: {args.quantization}")
    print("\nStatus: Pre-Alpha - Foundation Phase")
    print("\nCore systems to be initialized:")
    print("  [ ] Ethics Kernel (Rust)")
    print("  [ ] Pythia Inference Engine")
    print("  [ ] Fractal Memory System")
    print("  [ ] Tool Execution Layer")
    print("  [ ] Audit Logging")
    print("\n⚠️  Full functionality coming soon - see docs/ROADMAP.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
