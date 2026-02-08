#!/usr/bin/env python3
"""
KALA Deployment Script

Complete deployment workflow for KALA system.

Usage:
    python scripts/deploy_kala.py --mode production

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil


def check_requirements():
    """Check system requirements."""
    print("=" * 70)
    print("Checking Requirements")
    print("=" * 70)

    checks = []

    # Python version
    python_version = sys.version_info
    checks.append((
        "Python 3.10+",
        python_version >= (3, 10),
        f"Found {python_version.major}.{python_version.minor}"
    ))

    # Rust
    try:
        result = subprocess.run(["rustc", "--version"], capture_output=True, text=True)
        checks.append(("Rust compiler", True, result.stdout.strip()))
    except FileNotFoundError:
        checks.append(("Rust compiler", False, "Not found"))

    # Docker (optional)
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        checks.append(("Docker", True, result.stdout.strip()))
    except FileNotFoundError:
        checks.append(("Docker", False, "Not found (optional)"))

    # GPU (optional)
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            checks.append(("CUDA GPU", True, gpu_name))
        else:
            checks.append(("CUDA GPU", False, "Not available (will use CPU)"))
    except ImportError:
        checks.append(("PyTorch", False, "Not installed yet"))

    # Print results
    for name, passed, info in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name:<20} {info}")

    required_failed = [c for c in checks if not c[1] and "optional" not in c[2].lower()]
    if required_failed:
        print(f"\n❌ {len(required_failed)} required check(s) failed")
        return False

    print("\n✓ All required checks passed")
    return True


def install_dependencies():
    """Install Python dependencies."""
    print("\n" + "=" * 70)
    print("Installing Python Dependencies")
    print("=" * 70)

    subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ], check=True)

    print("✓ Python dependencies installed")


def build_ethics_kernel():
    """Build Rust ethics kernel."""
    print("\n" + "=" * 70)
    print("Building Ethics Kernel (Rust)")
    print("=" * 70)

    # Build with cargo
    subprocess.run(
        ["cargo", "build", "--release"],
        cwd="kala-ethics",
        check=True
    )

    # Build Python bindings
    subprocess.run(
        ["maturin", "develop", "--release"],
        cwd="kala-ethics",
        check=True
    )

    print("✓ Ethics kernel built")


def download_models(model_size: str = "6.9b"):
    """Download Pythia models."""
    print("\n" + "=" * 70)
    print(f"Downloading Pythia-{model_size}")
    print("=" * 70)

    subprocess.run([
        sys.executable, "scripts/download_models.py", model_size
    ], check=True)

    print(f"✓ Pythia-{model_size} downloaded")


def generate_datasets():
    """Generate Constitutional AI datasets."""
    print("\n" + "=" * 70)
    print("Generating Constitutional AI Datasets")
    print("=" * 70)

    subprocess.run([
        sys.executable, "scripts/train_kala.py", "--data-only"
    ], check=True)

    print("✓ Datasets generated")


def run_tests():
    """Run test suite."""
    print("\n" + "=" * 70)
    print("Running Test Suite")
    print("=" * 70)

    subprocess.run([
        sys.executable, "-m", "pytest", "tests/", "-v"
    ], check=True)

    print("✓ All tests passed")


def create_production_config():
    """Create production configuration."""
    print("\n" + "=" * 70)
    print("Creating Production Configuration")
    print("=" * 70)

    prod_config = Path("configs/production.yaml")

    if prod_config.exists():
        print(f"  Production config already exists: {prod_config}")
    else:
        # Copy from base config
        shutil.copy("configs/base_config.yaml", prod_config)
        print(f"  ✓ Created production config: {prod_config}")

    print("\n  ⚠️  Remember to customize production.yaml for your deployment")


def verify_installation():
    """Verify KALA installation."""
    print("\n" + "=" * 70)
    print("Verifying Installation")
    print("=" * 70)

    # Test ethics kernel import
    try:
        from kala.ethics.kernel import EthicsKernel
        kernel = EthicsKernel()
        print(f"  ✓ Ethics kernel: {kernel.is_using_rust()}")
    except Exception as e:
        print(f"  ✗ Ethics kernel failed: {e}")
        return False

    # Test inference engine import
    try:
        from kala.core.inference import InferenceConfig
        config = InferenceConfig()
        print(f"  ✓ Inference engine: {config.model_size}")
    except Exception as e:
        print(f"  ✗ Inference engine failed: {e}")
        return False

    # Test tools import
    try:
        from kala.tools import get_registry
        registry = get_registry(reset=True)
        print(f"  ✓ Tools: {len(registry.list_tools())} available")
    except Exception as e:
        print(f"  ✗ Tools failed: {e}")
        return False

    # Test unified session
    try:
        from kala.core.unified_session import UnifiedKALASession
        print(f"  ✓ Unified session ready")
    except Exception as e:
        print(f"  ✗ Unified session failed: {e}")
        return False

    print("\n✓ Installation verified successfully")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Deploy KALA system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["development", "production"],
        default="development",
        help="Deployment mode"
    )

    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip requirement checks"
    )

    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip test suite"
    )

    parser.add_argument(
        "--model-size",
        choices=["2.8b", "6.9b", "12b"],
        default="6.9b",
        help="Pythia model size to download"
    )

    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model download"
    )

    parser.add_argument(
        "--skip-datasets",
        action="store_true",
        help="Skip dataset generation"
    )

    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║              KALA Deployment Script                     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\nMode: {args.mode}")
    print(f"Model: Pythia-{args.model_size}")
    print()

    try:
        # Check requirements
        if not args.skip_checks:
            if not check_requirements():
                print("\n❌ Requirement checks failed")
                return 1

        # Install dependencies
        install_dependencies()

        # Build ethics kernel
        build_ethics_kernel()

        # Download models
        if not args.skip_models:
            download_models(args.model_size)

        # Generate datasets
        if not args.skip_datasets:
            generate_datasets()

        # Run tests
        if not args.skip_tests:
            run_tests()

        # Create production config
        if args.mode == "production":
            create_production_config()

        # Verify installation
        if not verify_installation():
            print("\n❌ Installation verification failed")
            return 1

        # Success
        print("\n" + "=" * 70)
        print("✓ KALA Deployment Complete!")
        print("=" * 70)

        print("\nNext steps:")
        print("  1. Review configuration files in configs/")
        print("  2. Test the system: python -m kala.main")
        print("  3. Train a model: python scripts/train_kala.py")
        print("  4. Read documentation: docs/QUICKSTART.md")

        print("\nTo start using KALA:")
        print("  python")
        print("  >>> from kala.core.unified_session import UnifiedKALASession")
        print("  >>> with UnifiedKALASession() as session:")
        print("  ...     response = session.chat('Hello, KALA!')")

        return 0

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Deployment failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
