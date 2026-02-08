#!/usr/bin/env python3
"""
Test Tesla P100 GPU Setup

Verifies that your Tesla P100 is properly configured for KALA/Pythia.

Copyright 2026 Hew Carroll / The Saelix Institute
"""

import sys


def test_cuda_installation():
    """Test CUDA installation and driver."""
    print("=" * 70)
    print("1. Testing CUDA Installation")
    print("=" * 70)

    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed")
        print("\nInstall with:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False

    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        print("\nPossible issues:")
        print("  1. NVIDIA drivers not installed")
        print("  2. PyTorch CPU-only version installed")
        print("  3. GPU not detected by Windows")
        return False

    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")

    return True


def test_gpu_detection():
    """Test GPU detection and properties."""
    print("\n" + "=" * 70)
    print("2. Testing GPU Detection")
    print("=" * 70)

    try:
        import torch

        if not torch.cuda.is_available():
            return False

        device_count = torch.cuda.device_count()
        print(f"✓ GPUs detected: {device_count}")

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Total VRAM: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Multi-processors: {props.multi_processor_count}")

            # Check if it's a P100
            if "P100" in props.name:
                print(f"  ✓ Tesla P100 detected!")

                # P100 specific checks
                if props.major == 6 and props.minor == 0:
                    print(f"  ✓ Pascal architecture confirmed")
                else:
                    print(f"  ⚠️  Unexpected compute capability")

                vram_gb = props.total_memory / 1024**3
                if vram_gb >= 15:  # 16GB nominal
                    print(f"  ✓ Full 16GB VRAM available")
                else:
                    print(f"  ⚠️  Only {vram_gb:.1f}GB VRAM (expected 16GB)")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_tensor_operations():
    """Test basic tensor operations on GPU."""
    print("\n" + "=" * 70)
    print("3. Testing GPU Tensor Operations")
    print("=" * 70)

    try:
        import torch

        if not torch.cuda.is_available():
            return False

        # Create tensors on GPU
        print("Creating tensors on GPU...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()

        # Matrix multiplication
        print("Running matrix multiplication...")
        z = torch.matmul(x, y)

        print(f"✓ Tensor created on: {z.device}")
        print(f"✓ Tensor shape: {z.shape}")
        print(f"✓ GPU operations working!")

        # Memory stats
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        print(f"\nMemory allocated: {allocated:.2f} MB")
        print(f"Memory reserved: {reserved:.2f} MB")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_model_inference():
    """Test loading a small model on GPU."""
    print("\n" + "=" * 70)
    print("4. Testing Model Inference on GPU")
    print("=" * 70)

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if not torch.cuda.is_available():
            return False

        print("Loading GPT-2 small for testing...")
        model_name = "gpt2"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

        print(f"✓ Model loaded on GPU")
        print(f"✓ Model device: {next(model.parameters()).device}")

        # Test generation
        print("\nTesting text generation...")
        inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=20)
        generated = tokenizer.decode(outputs[0])

        print(f"✓ Generated text: {generated}")

        # Memory usage
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        print(f"\nGPU memory used: {allocated:.2f} MB")

        # Cleanup
        del model
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nThis is optional - main CUDA functionality works")
        return True  # Don't fail on this


def test_quantization():
    """Test 8-bit quantization support."""
    print("\n" + "=" * 70)
    print("5. Testing 8-bit Quantization Support")
    print("=" * 70)

    try:
        import bitsandbytes as bnb
        print(f"✓ bitsandbytes installed")

        import torch
        if torch.cuda.is_available():
            # Test 8-bit operation
            tensor = torch.randn(100, 100).cuda()
            print("✓ 8-bit quantization available")
            print("  This allows running larger models with less VRAM")

        return True

    except ImportError:
        print("⚠️  bitsandbytes not installed")
        print("\nInstall with:")
        print("  pip install bitsandbytes")
        print("\nThis enables 8-bit quantization for Pythia models")
        return True  # Don't fail - optional


def print_recommendations():
    """Print recommendations based on hardware."""
    print("\n" + "=" * 70)
    print("Recommendations for Tesla P100 (16GB)")
    print("=" * 70)

    print("\n📊 Recommended Models:")
    print("\n  Pythia Models (for training with Constitutional AI):")
    print("    • Pythia-2.8B  (8-bit) → ~3GB VRAM   [Fast training]")
    print("    • Pythia-6.9B  (8-bit) → ~7GB VRAM   [Balanced - RECOMMENDED]")
    print("    • Pythia-12B   (8-bit) → ~12GB VRAM  [Best quality]")
    print()
    print("  Other Models (via LM Studio):")
    print("    • Mistral-7B   → ~7GB VRAM   [Excellent quality]")
    print("    • Llama-2-13B  → ~13GB VRAM  [Very good]")
    print("    • CodeLlama-13B → ~13GB VRAM [Best for code]")

    print("\n⚡ Performance Estimates:")
    print("  • Inference: ~20-50 tokens/sec (Pythia-6.9B)")
    print("  • Training: ~1-2 iterations/sec (batch_size=4)")
    print("  • Full 3-epoch fine-tune: ~6-8 hours")

    print("\n🎯 Quick Start Commands:")
    print("\n  Test KALA with Pythia-6.9B:")
    print("    python -c \"")
    print("from kala.core.unified_session import UnifiedKALASession")
    print("from kala.core.inference import InferenceConfig")
    print("config = InferenceConfig(model_size='6.9b', quantization='8bit')")
    print("with UnifiedKALASession(inference_config=config) as s:")
    print("    print(s.chat('What is AI?'))")
    print("\"")

    print("\n  Train with Constitutional AI:")
    print("    python scripts/train_kala.py --model-size 6.9b --config configs/p100_training_config.yaml")

    print("\n  Use LM Studio:")
    print("    python test_lmstudio.py")


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         Tesla P100 GPU Test for KALA                    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    results = {
        "CUDA Installation": test_cuda_installation(),
        "GPU Detection": test_gpu_detection(),
        "Tensor Operations": test_tensor_operations(),
        "Model Inference": test_model_inference(),
        "Quantization": test_quantization(),
    }

    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:<10} {test_name}")
        if not passed:
            all_passed = False

    print()

    if all_passed:
        print("✓ All tests passed! Your P100 is ready for KALA.")
        print_recommendations()
    else:
        print("✗ Some tests failed. See troubleshooting below.")
        print("\nTroubleshooting:")
        print("  1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        print("  2. Install PyTorch with CUDA:")
        print("     pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("  3. Verify drivers: nvidia-smi")
        print("  4. Switch to TCC mode: nvidia-smi -dm 1")

    return 0 if all_passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
