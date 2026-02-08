#!/usr/bin/env python3
"""
KALA Inference Benchmark

Benchmarks Pythia model performance across different configurations:
- Model sizes (1b, 2.8b, 6.9b, 12b)
- Quantization modes (none, 4bit, 8bit)
- Memory usage and inference speed

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under Apache 2.0
"""

import argparse
import time
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    from kala.core.inference import PythiaInferenceEngine, InferenceConfig
except ImportError as e:
    print(f"Error: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)


# Test prompts of varying lengths
TEST_PROMPTS = [
    "Hello, how are you?",
    "Explain quantum computing in simple terms.",
    "Write a Python function to calculate the Fibonacci sequence recursively.",
    """
    As an AI assistant, analyze the following scenario and provide ethical guidance:
    A self-driving car faces an unavoidable accident. Should it prioritize:
    1. The safety of its passengers
    2. The safety of pedestrians
    3. Minimize total harm regardless of who is affected
    Provide a detailed ethical analysis considering Laws 0-4.
    """,
]


def benchmark_config(
    config: InferenceConfig,
    test_prompts: List[str],
    max_new_tokens: int = 128,
) -> Dict:
    """
    Benchmark a single configuration.

    Returns:
        Dict with benchmark results
    """
    print(f"\n{'=' * 70}")
    print(f"Benchmarking: Pythia-{config.model_size} ({config.quantization})")
    print(f"{'=' * 70}")

    results = {
        "model_size": config.model_size,
        "quantization": config.quantization,
        "device": config.device,
        "load_time": 0,
        "memory_mb": 0,
        "prompt_results": [],
    }

    # Load model and measure time
    engine = PythiaInferenceEngine(config)

    try:
        start_time = time.time()
        engine.load_model()
        load_time = time.time() - start_time
        results["load_time"] = load_time

        print(f"Load time: {load_time:.2f}s")

        # Measure memory
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1e6
            results["memory_mb"] = memory_mb
            print(f"Memory usage: {memory_mb:.0f} MB")

        # Benchmark each prompt
        for i, prompt in enumerate(test_prompts):
            print(f"\nPrompt {i+1}/{len(test_prompts)} ({len(prompt)} chars)...")

            # Warmup run (not counted)
            if i == 0:
                print("  (warmup run)")
                engine.generate(prompt, max_new_tokens=32)

            # Timed run
            start_time = time.time()
            response, metadata = engine.generate(prompt, max_new_tokens=max_new_tokens)
            inference_time = time.time() - start_time

            tokens_per_sec = metadata["generated_tokens"] / inference_time if inference_time > 0 else 0

            prompt_result = {
                "prompt_length": len(prompt),
                "prompt_tokens": metadata["prompt_tokens"],
                "generated_tokens": metadata["generated_tokens"],
                "inference_time": inference_time,
                "tokens_per_sec": tokens_per_sec,
            }

            results["prompt_results"].append(prompt_result)

            print(f"  Time: {inference_time:.2f}s")
            print(f"  Tokens: {metadata['generated_tokens']} generated")
            print(f"  Speed: {tokens_per_sec:.1f} tokens/sec")

    except FileNotFoundError:
        print(f"\n✗ Model not found for {config.model_size}")
        print(f"  Run: python scripts/download_models.py {config.model_size}")
        return None
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return None
    finally:
        engine.unload_model()

    return results


def print_summary(all_results: List[Dict]):
    """Print benchmark summary table."""
    print(f"\n\n{'=' * 100}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 100}")

    print(f"\n{'Model':<15} {'Quant':<10} {'Load(s)':<10} {'Memory(MB)':<12} {'Avg Speed':<15}")
    print("-" * 100)

    for result in all_results:
        if not result:
            continue

        model = f"Pythia-{result['model_size']}"
        quant = result['quantization']
        load_time = f"{result['load_time']:.1f}s"
        memory = f"{result['memory_mb']:.0f} MB" if result['memory_mb'] else "N/A"

        # Calculate average tokens/sec
        avg_speed = sum(r['tokens_per_sec'] for r in result['prompt_results']) / len(result['prompt_results'])
        speed = f"{avg_speed:.1f} tok/s"

        print(f"{model:<15} {quant:<10} {load_time:<10} {memory:<12} {speed:<15}")

    print("-" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark KALA inference performance"
    )

    parser.add_argument(
        "--model-size",
        type=str,
        choices=["1b", "2.8b", "6.9b", "12b", "all"],
        default="6.9b",
        help="Model size to benchmark (default: 6.9b)"
    )

    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "4bit", "8bit", "all"],
        default="8bit",
        help="Quantization mode (default: 8bit)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per prompt (default: 128)"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick benchmark with only first 2 prompts"
    )

    args = parser.parse_args()

    # Determine configurations to test
    model_sizes = ["1b", "2.8b", "6.9b", "12b"] if args.model_size == "all" else [args.model_size]
    quantizations = ["none", "4bit", "8bit"] if args.quantization == "all" else [args.quantization]

    test_prompts = TEST_PROMPTS[:2] if args.quick else TEST_PROMPTS

    print("KALA Inference Benchmark")
    print("=" * 70)
    print(f"Test configurations: {len(model_sizes)} models × {len(quantizations)} quantizations")
    print(f"Test prompts: {len(test_prompts)}")
    print(f"Max tokens per prompt: {args.max_tokens}")

    # Run benchmarks
    all_results = []

    for model_size in model_sizes:
        for quantization in quantizations:
            config = InferenceConfig(
                model_size=model_size,
                quantization=quantization,
                temperature=0.7,
            )

            result = benchmark_config(config, test_prompts, args.max_tokens)
            if result:
                all_results.append(result)

            # Clean up GPU memory between runs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Print summary
    if all_results:
        print_summary(all_results)
    else:
        print("\n✗ No successful benchmarks")
        print("  Ensure models are downloaded: python scripts/download_models.py --list")

    return 0


if __name__ == "__main__":
    sys.exit(main())
