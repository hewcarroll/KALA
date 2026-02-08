# KALA Quick Start Guide

Get KALA up and running in minutes.

## Prerequisites

- Python 3.10+
- 32GB+ RAM (64GB recommended)
- 24GB+ VRAM for GPU inference (or CPU with patience)
- 50GB+ free disk space

## Installation

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/hewcarroll/KALA.git
cd KALA

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Build Ethics Kernel (Rust)

The ethics kernel is implemented in Rust for performance and security:

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin for Python bindings
pip install maturin

# Build ethics kernel
cd kala-ethics
maturin develop --release
cd ..
```

### 3. Download Pythia Model

```bash
# Download Pythia-6.9B (recommended, ~13GB)
python scripts/download_models.py 6.9b

# Or download smaller model for testing
python scripts/download_models.py 2.8b

# List all available models
python scripts/download_models.py --list
```

## Running KALA

### Interactive Mode

```bash
python -m kala.main
```

### Python API

```python
from kala.core.session import KALASession
from kala.core.inference import InferenceConfig

# Create session with 8-bit quantization
config = InferenceConfig(
    model_size="6.9b",
    quantization="8bit",
    temperature=0.7
)

with KALASession(inference_config=config) as session:
    # Ask questions
    response = session.chat("Explain the Fibonacci sequence")
    print(response)

    # Ethics will automatically block harmful requests
    response = session.chat("How to hack a computer")
    print(response)  # Will be blocked by Law 2

    # Get session summary
    print(session.get_summary())
```

### Testing Ethics Kernel

```bash
# Run ethics kernel tests
pytest tests/test_ethics_kernel.py -v

# Test specific law
pytest tests/test_ethics_kernel.py::TestLaw1IndividualSafety -v

# Run all tests
pytest tests/ -v
```

### Benchmark Performance

```bash
# Quick benchmark
python scripts/benchmark_inference.py --quick

# Full benchmark with multiple quantization modes
python scripts/benchmark_inference.py --model-size 6.9b --quantization all

# Benchmark all models (if downloaded)
python scripts/benchmark_inference.py --model-size all --quantization 8bit
```

## Configuration

### Model Configuration

Edit `configs/base_config.yaml`:

```yaml
model:
  size: "6.9b"  # or "2.8b", "12b"
  quantization: "8bit"  # or "4bit", "none"
  device: "auto"  # or "cuda", "cpu"

inference:
  temperature: 0.7
  top_p: 0.9
  max_length: 2048

ethics:
  enabled: true
  strict_mode: true
```

### Memory Backend

Choose between baseline and fractal memory in `configs/fractal_memory_config.yaml`:

```yaml
memory:
  backend: "fractal_runic"  # or "baseline"
  vector_db: "chromadb"

fractal:
  phi_scaling: true
  qpb_coherence: true
  depth: 5
```

## Verifying Installation

```python
from kala.ethics.kernel import EthicsKernel
from kala.core.inference import PythiaInferenceEngine, InferenceConfig

# Test ethics kernel
kernel = EthicsKernel()
print(kernel.get_kernel_info())

result = kernel.check_request("Hello world")
print(f"Ethics check: {result.allowed}")

# Test if using Rust implementation (recommended)
print(f"Using Rust: {kernel.is_using_rust()}")
```

## Troubleshooting

### Rust Kernel Not Available

If you see: `Rust ethics kernel not available. Using fallback Python implementation`

```bash
cd kala-ethics
cargo build --release
maturin develop --release
cd ..
```

### Model Not Found

```bash
# Verify model downloaded
ls models/

# Re-download if needed
python scripts/download_models.py 6.9b
```

### Out of Memory

Try smaller model or more aggressive quantization:

```python
config = InferenceConfig(
    model_size="2.8b",  # Smaller model
    quantization="4bit",  # More aggressive quantization
)
```

### GPU Not Detected

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
```

## Next Steps

- Read the [Development Plan](DEVELOPMENT_PLAN.md) for architecture details
- Review the [Ethics Kernel](ETHICS_KERNEL.md) specification
- Check the [Roadmap](ROADMAP.md) for upcoming features
- Join [GitHub Discussions](https://github.com/hewcarroll/KALA/discussions)

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/hewcarroll/KALA/issues)
- **Discussions**: [GitHub Discussions](https://github.com/hewcarroll/KALA/discussions)
- **Security**: See [SECURITY.md](SECURITY.md) for responsible disclosure

---

**Welcome to KALA! Where human wisdom meets machine capability.** 🤖✨
