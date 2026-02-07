# KALA Development Plan

## Technical Specification

KALA (Kognition Adaptive Learning Architecture) combines a Pythia-based language model with an immutable ethics kernel, multi-agent collaboration, and a novel fractal memory system.

## Architecture Stack

```
IMMUTABLE ETHICS KERNEL (Laws 0-4 + Decision Order + Hard Blocks)
        |
KALA REASONING ENGINE (Pythia-6.9B/12B, context window management)
        |
ETHICS COMPLIANCE VALIDATOR (Pre-action: Law 0→1→2→3→4)
        |
TOOL EXECUTION LAYER (OpenClaw-style: shell, filesystem, code, self-mod)
        |
MEMORY & PERSISTENCE (Fractal QR-code + Vector DB + Episodic JSONL)
```

## Base Model

- **Model**: Pythia-6.9B (primary) / Pythia-12B (advanced)
- **Provider**: EleutherAI (Apache 2.0)
- **Quantization**: 8-bit (bitsandbytes) for local deployment, 4-bit optional
- **Fine-tuning**: LoRA (rank 16-32, alpha 32, dropout 0.05)
- **Target modules**: query_key_value, dense, embed_in, embed_out

## Fractal Memory Subsystem

The core innovation is a fractal QR-code neural memory using a 44-character Ogham-Futhark encoding:

### Implementation Modules

| Module | Path | Phase | Description |
|--------|------|-------|-------------|
| Alphabet | `kala/fractal/alphabet.py` | 1 | 44-symbol 6-bit encoding (24 Futhark + 20 Ogham) |
| Geometry | `kala/fractal/geometry.py` | 1 | Golden Ratio primitives (phi, golden angle, scaling) |
| Tree | `kala/fractal/tree.py` | 2 | FractalCell dataclass, build/walk APIs |
| QPB Bias | `kala/fractal/qpb_bias.py` | 3 | Quantum Probability Bias coherence model |
| Error Correction | `kala/fractal/error_correction.py` | 4 | Semantic repair via aettir/aicmi groups |
| Fractal Memory | `kala/models/fractal_memory.py` | 5 | FractalEmbedding + FractalMemoryNetwork |
| Attention | `kala/models/attention.py` | 6 | Fractal attention with depth/group/angle bias |
| QR Encoder | `kala/qr/fractal_qr.py` | 7 | Fractal tree to QR code matrix adapter |
| Visualization | `kala/utils/visualization.py` | 8 | Tree rendering and QR matrix display |

### Three-Layer Coherence Architecture

```
Layer 3 (Top): Fractal QR / Rune-Ogham Memory
  - Coherence signal tilts branch/bind-glyph choices

Layer 2 (Middle): QPB Coherence Structure
  - Quantum-like description with phase, amplitude, Hamiltonian
  - Induces cumulative biases in discrete outcomes

Layer 1 (Bottom): FHN + Quantum Noise
  - FitzHugh-Nagumo dynamics with Brownian noise
  - Generates biophysically-inspired coherence patterns
```

## OpenClaw Integration (6-Stage Pipeline)

1. **Input Standardization**: Normalize user requests
2. **Session Coordination**: Manage multi-turn context
3. **Lane Queue System**: Serial execution with controlled parallelism
4. **Agent Runner**: Model selection and prompt assembly
5. **Agentic Loop**: Tool proposal → execution → validation → continuation
6. **Audit Trail**: JSONL transcript logging

## Self-Modification Gate

Protected modules (ethics_kernel, law_enforcement, decision_order, hard_blocks) are absolutely immutable. All other code modifications go through:

1. Intent classification (capability extension / bug fix / optimization / core modification)
2. Security analysis (vulnerability scanning)
3. Ethics compliance check
4. Human-in-the-loop approval for significant changes
5. Full audit trail

## Configuration

Memory backend is pluggable via config:

```yaml
memory:
  backend: "fractal_runic"  # "baseline" | "fractal_runic"
```

See `configs/base_config.yaml` and `configs/fractal_memory_config.yaml` for full configuration options.

## Hardware Requirements

| Component | Pythia-6.9B | Pythia-12B |
|-----------|-------------|------------|
| RAM | 32GB | 64GB |
| VRAM | 24GB (4-bit quant) | 48GB (4-bit quant) |
| Storage | 50GB SSD | 75GB SSD |
| CPU | 8-core modern | 16-core modern |

## References

- EleutherAI Pythia: https://github.com/EleutherAI/pythia
- OpenClaw: https://github.com/cfahlgren1/openclaw
- QPB: Carroll, H. (2026). Quantum Probability Bias. Saelix Institute.
- FHN: Ghose, P. & Pinotsis, D. A. (2025). CSBJ, 30, pp. 12-20.

---

*Copyright 2026 Hew Carroll / The Saelix Institute*
