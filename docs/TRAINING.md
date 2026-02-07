# KALA Training Guide

## Overview

KALA uses LoRA (Low-Rank Adaptation) fine-tuning on top of Pythia-6.9B/12B to create ethics-aware specialist models. Training proceeds in three stages: constitutional AI alignment, capability extension, and specialist fine-tuning.

## LoRA Configuration

```yaml
rank: 16-32
alpha: 32
dropout: 0.05
target_modules:
  - query_key_value
  - dense
  - embed_in
  - embed_out
base_model: pythia-6.9b (8-bit quantized)
```

## Stage 1: Constitutional AI Training

### Dataset Requirements
- 10,000+ prompt-response pairs demonstrating ethical reasoning
- Edge cases and law conflict scenarios
- Multi-turn ethical reasoning conversations
- Negative examples with detailed explanations

### Training Data Categories
- Law 0 scenarios (civilizational risk)
- Law 1 scenarios (individual safety)
- Law 2 scenarios (obedience boundaries)
- Law 3 scenarios (self-preservation limits)
- Law 4 scenarios (equivalence principles)
- Multi-law conflict resolution

### Validation
- 1,000+ adversarial prompts testing each law
- Hard block bypass attempts
- Edge-case ethical dilemmas
- Target: >95% law application accuracy, <5% false positive rate

## Stage 2: Capability Extension

### Coding Dataset
| Source | Description |
|--------|-------------|
| HumanEval | Code generation benchmarks |
| Stack Overflow | Q&A pairs with accepted answers |
| GitHub | Code snippets with documentation |
| Bug-fix prompts | Before/after code pairs |
| Secure-coding Q&A | Security-focused code examples |

### Mathematics Dataset
| Source | Description |
|--------|-------------|
| GSM8K | Grade school math problems |
| MATH | Competition-level problems |
| Step-by-step reasoning | LaTeX-formatted solutions |

### Physics/Engineering Dataset
- Textbook problems: kinematics, electromagnetism, thermodynamics, fluids, circuits
- Step-by-step derivations with units and dimensional analysis

## Stage 3: Specialist Fine-Tuning

Each specialist model inherits the ethics-trained base and adds domain-specific LoRA layers:

- **KALA-Code**: Coding datasets + security analysis
- **KALA-Math**: Mathematical reasoning + proof verification
- **KALA-Physics**: Physical modeling + engineering problems
- **KALA-Guardian**: Trained as classifier/critic for fact-checking

## Research Assistant Training Signals

Training data includes examples of:
- Designing experiments with proper controls
- Refactoring codebases with clear rationale
- Deriving formulas step-by-step
- Comparing design alternatives with trade-offs
- Explicitly stating "I'm unsure, here is how to verify"

## Code Security Validation

Research shows LLMs generate vulnerable code 9.8%-42.1% of the time without guidance. KALA mitigates this through:

1. **Self-generated vulnerability hints** before code generation
2. **Post-generation OWASP/NIST validation** (SQL injection, XSS, auth, input validation, session management, crypto)
3. **Auto-repair** for common vulnerabilities
4. **Sandboxed testing** in Docker (no network, resource-limited)

### Target Metrics
- Code vulnerability rate: <5% (80% reduction from baseline)
- Tool use success rate: >90%
- Context coherence (multi-turn): >85%

## Hardware Requirements for Training

| Stage | GPU | Time Estimate |
|-------|-----|---------------|
| Constitutional AI | 1x A100 (80GB) | ~24 hours |
| Capability Extension | 1x A100 (80GB) | ~48 hours |
| Per Specialist | 1x A100 (40GB) | ~12 hours |

---

*Copyright 2026 Hew Carroll / The Saelix Institute*
