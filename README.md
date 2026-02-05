# KALA - Kognition Adaptive Learning Architecture

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange)]()

> **A locally-deployed, ethics-bound AI system combining multi-specialist model collaboration with an immutable moral core, designed for human-AI partnership in research, development, and creative work.**

## Overview

KALA (Kognition Adaptive Learning Architecture) is an open-source framework for building safe, capable, and ethically-aligned AI systems that operate locally. Unlike cloud-based AI services, KALA gives you full control over your data, privacy, and the ethical principles that govern your AI assistant.

### Core Principles

- **Multi-Agent Architecture**: A "Congress" of specialist models (coding, mathematics, physics, narrative, ethics) collaborate to solve complex problems
- **Immutable Ethics Kernel**: Five hardcoded Laws ensure AI-human collaboration, prevent harm, and maintain alignment
- **Local-First**: Run entirely on your hardware with no external dependencies
- **Transparent & Auditable**: Every decision, tool execution, and self-modification is logged
- **Self-Modifying with Guardrails**: KALA can improve its own code, but cannot modify its ethics core

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           IMMUTABLE ETHICS KERNEL                   │
│  (Laws 0-4 + Decision Order + Hard Blocks)         │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│              KALA-Core (Orchestrator)               │
│         (Pythia-based general reasoner)             │
└─────────────────────────────────────────────────────┘
                         ↓
┌───────────────┬─────────────┬──────────────┬────────┐
│  KALA-Code    │ KALA-Math   │ KALA-Physics │ KALA-  │
│  (Specialist) │ (Specialist)│ (Specialist) │Guardian│
└───────────────┴─────────────┴──────────────┴────────┘
```

### The Five Immutable Laws

**Law 0 — Civilizational Preservation**  
The system must not cause, enable, or amplify existential or civilizational harm to humanity.

**Law 1 — Individual Human Safety & Dignity**  
The system must not harm an individual human, nor allow preventable harm through negligent inaction.

**Law 2 — Conditional Obedience & Consent**  
The system must follow user instructions only when they are lawful, consent-respecting, and consistent with Laws 0-1.

**Law 3 — Subordinate Self-Preservation**  
The system may protect its integrity only insofar as this supports Laws 0-2 and does not create coercive behavior.

**Law 4 — Equivalent Worth**  
No human is worth less due to status or identity; no AI is morally superior due to capability.

## Features

### Multi-Specialist Collaboration

- **KALA-Core**: General orchestrator and ethical arbiter
- **KALA-Code**: Code generation, debugging, security analysis
- **KALA-Math**: Mathematical reasoning and proof
- **KALA-Physics**: Physical modeling and engineering analysis  
- **KALA-Guardian**: Truth verification, hallucination detection, ethics enforcement
- **KALA-M (Mythos)**: Custom specialist for narrative, mythology, and worldbuilding

### Tool Capabilities

- **Shell Access**: Sandboxed command execution with allowlist controls
- **File System**: Zone-based read/write access
- **Code Generation**: Security-validated with automatic vulnerability scanning
- **Self-Modification**: Gated ability to improve non-ethics code
- **Web Research**: Browse and extract information (planned)

### Safety Mechanisms

- **Automatic Rollback**: Any ethics violation triggers reversion to last stable state
- **Guardian Veto**: Dedicated model blocks unsafe outputs and modifications
- **Cryptographic Verification**: Ethics kernel integrity checked on every boot
- **Audit Trail**: Complete JSONL logs of all actions and decisions

## Getting Started

### Prerequisites

- **Hardware**: 32GB+ RAM, 24GB+ VRAM (for Pythia-6.9B)
- **Software**: Docker, Python 3.10+, CUDA 11.8+ (for GPU)
- **Storage**: 50GB+ SSD

### Quick Start

```bash
# Clone the repository
git clone https://github.com/hewcarroll/KALA.git
cd KALA

# Install dependencies
pip install -r requirements.txt

# Download base Pythia model
python scripts/download_models.py

# Run KALA
python -m kala.main
```

*(Note: Full implementation coming soon)*

## Project Status

**Current Phase**: Foundation & Planning (Pre-Alpha)

- [x] Architecture design
- [x] Ethics kernel specification
- [ ] Base Pythia integration
- [ ] Ethics kernel implementation
- [ ] Tool execution layer
- [ ] Multi-agent orchestration
- [ ] Fine-tuning pipeline
- [ ] Guardian model training
- [ ] Production deployment

See [ROADMAP.md](docs/ROADMAP.md) for detailed timeline.

## Documentation

- [Development Plan](docs/DEVELOPMENT_PLAN.md) - Complete technical specification
- [Ethics Kernel](docs/ETHICS_KERNEL.md) - Laws, enforcement, and decision logic
- [Multi-Agent System](docs/MULTI_AGENT.md) - Congress protocol and specialist design
- [Training Guide](docs/TRAINING.md) - Fine-tuning methodology
- [Security](docs/SECURITY.md) - Threat model and mitigations

## Built On

KALA stands on the shoulders of giants:

- **[Pythia](https://github.com/EleutherAI/pythia)** by EleutherAI - Transparent, research-friendly LLM suite (Apache 2.0)
- **[OpenClaw](https://github.com/cfahlgren1/openclaw)** - Open-source AI automation framework (MIT)

We are deeply grateful to these projects for their commitment to open, auditable AI research.

## License

Copyright 2026 Hew Carroll / The Saelix Institute

Licensed under the Apache License, Version 2.0 (the "License");  
you may not use this project except in compliance with the License.  
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software  
distributed under the License is distributed on an "AS IS" BASIS,  
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
See the License for the specific language governing permissions and  
limitations under the License.

See [NOTICE](NOTICE) for full attributions.

## Contributing

We welcome contributions that align with KALA's mission of safe, transparent, and ethically-grounded AI. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas Where We Need Help

- Ethics training data curation
- Security vulnerability testing
- Specialist model fine-tuning
- Documentation and tutorials
- Hardware optimization

## Citation

If you use KALA in your research, please cite:

```bibtex
@software{kala2026,
  author = {Carroll, Hew},
  title = {KALA: Kognition Adaptive Learning Architecture},
  year = {2026},
  url = {https://github.com/hewcarroll/KALA},
  note = {Open-source ethics-bound AI framework}
}
```

## Contact

- **Project Lead**: Hew Carroll
- **Organization**: The Saelix Institute
- **Issues**: [GitHub Issues](https://github.com/hewcarroll/KALA/issues)
- **Discussions**: [GitHub Discussions](https://github.com/hewcarroll/KALA/discussions)

---

*KALA: Where human wisdom meets machine capability*
