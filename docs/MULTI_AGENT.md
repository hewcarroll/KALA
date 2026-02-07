# KALA Multi-Agent Collective

## Overview

KALA uses a multi-agent "Collective" (formerly "Congress") of specialist models collaborating via a round-table protocol. All specialists inherit the immutable ethics kernel from KALA-Core.

## Core Agents

| Agent | Norse Name | Role |
|-------|-----------|------|
| Core | Odin | Central orchestrator, axis mundi of the Collective |
| Memory | Muninn | Active memory retrieval (the raven that fetches) |
| Reasoning | Huginn | Logic, inference, and analytical reasoning |
| Pruner | Nidhogg | Connection pruning during sleep cycles |
| Router | Ratatosk | Query routing between realms/specialists |
| Guardian | Guardian | Truth verification, ethics enforcement, hallucination detection |
| Knowledge | Mimir | Deep knowledge storage (the well) |
| Verifier | Highseat | Real-time web verification agent |

## Domain Specialists (14 total)

| Specialist | Realm | Domain |
|-----------|-------|--------|
| Code | Alfheim | Programming, algorithms, security analysis |
| Math | -- | Mathematical reasoning and proof |
| Physics | -- | Physical modeling and engineering |
| Data | Svartalfheim | Statistics, visualization, data mining |
| CS | -- | Computer science theory, algorithms |
| EE | -- | Electrical engineering, circuits, signals |
| History | -- | Historical analysis across cultures |
| Research | Jotunheim | Academic methodology, literature review |
| Music | -- | Music theory, composition, generation |
| Image | -- | Image generation, manipulation, design |
| Video | -- | Video generation, editing, motion graphics |
| Mythos | Vanaheim | Narrative, mythology, worldbuilding |
| Economics | -- | Financial analysis (with sub-specialists) |
| Interface | Midgard | Human-facing interaction layer |

### Economics Sub-Specialists
- Stocks, ETFs, Crypto, Commodities, Futures

## Collective Protocol

### Round-Table Process

1. **Briefing**: Core restates problem and constraints for all agents
2. **Initial Proposals**: Each specialist responds from its perspective
3. **Debate/Refinement**: Critique rounds (token/round capped)
4. **Consensus/Voting**: Majority, confidence-weighted, or learned consensus mapper
5. **Ethics Pass**: Final synthesis run through immutable ethics kernel
6. **Guardian Review**: Fact-checking, hallucination detection, drift monitoring

### Rules
- Core can veto any Law 0-4 conflict
- High-stakes actions require super-majority + Core + optional human approval
- Each debate round is token-capped to prevent runaway computation

## Recursive Fractal Architecture

Each specialist can become the axis mundi for its own sub-specialists:

```
KALA-Core (Axis Mundi of the Collective)
  +-- Ethics Kernel (shared at every level)
  +-- Economics (Axis Mundi for Financial domain)
  |     +-- Ethics Kernel (inherited)
  |     +-- Stocks, ETFs, Crypto, Commodities, Futures
  +-- Code (could have sub-specialists: Python, JavaScript, Rust...)
  +-- Mythos (could have sub-specialists: Norse, Greek, Comparative...)
  +-- [Each specialist can become its own World Tree]
```

## Guardian Agent

Dedicated model trained as classifier/critic, not content creator:

- **Claim extraction + fact-checking**: Extract verifiable claims, rate as supported/contradicted/unknown
- **Uncertainty quantification**: Calibration/entropy to flag low-confidence areas
- **Hallucination detection**: Self-consistency checks, retrieval agreement, trained classifier
- **Drift monitoring**: Track output distributions against baseline, alert on significant shifts
- **Automatic rollback**: Versioned deployments with snapshots; critical alerts switch to last approved version

## Adaptive "Winning Ticket" Selection

Not all 14+ specialists are needed for every task. The system learns which 2-3 specialists form the "winning lottery ticket" for each task category, reducing compute overhead.

## Yggdrasil Memory Mapping

The Nine Realms map to memory/compute domains:

| Realm | Function | Agent |
|-------|----------|-------|
| Asgard | High-level reasoning, orchestration | Core/Odin |
| Midgard | Human-facing synthesis | Interface |
| Vanaheim | Creative/generative knowledge | Mythos |
| Alfheim | Technical precision | Code |
| Svartalfheim | Deep analysis | Data |
| Jotunheim | Exploration, experimentation | Research/Sandbox |
| Muspelheim | Hot memory (fast access cache) | ActiveCache |
| Niflheim | Cold storage (long-term archive) | Archive |
| Helheim | Dead/pruned memories | Graveyard |

---

*Copyright 2026 Hew Carroll / The Saelix Institute*
