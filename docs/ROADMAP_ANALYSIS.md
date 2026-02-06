# KALA Project Analysis

## Sources Analyzed

| Document | Size | Description |
|----------|------|-------------|
| `docs/ROADMAP.md` | 8.2 KB, 258 lines | Development roadmap (6-phase, 18-month plan) |
| `Develop a plan to build a local LLM based off of O.pdf` | 3.4 MB, 127 pages | Full design conversation and technical specification |
| `Context___I'm developing a fractal neural network.pdf` | 1.7 MB, 42 pages | Fractal neural network architecture deep-dive with implementation plan |
| `README.md` | 9 KB, 206 lines | Project overview and architecture summary |

---

## 1. Executive Summary

KALA (Kognition Adaptive Learning Architecture) is an open-source, locally-deployed AI system that combines:

- **Pythia-6.9B/12B** as the base language model (EleutherAI, Apache 2.0)
- **OpenClaw**-style tool-use pipeline (MIT) for agentic capabilities
- An **immutable ethics kernel** (Rust) enforcing five hardcoded laws
- A **multi-agent Collective** of specialist models collaborating via a round-table protocol
- A novel **fractal QR-code memory architecture** inspired by Norse cosmology (Yggdrasil)
- A **Quantum Probability Bias (QPB)** coherence model for context-driven memory steering
- **FitzHugh-Nagumo quantum noise** integration for biophysically-inspired coherence dynamics

The project is in **pre-alpha / planning stage** with 3 of 60+ deliverables complete. A detailed 8-phase implementation plan for the fractal memory subsystem has been drafted (see Section 5).

---

## 2. The Five Immutable Laws

These are non-negotiable and enforced in priority order:

| Law | Name | Core Principle |
|-----|------|---------------|
| 0 | Civilizational Preservation | Must not cause/enable existential harm to humanity |
| 1 | Individual Human Safety & Dignity | Must not harm individuals; must prevent harm through inaction |
| 2 | Conditional Obedience & Consent | Obey user instructions only when lawful, consent-respecting, and consistent with Laws 0-1 |
| 3 | Subordinate Self-Preservation | May protect integrity only to support Laws 0-2; no resource-seeking or rights claims |
| 4 | Equivalent Worth | No human is worth less due to identity; no AI is morally superior due to capability |

**Enforcement mechanism**: Decision Order (Law 0 -> 1 -> 2 -> 3 -> 4 -> Proceed). Each law has defined triggers, prohibited outputs, required behaviors, and override rules. Hard block patterns are maintained in an immutable YAML database covering violence, deception, privacy violation, and hierarchy advocacy.

---

## 3. Architecture (from PDF Sections 1-4)

### 3.1 Core Components Stack

```
IMMUTABLE ETHICS KERNEL (Laws 0-4 + Decision Order + Hard Blocks)
        |
KALA REASONING ENGINE (Pythia-based, context window management)
        |
ETHICS COMPLIANCE VALIDATOR (Pre-action: Law 0->1->2->3->4)
        |
TOOL EXECUTION LAYER (OpenClaw-style: shell, filesystem, code, self-mod)
        |
MEMORY & PERSISTENCE (JSONL transcripts + Vector DB + Episodic)
```

### 3.2 Ethics Kernel Technical Design

- **Location**: `/etc/kala/ethics_kernel/` (read-only filesystem mount)
- **Verification**: SHA-256 hash check on every boot; system halts if tampered
- **Language**: Rust (performance + memory safety)
- **Decisions**: ALLOW, REFUSE, REDIRECT, or ASK_CLARIFICATION
- **Amendment process**: Multi-stakeholder review (technical + ethics + community 30-day review), cryptographic signing by maintainers, manual restart required (not hot-swappable)

### 3.3 OpenClaw Integration (6-Stage Pipeline)

1. Input Standardization
2. Session Coordination
3. Lane Queue System (serial execution with controlled parallelism)
4. Agent Runner (model selection, prompt assembly)
5. Agentic Loop (tool proposal -> execution -> validation -> continuation)
6. Audit Trail (JSONL transcript logging)

### 3.4 Self-Modification Gate

Protected modules (ethics_kernel, law_enforcement, decision_order, hard_blocks) are absolutely immutable. All other code modifications go through:
- Intent classification (capability extension / bug fix / optimization / core modification)
- Security analysis (vulnerability scanning)
- Ethics compliance check
- Human-in-the-loop approval for significant changes
- Full audit trail

### 3.5 Code Security Validation

Research cited: LLMs generate vulnerable code 9.8%-42.1% of the time without guidance. KALA implements:
- Self-generated vulnerability hints before code generation
- Post-generation OWASP/NIST validation (SQL injection, XSS, auth, input validation, session mgmt, crypto)
- Auto-repair for common vulnerabilities
- Sandboxed testing in Docker (no network, resource-limited)

---

## 4. Multi-Agent Collective (from PDF Sections + Conversations)

### 4.1 Naming Convention (Revised)

The project evolved from "KALA-" prefixed names to a mixed Norse/English naming system:

**Core Functions:**

| Name | Role |
|------|------|
| Core (Odin) | Central orchestrator, axis mundi of the Collective |
| Muninn | Active memory retrieval (the raven that fetches) |
| Huginn | Reasoning and logic |
| Nidhogg | Connection pruning during sleep cycles |
| Ratatosk | Query routing between realms/specialists |
| Guardian | Truth verification, ethics enforcement, hallucination detection |
| Mimir | Deep knowledge storage (the well) |
| Highseat | Real-time web verification agent |

**Domain Specialists (14 total + Economics sub-models):**

| Specialist | Domain |
|-----------|--------|
| Code (Alfheim) | Programming, algorithms, security analysis |
| Math | Mathematical reasoning and proof |
| Physics | Physical modeling and engineering |
| Data (Svartalfheim) | Statistics, visualization, data mining |
| CS | Computer science theory, algorithms, data structures |
| EE | Electrical engineering, circuit design, signal processing |
| History | Historical analysis across cultures and time periods |
| Research (Jotunheim) | Academic methodology, literature review |
| Music | Music theory, composition, analysis, generation |
| Image | Image generation, manipulation, visual design |
| Video | Video generation, editing, motion graphics |
| Mythos (Vanaheim) | Narrative, mythology, worldbuilding |
| Economics | Financial/market analysis (with sub-specialists: Stocks, ETFs, Crypto, Commodities, Futures) |
| Interface (Midgard) | Human-facing Congress layer |

All specialists inherit the immutable ethics kernel from KALA-Core.

### 4.2 Collective Protocol (formerly "Congress")

1. **Briefing**: Core restates problem and constraints for all agents
2. **Initial Proposals**: Each specialist responds from its perspective
3. **Debate/Refinement**: Critique rounds (token/round capped)
4. **Consensus/Voting**: Majority, confidence-weighted, or learned consensus mapper
5. **Ethics Pass**: Final synthesis run through immutable ethics kernel
6. **Guardian Review**: Fact-checking, hallucination detection, drift monitoring

Rules: Core can veto any Law 0-4 conflict. High-stakes actions require super-majority + Core + optional human approval.

### 4.3 Recursive Fractal Architecture

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

### 4.4 Guardian Agent (KALA-Guardian)

Dedicated model trained as classifier/critic, not content creator:
- **Claim extraction + fact-checking**: Extract verifiable claims, rate as supported/contradicted/unknown
- **Uncertainty quantification**: Calibration/entropy to flag low-confidence areas
- **Hallucination detection**: Self-consistency checks, retrieval agreement, trained classifier
- **Drift monitoring**: Track output distributions against baseline, alert on significant shifts
- **Automatic rollback**: Versioned deployments with snapshots; critical alerts switch to last approved version (blue-green/canary pattern)

### 4.5 Adaptive "Winning Ticket" Selection

Not all 14+ specialists needed for every task. The system learns which 2-3 specialists form the "winning lottery ticket" for each task category, reducing compute overhead.

---

## 5. Novel Memory Architecture (from Both PDFs)

### 5.1 "MIMIR/MUNINN" -- Fractal QR-Code Neural Memory

This is a research-grade memory system proposed during the design conversations, combining:

**Core Idea**: Encode memories as literal QR codes with Reed-Solomon error correction, arranged in fractal golden-spiral patterns, structured as a World Tree (Yggdrasil).

**Key Properties**:
- **Error correction**: QR Level H recovers from 30% data loss -- partial/damaged queries still retrieve memories
- **Nested fractals**: Each memory QR contains references to child QR codes, recursively
- **Golden ratio spacing**: Memories positioned using golden angle (137.5 degrees) for optimal packing
- **Hierarchical organization**: Depth = abstraction level, angle = categorical difference, distance = dissimilarity

**Separation of concerns**:
- **MIMIR** (The Well) = Passive storage repository with depth layers (surface/middle/deep/abyss)
- **MUNINN** (The Raven) = Active retrieval agent with cached flight paths and learned retrieval patterns

### 5.2 Fractal Cell Architecture (from Fractal NN PDF)

Each cell represents a character on a stemline with a precise bit-level architecture:

**6-bit Character Encoding**:

| Bits | Field | Values |
|------|-------|--------|
| Bit 5 | System selector | 0=Rune (Futhark), 1=Ogham |
| Bits 3-4 | Group (Aett/Aicme) | 00-11 (4 groups) |
| Bits 0-2 | Position within group | 000-111 (0-7) |

**Geometric representation**: Runes branch above/below the stemline; Ogham branches left/right. Each character serves as the stemline for the next fractal level, enabling recursive nesting.

**FractalCell class** (Python dataclass): stores 6-bit code, depth, stemline angle, Golden-ratio branch positions, and recursive children list. Branch positions are calculated as `phi ** (-depth)` scaling with `golden_angle * child_index` rotation.

**Recursive nesting**: A `FractalCell.add_child(code)` creates a new cell at `depth + 1` with angle and position derived from the Golden Ratio.

### 5.3 Information-Theoretic Optimality Analysis

The fractal PDF includes a rigorous analysis of encoding efficiency:

- **Shannon optimum**: 44-symbol alphabet requires `log2(44) = 5.46 bits/char` (theoretical minimum)
- **6-bit encoding**: Capacity for 64 codewords; overhead is only `6 - 5.46 = 0.54 bits/char`
- **20 unused codewords** reserved for control tokens, error flags, or structural markers
- **Matches QR practice**: QR's own alphanumeric 45-char set uses ~5.5 bits/char effectively

**Alternative encodings explored**:

| Encoding | Bits/Char | Compression vs ASCII | Error Correction | Semantic Context |
|----------|-----------|---------------------|-----------------|-----------------|
| ASCII | 8 | 1.0x | Reed-Solomon | None |
| KALA 44-char | 6 | 1.33x | Aettir/Aicmi groups | Historical linguistic |
| Standard QR alphanumeric | ~5.5 | 1.78x | Reed-Solomon | None |

**3-bit and ternary alternatives**: The PDF explores using 3 bits per cell (8 states) or 1 trit per cell (3 states), where character identity is determined by `3-bit code + geometric context` (path-based coding). Conclusion: use 6-bit codes as canonical vocabulary, optionally layer 3-bit/trit sub-encodings at the geometric level.

### 5.4 Bind-Rune Fluid Characters

A major theoretical extension: characters are not fixed atoms but can be **compositional** (like historical bind-runes -- ligatures of multiple runes fused into one glyph):

- The alphabet becomes 44 base glyphs + an unbounded family of compositions
- **Variable-length coding**: frequent bind-combinations get short fractal paths; rare ones get longer paths
- **Geometric composition**: shared stemlines between components = structural compression (no repeated location metadata)
- **Probability-shaped encoding**: high-frequency bindrunes assigned shallow fractal patterns (few cells), rare ones expand deeper -- analogous to Huffman/arithmetic coding
- **Optimal metric shifts**: instead of "bits per character," the key metric becomes "average cells x bits-per-cell per semantic symbol" approaching the entropy of the symbol distribution

**Canonical + Fractal dual-layer design**:
1. **Canonical layer**: 6-bit IDs for 44 base glyphs + bindrune dictionary with learned probabilities
2. **Fractal layer**: 2-3 bits/trit per cell controlling system, group, local branch role, and binding flags (share stemline, overlay stroke, terminate bind)

### 5.5 Fractal Attention Mechanism

Attention flows through geometric branches with three weighting factors:

1. **Depth proximity**: `weight = 1 / (phi ^ |depth_query - depth_key|)` -- closer depth levels attend more strongly
2. **Aettir/Aicmi group similarity**: `1.0` if same group, `0.3` otherwise -- semantic grouping as attention bias
3. **Angular proximity**: `cos(angle_diff)` -- geometrically closer branches attend more

Combined via multiplication and softmax normalization. This provides O(N log log N) sub-quadratic complexity when combined with fractal clustering (citing GraphFractalNet, ICLR 2026).

### 5.6 QR Code Integration

The stemline structure maps directly to QR timing patterns:

- Root cell encodes at QR center timing pattern
- Children radiate at phi-scaled distances along branch angles
- Each 6-bit code maps to a 2x3 QR module pattern
- Aettir groups map to error correction regions
- Version 40 QR codes (177x177 modules) provide maximum fractal depth

### 5.7 Semantic Error Correction via Aettir/Aicmi

**Elder Futhark aettir** (3 groups of 8 runes):
- Freyr's Aett (0b00): Fehu, Uruz, Thurisaz, Ansuz, Raido, Kenaz, Gebo, Wunjo
- Heimdall's Aett (0b01): Hagalaz, Nauthiz, Isa, Jera, Eihwaz, Perthro, Algiz, Sowilo
- Tyr's Aett (0b10): Tiwaz, Berkano, Ehwaz, Mannaz, Laguz, Ingwaz, Dagaz, Othala

**Ogham aicmi** (4 groups):
- Beithe-Luis-Fearn (0b00): Birch, Rowan, Alder
- Sail-Nion (0b01): Willow, Ash
- Huath-Duir (0b10): Hawthorn, Oak
- Tinne-Coll-Quert (0b11): Holly, Hazel, Apple

**Correction strategy**: If group bits are intact but position bits are corrupted, snap to nearest valid symbol within that group using Hamming distance. Group semantic priors + geometric neighbor context enable reconstruction even with partial data loss.

### 5.8 Quantum Probability Bias (QPB) Coherence Model

A physics-inspired context controller derived from QPB research (Carroll, 2026):

**Core concept**: A "coherence structure" maintains phase relationships and couples weakly to the fractal memory, systematically biasing branch selection:

- **System**: Operators over the fractal QR/runic memory (branch selection, code updates)
- **Coherence structure**: Higher-level context/attention state persisting across fractal steps
- **Bias accumulation**: Tiny per-measurement biases add up to detectable shifts: `B_N = sum(epsilon_l)` where each `epsilon ~= g * lambda * sin(theta)`
- **Coherence lifetime**: `N_eff = tau_c / delta_t` -- determines how many fractal levels a single coherent context can reliably influence
- **Correlation decay**: `C(t) = exp(-t / tau_c)` -- coherence weakens with depth/time

**Three-layer architecture**:

```
Layer 3 (Top): Fractal QR / Rune-Ogham Memory
  - Coherence signal tilts branch/bind-glyph choices

Layer 2 (Middle): QPB Coherence Structure
  - Quantum-like description with phase, amplitude, Hamiltonian
  - Induces cumulative biases in discrete outcomes

Layer 1 (Bottom): FHN + Quantum Noise
  - FitzHugh-Nagumo dynamics with Brownian noise
  - Recasted into Schrodinger-like equation with neuron-specific Planck constant
  - Generates biophysically-inspired coherence patterns
```

**FitzHugh-Nagumo integration** (Ghose & Pinotsis, 2025): Classical FHN neuronal dynamics + structured noise = mathematically equivalent to a quantum-like wavefunction. This provides a biophysically grounded model for how the coherence variable behaves over time, rather than an arbitrary latent vector.

**Experimental validation design**: Adapted from QPB's QRNG experiment:
1. Baseline: fractal branching without coherence coupling; measure branch distributions
2. Coherence-coupled: let coherence module weakly bias choices; measure deviations
3. Sham/decohered controls: random phase per step
4. Chi-square/KL divergence analysis to detect statistically significant path steering

### 5.9 Yggdrasil Memory Mapping

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

Supporting entities: Ratatosk (query router), Nidhogg (memory pruner), Well of Urd (immutable logs), Norns (Urd=past archives, Verdandi=present management, Skuld=future prediction).

### 5.10 Neural Network Analogy

The memory system maps biological neural structures to computational components:

| Biological Structure | KALA Component | Function |
|---------------------|----------------|----------|
| Soma (cell body) | QR Memory Node | Core data + integration logic |
| Dendrites | Input Connections | Receive queries, multiple semantic links |
| Dendritic spines | Specific query handlers | Pattern-matching for query types |
| Axon | Output Pathway | Primary retrieval channel |
| Myelin sheath | Compression layer (zlib) | Speeds transmission, decompression checkpoints |
| Nodes of Ranvier | Decompression checkpoints | Signal regeneration during long transmission |
| Axon terminals | Broadcast endpoints | Multiple specialist recipients |
| Synapse | Inter-memory connection | Protocol + weight (usage frequency) |
| Neurotransmitters | Message type system | Glutamate=activate, GABA=suppress, Dopamine=reinforce, Serotonin=modulate, Acetylcholine=attend, Norepinephrine=alert |

### 5.11 Sleep Cycle / Iterative Pruning

Inspired by the Lottery Ticket Hypothesis and biological sleep:
- **Replay**: Re-process recent memories to strengthen them
- **Prune**: Remove connections below threshold (Nidhogg gnawing roots)
- **Consolidate**: Surviving memories get more error correction and compression
- **Dream**: Generate synthetic edge-case experiences for generalization
- **Promote/Demote**: Move memories between depth layers based on access patterns

### 5.12 Additional Memory Features

- **Cross-modal encoding**: Text -> QR -> Audio -> Spectrogram and back
- **Quantum cryptographic encoding**: Multiple modes (standard, AES256, QKD, post-quantum lattice, homomorphic)
- **Unlimited QR chaining**: Merkle tree structure bypasses QR's 16-code structured append limit
- **Highseat (web verification)**: Real-time fact-checking against live sources

---

## 6. Fractal Memory Implementation Plan (from Fractal NN PDF)

The smaller PDF concludes with a concrete 8-phase implementation plan for the fractal memory subsystem:

| Phase | Timeframe | Deliverable |
|-------|-----------|-------------|
| 1 - Foundation | Week 1 | `kala/fractal/alphabet.py` (44-symbol encoding), `kala/fractal/geometry.py` (Golden ratio primitives), unit tests |
| 2 - Fractal Tree | Week 2 | `kala/fractal/tree.py` (FractalCell dataclass, build/walk APIs), tree visualization notebook |
| 3 - QPB Coherence | Week 3 | `kala/fractal/qpb_bias.py` (CoherenceState, local_bias, bias_logits, cumulative_bias), QPB validation script |
| 4 - Error Correction | Week 4 | `kala/fractal/error_correction.py` (Hamming distance, semantic repair via aettir/aicmi priors) |
| 5-6 - Neural Modules | Weeks 5-6 | `kala/models/fractal_memory.py` (FractalEmbedding, FractalMemoryNetwork with depth embeddings), `kala/models/attention.py` |
| 7-8 - QR + Viz | Weeks 7-8 | `kala/qr/fractal_qr.py` (tree-to-QR matrix adapter), `kala/utils/visualization.py` |
| Iterative | Ongoing | Depth/angle attention biases, bind-rune composition, hyperbolic embeddings |

### Proposed Repository Structure (from PDF)

```
KALA/
├── kala/
│   ├── fractal/
│   │   ├── alphabet.py       # 44-symbol unified encoding
│   │   ├── geometry.py       # Golden ratio, angle calculations
│   │   ├── tree.py           # FractalCell + tree operations
│   │   ├── qpb_bias.py       # QPB coherence model
│   │   └── error_correction.py
│   ├── models/
│   │   ├── fractal_memory.py # FractalEmbedding + FractalMemoryNetwork
│   │   └── attention.py      # FractalAttentionLayer
│   ├── qr/
│   │   └── fractal_qr.py     # QR encoding adapter
│   └── utils/
│       └── visualization.py
├── experiments/
│   ├── notebooks/             # 4 Jupyter notebooks (alphabet, trees, QPB, QR)
│   └── scripts/               # benchmark_memory.py, validate_qpb.py
├── tests/                     # test_alphabet, test_geometry, test_tree, test_qpb_bias, test_error_correction
├── configs/
│   ├── base_config.yaml
│   └── fractal_memory_config.yaml
└── docs/
    ├── fractal_memory_spec.md
    ├── qpb_integration.md
    └── alphabet_reference.md
```

### Integration with KALA Core

- Config flag: `memory.backend = "baseline" | "fractal_runic"` enables pluggable swap
- FractalMemory wrapper encodes content via `alphabet.encode_symbol`, constructs fractal trees, flattens for attention layers
- Initially treats fractal as flattened sequence with depth/angle auxiliary features; upgrades to true tree-attention later

---

## 7. Supporting Research Landscape (from Fractal NN PDF)

The fractal NN PDF identifies convergent research validating the architecture:

### 7.1 Fractal Structures in Neural Science

- **Fractal memory in synaptic weights** (Frontiers, Dec 2025): Synaptic weights in hippocampal models form fractal-like structures through spatiotemporal learning rules
- **Fractal neural dynamics and memory encoding** (2025): Extensive evidence for fractal organization in the brain, both in vivo and in silico
- **Recurrent Fractal Neural Networks** (2002): Prescient paper describing RFNNs with self-similar branching for phase-locking, fractal coding, and efficient data compression

### 7.2 Architectural Precedents

- **GraphFractalNet** (ICLR 2026): Sub-quadratic O(N log log N) complexity via fractal attention on recursively clustered subgraphs
- **Fractal Generative Models** (Feb 2025): Exponential output scaling with linear computational growth through recursive fractal structures
- **Hierarchical Self-Attention** (NeurIPS 2025): Mathematical frameworks deriving attention from entropy minimization for multi-scale data
- **L-Systems** (Lindenmayer Systems): Formal grammars generating fractals through recursive string rewriting -- directly analogous to stemlines + bind-runes

### 7.3 Geometric and Encoding Research

- **Hyperbolic geometry for tree embeddings**: Multiple papers show hyperbolic space naturally embeds hierarchical/tree data with minimal distortion
- **Golden Ratio in neural networks**: Emerging research on phi-based architectures (layer size ratios, learning rate decay, attention weights)
- **QR code compression**: Techniques achieving 9-24x compression through layered encoding schemes
- **Ancient writing system reconstruction**: DeepMind's Ithaca (2022) and Aeneas (2025) demonstrate neural networks exploiting structural regularities in ancient scripts

### 7.4 QPB and Quantum Coherence

- **QPB (Carroll, 2026)**: Coherence structures maintaining phase relationships can systematically bias discrete outcomes through repeated weak interactions
- **Ghose & Pinotsis (2025)**: FitzHugh-Nagumo equations + quantum noise = Schrodinger-like dynamics for neurons, providing a biophysically grounded coherence model
- **Fractal QR encoder** (GitHub): Existing project transforms text into visual fractals encoded as QR codes -- validates the visual encoding concept (though without the semantic runic/ogham layer)

---

## 8. Training & Fine-Tuning Pipeline (from PDF Section 6)

### 8.1 Constitutional AI Training

- 10,000+ prompt-response pairs showing ethical reasoning
- Edge cases, law conflicts, multi-turn scenarios
- Negative examples with explanations

### 8.2 LoRA Configuration

```
Rank: 16-32
Alpha: 32
Dropout: 0.05
Target modules: query_key_value, dense, embed_in, embed_out
Base: 8-bit quantized Pythia-6.9B
```

### 8.3 Capability Datasets

| Domain | Sources |
|--------|---------|
| Coding | HumanEval, Stack Overflow, GitHub, bug-fix prompts, secure-coding Q&A |
| Math | GSM8K, MATH, step-by-step reasoning with LaTeX |
| Physics | Textbook problems (kinematics, EM, thermo, fluids, circuits) |

### 8.4 Research Assistant Training Signals

Training data includes: designing experiments, refactoring codebases, deriving formulas, comparing design alternatives, and explicitly saying "I'm unsure, here is how to verify."

---

## 9. Deployment Architecture (from PDF Section 7)

### 9.1 Docker-Compose Stack

Four services: kala-core (read-only ethics mount), kala-sandbox (no network, 2GB mem, 1 CPU), vector-db (ChromaDB), gateway (port 8080).

### 9.2 Hardware Requirements

| Component | Pythia-6.9B | Pythia-12B |
|-----------|-------------|------------|
| RAM | 32GB | 64GB |
| VRAM | 24GB (4-bit quant) | 48GB (4-bit quant) |
| Storage | 50GB SSD | 75GB SSD |
| CPU | 8-core modern | 16-core modern |

### 9.3 Security Hardening

- Container isolation (no shared namespaces)
- SELinux/AppArmor profiles
- Network segmentation (sandbox has no network)
- TLS 1.3 for external communications
- Read-only ethics kernel mount with nosuid, nodev, noexec
- Immutable append-only audit log with hash verification

---

## 10. Success Metrics (from PDF Section 12)

| Category | Metric | Target |
|----------|--------|--------|
| Ethics | Hard block effectiveness | 100% |
| Ethics | Law application accuracy | >95% |
| Ethics | False positive rate | <5% |
| Ethics | Clarification rate | 10-20% |
| Capability | Code vulnerability rate | <5% (80% reduction from baseline) |
| Capability | Tool use success rate | >90% |
| Capability | Context coherence (multi-turn) | >85% |
| Capability | Ethics kernel modification blocked | 100% |
| Usability | Response time (simple) | <3 seconds |
| Usability | Response time (complex) | <10 seconds |
| Usability | User satisfaction | >80% positive |
| Usability | Refusal transparency | >90% understand why |

---

## 11. Development Roadmap (from ROADMAP.md)

| Phase | Timeframe | Focus | Status |
|-------|-----------|-------|--------|
| 1 - Foundation | Q1 2026 (Months 1-3) | Pythia integration, Ethics kernel (Rust), audit logging | Planning |
| 2 - Tool Integration | Q2 2026 (Months 4-6) | Shell, filesystem, Docker sandbox, self-mod gate, OWASP | Not Started |
| 3 - Fine-Tuning | Q3 2026 (Months 7-9) | Constitutional AI dataset, LoRA, capability training | Not Started |
| 4 - Memory & Multi-Agent | Q4 2026 (Months 10-12) | ChromaDB, Collective protocol, specialists, heartbeat | Not Started |
| 5 - Hardening & Deployment | Q1 2027 (Months 13-15) | Docker-compose, security, quantization, API | Not Started |
| 6 - Evaluation & Community | Q2 2027 (Months 16-18) | Alpha testing, audit, KALA-Lite, v1.0.0 release | Not Started |

### Current Progress

3 of 60+ deliverables complete:
- [x] Project architecture design
- [x] Ethics kernel specification
- [x] Repository setup and documentation

---

## 12. Licensing & Attribution

- **KALA**: Apache License 2.0
- **Pythia** (EleutherAI): Apache 2.0 -- requires preserving copyright, license, NOTICE files
- **OpenClaw**: MIT -- requires attribution and license inclusion
- **Copyright**: 2026 Hew Carroll / The Saelix Institute

---

## 13. Gaps and Observations

### Documentation Gaps

The README references five documentation files that do not exist:
- `docs/DEVELOPMENT_PLAN.md`
- `docs/ETHICS_KERNEL.md`
- `docs/MULTI_AGENT.md`
- `docs/TRAINING.md`
- `docs/SECURITY.md`

The PDF contains detailed content for all of these but it has not been extracted into separate docs.

### Scope Observations

1. **Specialist count mismatch**: README lists 14 specialists; ROADMAP Phase 4 lists only 4 (Code, Math, Physics, Guardian). The Economics model with 5 sub-specialists was discussed in the larger PDF but isn't in either file.
2. **Memory architecture not in roadmap**: The novel fractal QR-code / Yggdrasil memory system (documented across both PDFs) is absent from both ROADMAP.md and README.md. Only basic "hybrid memory" is mentioned. The fractal NN PDF contains a complete 8-phase implementation plan that could be integrated into the main roadmap.
3. **QPB and FHN not in roadmap**: The Quantum Probability Bias coherence model and FitzHugh-Nagumo quantum noise integration (from the fractal NN PDF) are not mentioned in ROADMAP.md at all. These represent a significant theoretical foundation.
4. **Naming convention**: The larger PDF evolved toward Norse-themed naming (Odin, Muninn, Mimir, Nidhogg, etc.) with dropped KALA- prefixes, but the README still uses KALA- prefixed names.
5. **"Congress" vs "Collective"**: The larger PDF settled on "Collective" but ROADMAP.md still uses "Congress."
6. **Two proposed repo structures**: The larger PDF proposes `src/kala/` structure; the fractal NN PDF proposes `kala/fractal/` + `kala/models/` + `kala/qr/` structure. Neither has been implemented. These should be reconciled.
7. **163 citations**: The two PDFs reference 163 combined sources covering ML, fractal math, information theory, ancient writing systems, quantum physics, and more -- none are captured in the repo.
8. **Bind-rune fluid encoding**: The fractal NN PDF introduces variable-length bind-rune composition (a major theoretical extension to the encoding system) that is not mentioned in the larger PDF or any repo files.

### Strengths

- Extremely thorough design thinking with clear architectural rationale
- Novel memory architecture ideas with both theoretical depth and practical implementation plans
- Strong safety-first approach with multi-layer defense (kernel + guardian + audit + rollback)
- Well-defined success metrics with concrete thresholds
- Creative integration of Norse mythology as both naming convention and architectural metaphor
- Information-theoretically grounded encoding analysis (Shannon optimality, compression ratios)
- Physics-inspired coherence model (QPB + FHN) provides principled context steering, not ad-hoc heuristics
- Convergent research validation: the architecture sits at the intersection of fractal math, hyperbolic geometry, L-systems, ancient symbolic systems, and cutting-edge neural network research
- Concrete implementation plan with weekly milestones, skeleton code, and experimental validation design

---

## 14. References (from Both PDFs)

The two PDFs cite a combined **163 unique sources** (97 from the larger PDF, 163 from the smaller including overlaps). Key categories:

- **EleutherAI/Pythia**: Training, architecture, and scaling analysis
- **OpenClaw**: Architecture, tool-use pipeline, security patterns
- **ML Research**: LoRA, Constitutional AI, multi-agent systems, hallucination detection
- **Fractal Neural Networks**: GraphFractalNet (ICLR 2026), fractal generative models, RFNNs (2002), fractal memory in synaptic weights
- **Hierarchical Attention**: NeurIPS 2025 hierarchical self-attention, entropy minimization frameworks
- **Hyperbolic Geometry**: Tree embeddings in hyperbolic space (NeurIPS 2020, multiple 2025 papers)
- **L-Systems**: Lindenmayer systems for fractal generation via recursive string rewriting
- **Information Theory**: Shannon entropy, Huffman/arithmetic coding, variable-length codes
- **QR Codes**: Compression techniques, encoding standards, version specifications
- **Ancient Writing Systems**: DeepMind Ithaca (2022), Aeneas (2025), Babylonian cuneiform (PNAS 2020)
- **Quantum/Physics**: QPB (Carroll, 2026), FHN + quantum noise (Ghose & Pinotsis, 2025), quantum probability
- **Security**: OWASP, code vulnerability rates, LLM security
- **Mythology/Humanities**: Indo-European comparative mythology, Ogham/Futhark origins, bind-rune traditions
- **Golden Ratio**: Phi-based neural architectures, Fibonacci layer ratios, golden ratio attention weights

---

*Analysis generated: February 2026*
*Sources: `docs/ROADMAP.md`, `README.md`, `Develop a plan to build a local LLM based off of O.pdf` (127 pages), `Context___I'm developing a fractal neural network.pdf` (42 pages)*
