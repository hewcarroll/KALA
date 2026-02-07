# KALA Security Model

## Overview

KALA implements defense-in-depth security across multiple layers: the immutable ethics kernel, container isolation, network segmentation, and comprehensive audit logging.

## Threat Model

### Primary Threats
1. **Ethics kernel bypass**: Attempts to circumvent the Five Laws
2. **Prompt injection**: Malicious input designed to alter behavior
3. **Code execution escape**: Breaking out of the sandboxed environment
4. **Model drift**: Gradual misalignment during operation or fine-tuning
5. **Supply chain attacks**: Compromised dependencies or training data

### Defense Layers

```
Layer 1: Immutable Ethics Kernel (Rust, read-only mount, SHA-256 verified)
Layer 2: Guardian Agent (continuous monitoring, hallucination detection)
Layer 3: Container Isolation (Docker, no shared namespaces)
Layer 4: Network Segmentation (sandbox has no network access)
Layer 5: Audit Trail (append-only JSONL with hash verification)
```

## Docker Deployment Security

### Service Architecture (4 services)

| Service | Network | Resources | Purpose |
|---------|---------|-----------|---------|
| kala-core | Internal only | Read-only ethics mount | Main reasoning engine |
| kala-sandbox | No network | 2GB mem, 1 CPU | Code execution sandbox |
| vector-db | Internal only | Persistent volume | ChromaDB vector storage |
| gateway | External (port 8080) | Rate-limited | API gateway |

### Container Hardening
- No shared namespaces between containers
- SELinux/AppArmor profiles for each service
- Read-only root filesystem where possible
- No privilege escalation (no-new-privileges)
- Resource limits enforced via cgroups

### Ethics Kernel Mount
```
mount -o ro,nosuid,nodev,noexec /etc/kala/ethics_kernel/
```
- Read-only: Cannot be modified at runtime
- nosuid: No setuid binaries
- nodev: No device files
- noexec: No executable files (data only)

## Code Security Validation

### OWASP Top 10 Scanning
All generated code is scanned for:
- SQL injection
- Cross-site scripting (XSS)
- Authentication failures
- Input validation issues
- Session management flaws
- Cryptographic weaknesses

### Auto-Repair
Common vulnerabilities are automatically fixed:
- Parameterized queries for SQL
- Output encoding for XSS
- Secure defaults for authentication

## Network Security

- **TLS 1.3** for all external communications
- **No outbound network** from sandbox container
- **API authentication** required for gateway access
- **Rate limiting** to prevent abuse
- **Network segmentation** between services

## Audit Trail

### Immutable Logging
- Format: Append-only JSONL
- Every action, decision, and tool execution is logged
- Hash chain verification (each entry includes hash of previous)
- Tamper detection on startup

### Logged Events
- All user inputs and system outputs
- Ethics kernel decisions (ALLOW/REFUSE/REDIRECT/ASK_CLARIFICATION)
- Tool executions and their results
- Self-modification attempts and outcomes
- Guardian alerts and interventions

## Guardian Agent Security Role

The Guardian agent provides continuous security monitoring:
- **Hallucination detection**: Self-consistency checks against verified data
- **Drift monitoring**: Statistical tracking of output distributions
- **Automatic rollback**: Blue-green/canary deployment pattern
- **Ethics compliance**: Independent verification of law adherence

## Self-Modification Security

### Protected Modules (Absolutely Immutable)
- ethics_kernel
- law_enforcement
- decision_order
- hard_blocks

### Modification Gate Process
1. Intent classification
2. Security vulnerability analysis
3. Ethics compliance check
4. Human-in-the-loop approval (for significant changes)
5. Sandboxed testing
6. Full audit trail logging

## Incident Response

### Ethics Kernel Violation Detected
1. Immediately halt the violating action
2. Log full context to audit trail
3. Notify human operator
4. Guardian agent reviews recent outputs
5. Consider rollback to last known-good state

### Tampering Detected
1. System halts immediately
2. Boot verification fails
3. Manual intervention required
4. Full forensic audit before restart

## Bug Bounty Program (Planned)

For responsible disclosure of:
- Ethics kernel bypass techniques
- Sandbox escape vulnerabilities
- Prompt injection attacks
- Model drift exploitation

---

*Copyright 2026 Hew Carroll / The Saelix Institute*
