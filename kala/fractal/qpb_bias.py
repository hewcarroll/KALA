"""
Quantum Probability Bias (QPB) inspired coherence model.

Based on: Carroll, H. (2026). Quantum Probability Bias. Saelix Institute.
Integration with: Ghose, P. & Pinotsis, D. A. (2025). The FitzHugh-Nagumo
    Equations and Quantum Noise. CSBJ, 30, pp. 12-20.

Three-layer architecture:
    Layer 1 (Bottom): FHN + quantum noise -- generates biophysically-inspired
        coherence patterns.
    Layer 2 (Middle): QPB coherence structure -- maintains phase relationships,
        induces cumulative biases in discrete outcomes.
    Layer 3 (Top): Fractal QR / rune-ogham memory -- coherence signal tilts
        branch/bind-glyph choices.

The coherence model provides context-driven probabilistic steering of
fractal memory traversal, where tiny per-measurement biases accumulate
to produce detectable shifts in branch selection statistics.

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under the Apache License, Version 2.0
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class CoherenceState:
    """Represents a coherence structure maintaining phase over time.

    Maps to QPB's external coherence structure with lifetime tau_c.
    The coherence structure weakly couples to the fractal memory system
    to bias branch selection decisions.

    Attributes:
        phase: Phase angle theta in [0, 2*pi].
        lifetime: Coherence decay timescale tau_c.
        coupling: Coupling constant g (interaction strength).
        timestamp: Time of last update.
    """
    phase: float                # theta in [0, 2*pi]
    lifetime: float = 10.0     # tau_c (coherence decay timescale)
    coupling: float = 0.1      # g (coupling constant)
    timestamp: float = 0.0     # time of last measurement

    def correlation(self, dt: float) -> float:
        """Coherence correlation function: C(t) = exp(-t/tau_c).

        Returns the correlation strength after time interval dt.
        As dt grows, coherence decays exponentially.
        """
        if self.lifetime <= 0:
            return 0.0
        return math.exp(-dt / self.lifetime)

    def effective_interactions(self, dt: float, total_steps: int) -> float:
        """Effective number of coherent interactions.

        N_eff = min(N, tau_c / dt)

        This determines how many fractal levels a single coherent context
        can reliably influence before its steering power saturates.
        """
        if dt <= 0:
            return float(total_steps)
        return min(total_steps, self.lifetime / dt)

    def advance(self, dt: float) -> "CoherenceState":
        """Return a new CoherenceState advanced by dt time units.

        Phase evolves; correlation decays.
        """
        return CoherenceState(
            phase=self.phase,
            lifetime=self.lifetime,
            coupling=self.coupling,
            timestamp=self.timestamp + dt,
        )


def local_bias(coherence: CoherenceState, measurement_strength: float = 0.01) -> float:
    """Compute instantaneous bias epsilon_l for a single measurement.

    In QPB: delta_p = -2 * lambda * theta * Im Tr[AC rho]
    Simplified surrogate: epsilon ~= g * lambda * sin(theta)

    Args:
        coherence: Current coherence state.
        measurement_strength: Lambda (weak measurement strength).

    Returns:
        Bias magnitude (can be positive or negative).
    """
    return coherence.coupling * measurement_strength * math.sin(coherence.phase)


def cumulative_bias(
    coherences: List[CoherenceState],
    measurement_strength: float = 0.01,
) -> float:
    """Compute total accumulated bias over N weak measurements.

    B_N = sum(epsilon_l) for l = 1..N

    In QPB, this demonstrates that tiny per-measurement biases can add up
    to a detectable shift when coherence is maintained.

    Args:
        coherences: List of coherence states at each step.
        measurement_strength: Lambda for each measurement.

    Returns:
        Total accumulated bias.
    """
    return sum(local_bias(c, measurement_strength) for c in coherences)


def bias_logits_numpy(
    base_logits: List[float],
    coherence: CoherenceState,
    dt: float = 1.0,
    measurement_strength: float = 0.01,
) -> List[float]:
    """Apply QPB-style bias to branch selection logits (pure Python).

    Args:
        base_logits: Unbiased logits for branch choices.
        coherence: Current coherence state.
        dt: Time since last measurement.
        measurement_strength: Lambda (weak measurement strength).

    Returns:
        Biased logits incorporating coherence effect.
    """
    corr = coherence.correlation(dt)
    epsilon = local_bias(coherence, measurement_strength) * corr
    return [logit + epsilon for logit in base_logits]


def softmax(logits: List[float]) -> List[float]:
    """Numerically stable softmax over a list of logits."""
    max_logit = max(logits)
    exps = [math.exp(l - max_logit) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def biased_branch_probabilities(
    num_branches: int,
    coherence: CoherenceState,
    base_logits: Optional[List[float]] = None,
    dt: float = 1.0,
    measurement_strength: float = 0.01,
) -> List[float]:
    """Compute branch selection probabilities with QPB coherence bias.

    Args:
        num_branches: Number of available branches.
        coherence: Current coherence state.
        base_logits: Optional base logits (defaults to uniform).
        dt: Time since last measurement.
        measurement_strength: Weak measurement strength.

    Returns:
        Probability distribution over branches.
    """
    if base_logits is None:
        base_logits = [0.0] * num_branches

    biased = bias_logits_numpy(base_logits, coherence, dt, measurement_strength)
    return softmax(biased)


# ---------------------------------------------------------------------------
# PyTorch integration (optional)
# ---------------------------------------------------------------------------

if HAS_TORCH:
    def bias_logits(
        base_logits: "torch.Tensor",
        coherence: CoherenceState,
        dt: float = 1.0,
        measurement_strength: float = 0.01,
    ) -> "torch.Tensor":
        """Apply QPB-style bias to branch selection logits (PyTorch).

        Args:
            base_logits: Unbiased logits tensor [num_branches] or [batch, num_branches].
            coherence: Current coherence state.
            dt: Time since last measurement.
            measurement_strength: Lambda (weak measurement strength).

        Returns:
            Biased logits tensor.
        """
        corr = coherence.correlation(dt)
        epsilon = local_bias(coherence, measurement_strength) * corr
        return base_logits + epsilon


# ---------------------------------------------------------------------------
# FitzHugh-Nagumo quantum noise integration (Layer 1)
# ---------------------------------------------------------------------------

@dataclass
class FHNState:
    """FitzHugh-Nagumo neural oscillator state.

    Models the bottom layer of the coherence architecture:
    Classical FHN dynamics + structured noise = quantum-like wavefunction.

    Based on Ghose & Pinotsis (2025):
        The FHN equations with Brownian noise can be recast into a
        Schrodinger-like equation with a neuron-specific Planck constant.

    Attributes:
        v: Membrane potential (fast variable).
        w: Recovery variable (slow variable).
        a: Recovery rate parameter.
        b: Recovery sensitivity.
        tau: Time constant ratio (slow/fast).
        I_ext: External input current.
    """
    v: float = 0.0      # membrane potential
    w: float = 0.0      # recovery variable
    a: float = 0.7      # recovery rate
    b: float = 0.8      # recovery sensitivity
    tau: float = 12.5   # time constant ratio
    I_ext: float = 0.5  # external input

    def step(self, dt: float = 0.01, noise_amplitude: float = 0.0) -> "FHNState":
        """Advance the FHN oscillator by one time step.

        Uses Euler-Maruyama integration:
            dv/dt = v - v^3/3 - w + I_ext
            dw/dt = (v + a - b*w) / tau

        Args:
            dt: Time step size.
            noise_amplitude: Standard deviation of Gaussian noise on v.

        Returns:
            New FHNState after one step.
        """
        import random
        noise = random.gauss(0, noise_amplitude) if noise_amplitude > 0 else 0.0

        dv = (self.v - (self.v ** 3) / 3.0 - self.w + self.I_ext) * dt + noise * math.sqrt(dt)
        dw = ((self.v + self.a - self.b * self.w) / self.tau) * dt

        return FHNState(
            v=self.v + dv,
            w=self.w + dw,
            a=self.a,
            b=self.b,
            tau=self.tau,
            I_ext=self.I_ext,
        )

    def to_coherence_phase(self) -> float:
        """Map FHN state to a coherence phase angle.

        Uses the membrane potential cycle to derive phase,
        following the quantum-like reformulation of Ghose & Pinotsis.
        """
        return math.atan2(self.w, self.v) % (2 * math.pi)

    def to_coherence_state(self, lifetime: float = 10.0, coupling: float = 0.1) -> CoherenceState:
        """Convert FHN oscillator state to a CoherenceState for QPB.

        This bridges Layer 1 (FHN) to Layer 2 (QPB):
        The FHN phase becomes the coherence structure's phase.
        """
        return CoherenceState(
            phase=self.to_coherence_phase(),
            lifetime=lifetime,
            coupling=coupling,
            timestamp=0.0,
        )


def simulate_fhn_trajectory(
    steps: int = 1000,
    dt: float = 0.01,
    noise_amplitude: float = 0.05,
    initial_state: Optional[FHNState] = None,
) -> List[FHNState]:
    """Simulate an FHN oscillator trajectory.

    Returns a list of states, one per time step, which can be
    converted to coherence phases for the QPB layer.

    Args:
        steps: Number of simulation steps.
        dt: Time step size.
        noise_amplitude: Noise level (0 = deterministic).
        initial_state: Starting state (defaults to rest).

    Returns:
        List of FHNState objects.
    """
    state = initial_state or FHNState()
    trajectory = [state]
    for _ in range(steps):
        state = state.step(dt, noise_amplitude)
        trajectory.append(state)
    return trajectory
