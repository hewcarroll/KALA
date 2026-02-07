"""Tests for kala.fractal.qpb_bias -- QPB coherence model."""

import math

import pytest

from kala.fractal.qpb_bias import (
    CoherenceState,
    FHNState,
    biased_branch_probabilities,
    bias_logits_numpy,
    cumulative_bias,
    local_bias,
    simulate_fhn_trajectory,
    softmax,
)


class TestCoherenceState:
    """Test CoherenceState operations."""

    def test_default_values(self):
        cs = CoherenceState(phase=0.0)
        assert cs.phase == 0.0
        assert cs.lifetime == 10.0
        assert cs.coupling == 0.1

    def test_correlation_at_zero(self):
        cs = CoherenceState(phase=0.0, lifetime=10.0)
        assert cs.correlation(0.0) == 1.0

    def test_correlation_decay(self):
        cs = CoherenceState(phase=0.0, lifetime=10.0)
        c1 = cs.correlation(5.0)
        c2 = cs.correlation(10.0)
        assert c1 > c2
        assert abs(c2 - math.exp(-1.0)) < 1e-12

    def test_correlation_zero_lifetime(self):
        cs = CoherenceState(phase=0.0, lifetime=0.0)
        assert cs.correlation(1.0) == 0.0

    def test_effective_interactions(self):
        cs = CoherenceState(phase=0.0, lifetime=10.0)
        n_eff = cs.effective_interactions(dt=1.0, total_steps=100)
        assert n_eff == 10.0

    def test_effective_interactions_capped(self):
        cs = CoherenceState(phase=0.0, lifetime=100.0)
        n_eff = cs.effective_interactions(dt=1.0, total_steps=5)
        assert n_eff == 5

    def test_advance(self):
        cs = CoherenceState(phase=1.0, lifetime=10.0, timestamp=0.0)
        cs2 = cs.advance(5.0)
        assert cs2.timestamp == 5.0
        assert cs2.phase == cs.phase
        assert cs2.lifetime == cs.lifetime


class TestLocalBias:
    """Test instantaneous bias calculation."""

    def test_zero_phase_zero_bias(self):
        cs = CoherenceState(phase=0.0, coupling=0.1)
        assert local_bias(cs) == 0.0

    def test_pi_half_maximum_bias(self):
        cs = CoherenceState(phase=math.pi / 2, coupling=0.1)
        expected = 0.1 * 0.01 * 1.0  # g * lambda * sin(pi/2)
        assert abs(local_bias(cs) - expected) < 1e-12

    def test_negative_bias_at_negative_phase(self):
        cs = CoherenceState(phase=-math.pi / 2, coupling=0.1)
        assert local_bias(cs) < 0


class TestCumulativeBias:
    """Test accumulated bias over multiple measurements."""

    def test_single_measurement(self):
        cs = CoherenceState(phase=math.pi / 2, coupling=0.1)
        total = cumulative_bias([cs])
        assert total == local_bias(cs)

    def test_accumulation(self):
        states = [CoherenceState(phase=math.pi / 2, coupling=0.1) for _ in range(10)]
        total = cumulative_bias(states)
        expected = 10 * local_bias(states[0])
        assert abs(total - expected) < 1e-10

    def test_cancellation(self):
        """Opposite phases should cancel out."""
        cs_pos = CoherenceState(phase=math.pi / 2, coupling=0.1)
        cs_neg = CoherenceState(phase=-math.pi / 2, coupling=0.1)
        total = cumulative_bias([cs_pos, cs_neg])
        assert abs(total) < 1e-12


class TestBiasLogitsNumpy:
    """Test bias application to logits."""

    def test_uniform_logits_stay_uniform_at_zero_phase(self):
        cs = CoherenceState(phase=0.0)
        logits = [0.0, 0.0, 0.0]
        biased = bias_logits_numpy(logits, cs)
        # All should be equal (zero bias)
        assert all(abs(b - biased[0]) < 1e-12 for b in biased)

    def test_bias_shifts_logits(self):
        cs = CoherenceState(phase=math.pi / 2, coupling=0.5)
        logits = [0.0, 0.0]
        biased = bias_logits_numpy(logits, cs, dt=0.1)
        # Biased logits should be different from original
        assert biased[0] != 0.0


class TestSoftmax:
    """Test softmax implementation."""

    def test_uniform_logits(self):
        probs = softmax([0.0, 0.0, 0.0])
        for p in probs:
            assert abs(p - 1.0 / 3) < 1e-12

    def test_sums_to_one(self):
        probs = softmax([1.0, 2.0, 3.0])
        assert abs(sum(probs) - 1.0) < 1e-12

    def test_ordering_preserved(self):
        probs = softmax([1.0, 2.0, 3.0])
        assert probs[0] < probs[1] < probs[2]

    def test_numerical_stability(self):
        """Large values should not cause overflow."""
        probs = softmax([1000.0, 1001.0, 1002.0])
        assert abs(sum(probs) - 1.0) < 1e-10


class TestBiasedBranchProbabilities:
    """Test branch probability computation."""

    def test_uniform_without_bias(self):
        cs = CoherenceState(phase=0.0)
        probs = biased_branch_probabilities(3, cs)
        for p in probs:
            assert abs(p - 1.0 / 3) < 1e-10

    def test_sums_to_one(self):
        cs = CoherenceState(phase=math.pi / 4, coupling=0.5)
        probs = biased_branch_probabilities(4, cs)
        assert abs(sum(probs) - 1.0) < 1e-10


class TestFHNState:
    """Test FitzHugh-Nagumo oscillator."""

    def test_default_state(self):
        state = FHNState()
        assert state.v == 0.0
        assert state.w == 0.0

    def test_single_step(self):
        state = FHNState()
        new_state = state.step(dt=0.01)
        # State should change (external input drives it away from rest)
        assert new_state.v != state.v or new_state.w != state.w

    def test_deterministic_reproducibility(self):
        s1 = FHNState().step(dt=0.01, noise_amplitude=0.0)
        s2 = FHNState().step(dt=0.01, noise_amplitude=0.0)
        assert s1.v == s2.v
        assert s1.w == s2.w

    def test_to_coherence_phase(self):
        state = FHNState(v=1.0, w=0.0)
        phase = state.to_coherence_phase()
        assert 0 <= phase < 2 * math.pi

    def test_to_coherence_state(self):
        fhn = FHNState(v=1.0, w=1.0)
        cs = fhn.to_coherence_state(lifetime=5.0, coupling=0.2)
        assert isinstance(cs, CoherenceState)
        assert cs.lifetime == 5.0
        assert cs.coupling == 0.2


class TestFHNSimulation:
    """Test FHN trajectory simulation."""

    def test_trajectory_length(self):
        traj = simulate_fhn_trajectory(steps=100, dt=0.01)
        assert len(traj) == 101  # initial + 100 steps

    def test_trajectory_deterministic(self):
        t1 = simulate_fhn_trajectory(steps=10, noise_amplitude=0.0)
        t2 = simulate_fhn_trajectory(steps=10, noise_amplitude=0.0)
        for s1, s2 in zip(t1, t2):
            assert s1.v == s2.v
