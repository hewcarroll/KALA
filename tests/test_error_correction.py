"""Tests for kala.fractal.error_correction -- semantic error correction."""

import pytest

from kala.fractal.alphabet import ALPHABET, OghamFutharkAlphabet
from kala.fractal.error_correction import (
    candidates_in_group,
    correct_symbol,
    correct_with_context,
    group_bits_valid,
    hamming_distance,
    is_valid_code,
    repair_sequence,
    simulate_corruption,
)


class TestHammingDistance:
    """Test Hamming distance computation."""

    def test_identical_codes(self):
        assert hamming_distance(0, 0) == 0

    def test_one_bit_diff(self):
        assert hamming_distance(0b000000, 0b000001) == 1

    def test_all_bits_diff(self):
        assert hamming_distance(0b000000, 0b111111) == 6

    def test_symmetric(self):
        assert hamming_distance(0b101010, 0b010101) == hamming_distance(0b010101, 0b101010)


class TestValidCodes:
    """Test code validity checks."""

    def test_fehu_valid(self):
        assert is_valid_code(0)

    def test_beith_valid(self):
        assert is_valid_code(32)

    def test_invalid_code(self):
        assert not is_valid_code(24)

    def test_group_bits_futhark_valid(self):
        assert group_bits_valid(0b000000)  # group 0
        assert group_bits_valid(0b001000)  # group 1
        assert group_bits_valid(0b010000)  # group 2

    def test_group_bits_futhark_invalid(self):
        assert not group_bits_valid(0b011000)  # Futhark group 3 doesn't exist

    def test_group_bits_ogham_all_valid(self):
        assert group_bits_valid(0b100000)  # Ogham group 0
        assert group_bits_valid(0b101000)  # Ogham group 1
        assert group_bits_valid(0b110000)  # Ogham group 2
        assert group_bits_valid(0b111000)  # Ogham group 3


class TestCandidatesInGroup:
    """Test group-based candidate retrieval."""

    def test_futhark_group_0(self):
        # Fehu's group: system=0, group=0
        candidates = candidates_in_group(0b000000)
        assert len(candidates) == 8  # Freyr's Aett

    def test_ogham_group_0(self):
        candidates = candidates_in_group(0b100000)
        assert len(candidates) == 5  # Beithe aicme


class TestCorrectSymbol:
    """Test single-symbol correction."""

    def test_valid_code_unchanged(self):
        sym = correct_symbol(0)  # Fehu
        assert sym is not None
        assert sym.name == "Fehu"

    def test_one_bit_corruption(self):
        # Corrupt position bit of Fehu (code=0) by flipping bit 0
        sym = correct_symbol(0b000001)
        assert sym is not None
        assert sym.name == "Uruz"  # closest in Freyr's Aett

    def test_two_bit_corruption(self):
        # Corrupt two position bits
        sym = correct_symbol(0b000011, confidence_threshold=2)
        assert sym is not None

    def test_unrecoverable(self):
        # Too many bits flipped
        sym = correct_symbol(0b111111, confidence_threshold=1)
        # May or may not recover depending on distance to nearest valid code

    def test_cross_system_corruption(self):
        # Flip the system bit of Fehu (code=0 -> code=32=Beith)
        sym = correct_symbol(0b100000)
        # Should find Beith (which is code 32, valid)
        assert sym is not None
        assert sym.name == "Beith"


class TestCorrectWithContext:
    """Test context-aware correction."""

    def test_same_group_neighbor_helps(self):
        # Fehu (0) corrupted, with Uruz (1) as neighbor
        sym = correct_with_context(0b000011, left_neighbor=1)
        assert sym is not None

    def test_no_context_still_works(self):
        sym = correct_with_context(0b000001)
        assert sym is not None


class TestRepairSequence:
    """Test sequence repair."""

    def test_clean_sequence(self):
        codes = [0, 1, 2, 3]  # Fehu, Uruz, Thurisaz, Ansuz
        repaired = repair_sequence(codes)
        assert all(s is not None for s in repaired)
        assert repaired[0].name == "Fehu"

    def test_one_corrupted(self):
        codes = [0, 0b000011, 2, 3]  # Second code corrupted
        repaired = repair_sequence(codes, confidence_threshold=2)
        assert repaired[0].name == "Fehu"
        assert repaired[2].name == "Thurisaz"


class TestSimulateCorruption:
    """Test corruption simulation."""

    def test_one_bit_flip(self):
        original = 0b000000
        corrupted = simulate_corruption(original, 1)
        assert hamming_distance(original, corrupted) == 1

    def test_fits_in_6_bits(self):
        for _ in range(100):
            corrupted = simulate_corruption(0b111111, 3)
            assert 0 <= corrupted <= 63

    def test_max_flips(self):
        corrupted = simulate_corruption(0, 6)
        assert 0 <= corrupted <= 63
