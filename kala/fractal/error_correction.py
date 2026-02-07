"""
Semantic error correction using aettir/aicmi group context.

The Elder Futhark aettir (3 groups of 8 runes) and Ogham aicmi (4 groups
of 5 characters) provide semantic context for reconstructing corrupted
6-bit codes. When position bits are damaged but group/system bits survive,
the group membership constrains which symbols are plausible replacements.

Correction strategy:
    1. If group bits intact: snap to nearest valid symbol within that group
       using Hamming distance.
    2. If group bits corrupted: search all symbols, using geometric neighbor
       context and sequence statistics for disambiguation.

This provides built-in architectural error correction beyond QR's
Reed-Solomon codes.

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under the Apache License, Version 2.0
"""

from typing import List, Optional, Tuple

from .alphabet import (
    ALPHABET,
    VALID_CODES,
    OghamFutharkAlphabet,
    Symbol,
)


def hamming_distance(code1: int, code2: int) -> int:
    """Count differing bits between two 6-bit codes."""
    return bin(code1 ^ code2).count("1")


def is_valid_code(code: int) -> bool:
    """Check if a 6-bit code maps to a valid symbol."""
    return code in VALID_CODES


def group_bits_valid(code: int) -> bool:
    """Check whether the system+group bits (bits 3-5) form a valid group.

    Valid combinations:
        Futhark (system=0): groups 0, 1, 2 (not 3)
        Ogham (system=1): groups 0, 1, 2, 3
    """
    system = (code >> 5) & 0b1
    group = (code >> 3) & 0b11
    if system == 0:  # Futhark
        return group <= 2
    else:  # Ogham
        return True  # all 4 groups valid


def candidates_in_group(code: int) -> List[Symbol]:
    """Find all valid symbols sharing the same system+group bits."""
    system_group = (code >> 3) & 0b111
    return [s for s in ALPHABET if ((s.code >> 3) & 0b111) == system_group]


def correct_symbol(
    corrupted_code: int,
    confidence_threshold: int = 2,
) -> Optional[Symbol]:
    """Attempt to correct a corrupted 6-bit code using semantic priors.

    Strategy:
        1. Extract system + group bits (assume they're more robust).
        2. Find nearest valid symbol within that group by Hamming distance.
        3. Accept correction if distance <= threshold.

    Args:
        corrupted_code: Potentially corrupted 6-bit code.
        confidence_threshold: Maximum Hamming distance to accept correction.

    Returns:
        Corrected Symbol if recoverable, None if unrecoverable.
    """
    # Fast path: code is already valid
    if is_valid_code(corrupted_code):
        return OghamFutharkAlphabet.decode(corrupted_code)

    # Try group-constrained correction first
    if group_bits_valid(corrupted_code):
        candidates = candidates_in_group(corrupted_code)
        if candidates:
            best = min(candidates, key=lambda s: hamming_distance(s.code, corrupted_code))
            dist = hamming_distance(best.code, corrupted_code)
            if dist <= confidence_threshold:
                return best

    # Fallback: search all symbols
    best = min(ALPHABET, key=lambda s: hamming_distance(s.code, corrupted_code))
    dist = hamming_distance(best.code, corrupted_code)
    if dist <= confidence_threshold:
        return best

    return None


def correct_with_context(
    corrupted_code: int,
    left_neighbor: Optional[int] = None,
    right_neighbor: Optional[int] = None,
    confidence_threshold: int = 3,
) -> Optional[Symbol]:
    """Correct a corrupted code using group context and neighbor information.

    Uses the observation that symbols near each other in a sequence
    often belong to the same or related groups, providing additional
    disambiguation.

    Args:
        corrupted_code: Potentially corrupted 6-bit code.
        left_neighbor: Code of the preceding symbol (if available).
        right_neighbor: Code of the following symbol (if available).
        confidence_threshold: Maximum effective distance to accept.

    Returns:
        Corrected Symbol or None.
    """
    candidates = list(ALPHABET)

    # Score each candidate
    scored: List[Tuple[float, Symbol]] = []
    for sym in candidates:
        # Hamming distance (lower is better)
        h_dist = hamming_distance(sym.code, corrupted_code)

        # Group continuity bonus (neighbors in same group get score boost)
        group_bonus = 0.0
        for neighbor_code in [left_neighbor, right_neighbor]:
            if neighbor_code is not None and is_valid_code(neighbor_code):
                neighbor = OghamFutharkAlphabet.decode(neighbor_code)
                if neighbor.system == sym.system and neighbor.group == sym.group:
                    group_bonus -= 0.5  # Reduce effective distance

        effective_distance = h_dist + group_bonus
        scored.append((effective_distance, sym))

    scored.sort(key=lambda x: x[0])

    if scored and scored[0][0] <= confidence_threshold:
        return scored[0][1]

    return None


def repair_sequence(
    codes: List[int],
    confidence_threshold: int = 2,
) -> List[Optional[Symbol]]:
    """Repair a sequence of potentially corrupted codes.

    Uses both individual Hamming-distance correction and sequence
    context from neighboring symbols.

    Args:
        codes: List of (potentially corrupted) 6-bit codes.
        confidence_threshold: Maximum Hamming distance per symbol.

    Returns:
        List of corrected Symbols (None for unrecoverable positions).
    """
    repaired: List[Optional[Symbol]] = []

    for i, code in enumerate(codes):
        # Try context-aware correction
        left = codes[i - 1] if i > 0 else None
        right = codes[i + 1] if i < len(codes) - 1 else None
        corrected = correct_with_context(code, left, right, confidence_threshold)
        repaired.append(corrected)

    return repaired


def simulate_corruption(code: int, num_bit_flips: int = 1) -> int:
    """Simulate random bit corruption on a 6-bit code (for testing).

    Args:
        code: Original 6-bit code.
        num_bit_flips: Number of random bits to flip.

    Returns:
        Corrupted code.
    """
    import random
    corrupted = code
    bits_to_flip = random.sample(range(6), min(num_bit_flips, 6))
    for bit in bits_to_flip:
        corrupted ^= (1 << bit)
    return corrupted & 0b111111
