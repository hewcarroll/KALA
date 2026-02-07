"""
Golden Ratio geometry for fractal branching.

The fractal memory tree uses the Golden Ratio (phi) to govern:
- Branch angles (golden angle ~137.5 degrees for optimal packing)
- Scale factors (phi^(-depth) for self-similar scaling)
- Stemline positions (polar coordinates from parent nodes)

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under the Apache License, Version 2.0
"""

import math
from typing import Tuple

# Golden Ratio
PHI: float = 1.618033988749895

# Golden Angle in degrees: 360 / phi^2 ≈ 137.508 degrees
GOLDEN_ANGLE_DEG: float = 360.0 / (PHI * PHI)

# Golden Angle in radians
GOLDEN_ANGLE_RAD: float = math.radians(GOLDEN_ANGLE_DEG)


class GoldenRatioGeometry:
    """Geometric calculations for fractal tree layout using the Golden Ratio."""

    @staticmethod
    def golden_angle_degrees() -> float:
        """Return the golden angle in degrees (~137.508)."""
        return GOLDEN_ANGLE_DEG

    @staticmethod
    def golden_angle_radians() -> float:
        """Return the golden angle in radians."""
        return GOLDEN_ANGLE_RAD

    @staticmethod
    def branch_angle(child_index: int) -> float:
        """Calculate branch angle for the i-th child using golden spiral spacing.

        Args:
            child_index: Zero-based index of the child among siblings.

        Returns:
            Angle in degrees.
        """
        return (child_index * GOLDEN_ANGLE_DEG) % 360.0

    @staticmethod
    def scale_factor(depth: int) -> float:
        """Branch length scaling by depth using phi.

        Each depth level scales by 1/phi, producing self-similar branching.

        Args:
            depth: Depth level in the fractal tree (0 = root).

        Returns:
            Scale factor (1.0 at depth 0, ~0.618 at depth 1, etc.)
        """
        return PHI ** (-depth)

    @staticmethod
    def stemline_position(
        depth: int,
        angle_deg: float,
        parent_pos: Tuple[float, float] = (0.0, 0.0),
    ) -> Tuple[float, float]:
        """Calculate (x, y) position of a stemline at given depth and angle.

        Args:
            depth: Depth in the fractal tree.
            angle_deg: Branch angle in degrees.
            parent_pos: (x, y) of the parent node.

        Returns:
            (x, y) position tuple.
        """
        scale = GoldenRatioGeometry.scale_factor(depth)
        angle_rad = math.radians(angle_deg)
        x = parent_pos[0] + scale * math.cos(angle_rad)
        y = parent_pos[1] + scale * math.sin(angle_rad)
        return (x, y)

    @staticmethod
    def rune_branch_positions(phi: float = PHI) -> dict:
        """Branch positions for a Futhark rune (above/center/below stemline).

        Returns:
            Dictionary with 'above', 'center', 'below' offsets.
        """
        return {
            "above": -phi,
            "center": 0.0,
            "below": phi,
        }

    @staticmethod
    def ogham_branch_positions(phi: float = PHI) -> dict:
        """Branch positions for an Ogham character (left/stemline/right).

        Returns:
            Dictionary with 'left', 'stemline', 'right' offsets.
        """
        return {
            "left": -phi,
            "stemline": 0.0,
            "right": phi,
        }

    @staticmethod
    def distance(
        pos_a: Tuple[float, float], pos_b: Tuple[float, float]
    ) -> float:
        """Euclidean distance between two positions."""
        dx = pos_a[0] - pos_b[0]
        dy = pos_a[1] - pos_b[1]
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def angular_distance(angle_a: float, angle_b: float) -> float:
        """Smallest angular distance between two angles in degrees.

        Returns a value in [0, 180].
        """
        diff = abs(angle_a - angle_b) % 360.0
        return min(diff, 360.0 - diff)

    @staticmethod
    def angular_similarity(angle_a: float, angle_b: float) -> float:
        """Cosine-based angular similarity in [-1, 1].

        1.0 = same angle, -1.0 = opposite angles.
        """
        diff_deg = GoldenRatioGeometry.angular_distance(angle_a, angle_b)
        return math.cos(math.radians(diff_deg))
