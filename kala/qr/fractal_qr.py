"""
Fractal-to-QR code mapping adapter.

Maps fractal cell trees to QR code module matrices:
    - Root cell encodes at QR center timing pattern
    - Children radiate at phi-scaled distances along branch angles
    - Each 6-bit code maps to a 2x3 QR module pattern
    - Aettir groups map to error correction regions
    - Version 40 QR codes (177x177 modules) provide maximum fractal depth

This is an auxiliary adapter -- not on the critical path for NN training.

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under the Apache License, Version 2.0
"""

import math
from typing import List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    raise ImportError("numpy is required for kala.qr.fractal_qr")

from kala.fractal.geometry import PHI, GoldenRatioGeometry
from kala.fractal.tree import FractalCell, FractalTree


# QR Code version sizes (modules per side)
QR_VERSION_SIZES = {v: 17 + 4 * v for v in range(1, 41)}


class FractalQREncoder:
    """Maps fractal cell trees to QR code module matrices.

    The encoder places 6-bit symbol patterns onto a 2D module grid
    following the fractal tree's geometric layout. This creates a
    machine-readable QR code that preserves the fractal structure.

    Attributes:
        version: QR code version (1-40).
        modules: Number of modules per side.
    """

    def __init__(self, version: int = 40):
        if version < 1 or version > 40:
            raise ValueError(f"QR version must be 1-40, got {version}")
        self.version = version
        self.modules = QR_VERSION_SIZES[version]

    def encode_6bit_pattern(self, code: int) -> np.ndarray:
        """Convert a 6-bit code to a 2x3 QR module pattern.

        The 6 bits are arranged in a 2-row by 3-column grid:
            Row 0: [bit5, bit4, bit3]
            Row 1: [bit2, bit1, bit0]

        Args:
            code: 6-bit symbol code (0-63).

        Returns:
            2x3 numpy array of 0/1 values.
        """
        pattern = np.array([
            [(code >> 5) & 1, (code >> 4) & 1, (code >> 3) & 1],
            [(code >> 2) & 1, (code >> 1) & 1, code & 1],
        ], dtype=np.int8)
        return pattern

    def encode_fractal_tree(
        self,
        root: FractalCell,
        max_depth: Optional[int] = None,
    ) -> np.ndarray:
        """Map a fractal cell tree to a QR code module matrix.

        Places each cell's 2x3 pattern at positions determined by
        the fractal geometry (Golden Ratio scaling and angular spacing).

        Args:
            root: Root FractalCell of the tree.
            max_depth: Maximum depth to encode (None = all).

        Returns:
            2D numpy array (modules x modules) with encoded patterns.
        """
        if max_depth is None:
            max_depth = root.max_depth()

        matrix = np.zeros((self.modules, self.modules), dtype=np.int8)
        center = self.modules // 2

        # Track which positions have been written (avoid overlap)
        occupied = np.zeros((self.modules, self.modules), dtype=bool)

        self._recursive_encode(
            root, matrix, occupied,
            center, center,
            self.modules / 4.0,
            max_depth,
        )

        return matrix

    def _recursive_encode(
        self,
        cell: FractalCell,
        matrix: np.ndarray,
        occupied: np.ndarray,
        x: int,
        y: int,
        scale: float,
        max_depth: int,
    ) -> None:
        """Recursively place cell patterns on the QR matrix."""
        # Place 2x3 pattern at (x, y) if within bounds and unoccupied
        pattern = self.encode_6bit_pattern(cell.code)
        self._place_pattern(matrix, occupied, pattern, x, y)

        if cell.depth >= max_depth:
            return

        # Place children at phi-scaled distances along branch angles
        child_scale = scale / PHI

        for i, child in enumerate(cell.children):
            angle_deg = cell.angle + GoldenRatioGeometry.branch_angle(i)
            angle_rad = math.radians(angle_deg)
            dx = int(child_scale * math.cos(angle_rad))
            dy = int(child_scale * math.sin(angle_rad))

            child_x = x + dx
            child_y = y + dy

            # Clamp to matrix bounds
            child_x = max(0, min(child_x, self.modules - 3))
            child_y = max(0, min(child_y, self.modules - 2))

            self._recursive_encode(
                child, matrix, occupied,
                child_x, child_y,
                child_scale, max_depth,
            )

    def _place_pattern(
        self,
        matrix: np.ndarray,
        occupied: np.ndarray,
        pattern: np.ndarray,
        x: int,
        y: int,
    ) -> bool:
        """Place a 2x3 pattern on the matrix if space is available.

        Returns True if placed, False if position was occupied or out of bounds.
        """
        rows, cols = pattern.shape  # 2, 3

        # Check bounds
        if y + rows > self.modules or x + cols > self.modules:
            return False
        if x < 0 or y < 0:
            return False

        # Check occupancy
        if occupied[y:y + rows, x:x + cols].any():
            return False

        matrix[y:y + rows, x:x + cols] = pattern
        occupied[y:y + rows, x:x + cols] = True
        return True

    def decode_pattern(self, pattern: np.ndarray) -> int:
        """Decode a 2x3 module pattern back to a 6-bit code.

        Args:
            pattern: 2x3 numpy array.

        Returns:
            6-bit integer code.
        """
        code = 0
        code |= int(pattern[0, 0]) << 5
        code |= int(pattern[0, 1]) << 4
        code |= int(pattern[0, 2]) << 3
        code |= int(pattern[1, 0]) << 2
        code |= int(pattern[1, 1]) << 1
        code |= int(pattern[1, 2])
        return code

    def matrix_to_cells(
        self,
        matrix: np.ndarray,
        positions: List[Tuple[int, int]],
    ) -> List[int]:
        """Extract 6-bit codes from known positions in a QR matrix.

        Args:
            matrix: QR module matrix.
            positions: List of (x, y) positions of 2x3 patterns.

        Returns:
            List of decoded 6-bit codes.
        """
        codes = []
        for x, y in positions:
            if y + 2 <= self.modules and x + 3 <= self.modules:
                pattern = matrix[y:y + 2, x:x + 3]
                codes.append(self.decode_pattern(pattern))
        return codes
