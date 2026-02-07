"""
Fractal tree with recursive nesting and Golden Ratio geometry.

Each FractalCell represents a node in the fractal memory tree, encoding
a symbol on a stemline that can branch into children at deeper levels.
The tree structure mirrors the Yggdrasil memory mapping.

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under the Apache License, Version 2.0
"""

from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Tuple

from .alphabet import OghamFutharkAlphabet, Symbol
from .geometry import GOLDEN_ANGLE_DEG, GoldenRatioGeometry


@dataclass
class FractalCell:
    """A single node in the fractal memory tree.

    Each cell encodes a 6-bit symbol and can recursively contain children,
    forming a fractal structure governed by Golden Ratio geometry.

    Attributes:
        code: 6-bit canonical symbol code.
        depth: Depth level in the tree (0 = root).
        angle: Stemline orientation in degrees.
        position: (x, y) position computed from Golden Ratio scaling.
        parent: Reference to parent cell (None for root).
        children: List of child FractalCells.
    """
    code: int
    depth: int = 0
    angle: float = 0.0
    position: Tuple[float, float] = (0.0, 0.0)
    parent: Optional["FractalCell"] = field(default=None, repr=False)
    children: List["FractalCell"] = field(default_factory=list, repr=False)

    @property
    def symbol(self) -> Optional[Symbol]:
        """Decode this cell's code to a Symbol, or None if invalid."""
        try:
            return OghamFutharkAlphabet.decode(self.code)
        except KeyError:
            return None

    @property
    def system_bit(self) -> int:
        return (self.code >> 5) & 0b1

    @property
    def group_bits(self) -> int:
        return (self.code >> 3) & 0b11

    @property
    def position_bits(self) -> int:
        return self.code & 0b111

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def num_children(self) -> int:
        return len(self.children)

    def add_child(self, child_code: int, child_index: Optional[int] = None) -> "FractalCell":
        """Add a child cell with Golden Ratio geometric positioning.

        Args:
            child_code: 6-bit code for the child symbol.
            child_index: Optional override for sibling index (defaults to
                current number of children).

        Returns:
            The newly created child FractalCell.
        """
        if child_index is None:
            child_index = len(self.children)

        child_angle = (self.angle + GoldenRatioGeometry.branch_angle(child_index)) % 360.0
        child_pos = GoldenRatioGeometry.stemline_position(
            self.depth + 1, child_angle, self.position
        )

        child = FractalCell(
            code=child_code,
            depth=self.depth + 1,
            angle=child_angle,
            position=child_pos,
            parent=self,
        )
        self.children.append(child)
        return child

    def to_path(self) -> List[int]:
        """Return the path from root to this cell as a list of codes."""
        path = []
        current: Optional[FractalCell] = self
        while current is not None:
            path.append(current.code)
            current = current.parent
        return list(reversed(path))

    def subtree_size(self) -> int:
        """Count the total number of nodes in this cell's subtree (including self)."""
        return 1 + sum(child.subtree_size() for child in self.children)

    def max_depth(self) -> int:
        """Return the maximum depth in this cell's subtree."""
        if not self.children:
            return self.depth
        return max(child.max_depth() for child in self.children)

    def __repr__(self) -> str:
        sym = self.symbol
        name = sym.name if sym else f"code={self.code}"
        return f"FractalCell({name}, depth={self.depth}, children={len(self.children)})"


class FractalTree:
    """Operations on fractal cell trees."""

    @staticmethod
    def build_linear(codes: List[int]) -> FractalCell:
        """Build a simple linear fractal tree from a sequence of codes.

        Each symbol becomes a child of the previous one, forming a
        chain from root to leaf.

        Args:
            codes: List of 6-bit symbol codes.

        Returns:
            Root FractalCell.

        Raises:
            ValueError: If codes list is empty.
        """
        if not codes:
            raise ValueError("Cannot build tree from empty code list")

        root = FractalCell(code=codes[0])
        current = root
        for code in codes[1:]:
            current = current.add_child(code, 0)
        return root

    @staticmethod
    def build_branching(codes: List[int], branching_factor: int = 2) -> FractalCell:
        """Build a balanced branching tree from a sequence of codes.

        Distributes codes across branches up to the given branching factor.

        Args:
            codes: List of 6-bit symbol codes.
            branching_factor: Maximum children per node.

        Returns:
            Root FractalCell.
        """
        if not codes:
            raise ValueError("Cannot build tree from empty code list")

        root = FractalCell(code=codes[0])
        queue = [root]
        idx = 1

        while idx < len(codes) and queue:
            parent = queue.pop(0)
            for b in range(branching_factor):
                if idx >= len(codes):
                    break
                child = parent.add_child(codes[idx], b)
                queue.append(child)
                idx += 1

        return root

    @staticmethod
    def walk_depth_first(root: FractalCell) -> Iterator[FractalCell]:
        """Depth-first traversal of the tree."""
        yield root
        for child in root.children:
            yield from FractalTree.walk_depth_first(child)

    @staticmethod
    def walk_breadth_first(root: FractalCell) -> Iterator[FractalCell]:
        """Breadth-first traversal of the tree."""
        queue = [root]
        while queue:
            node = queue.pop(0)
            yield node
            queue.extend(node.children)

    @staticmethod
    def walk_paths(
        root: FractalCell, max_depth: Optional[int] = None
    ) -> List[List[FractalCell]]:
        """Enumerate all root-to-leaf paths.

        Args:
            root: Root of the tree.
            max_depth: If set, treat nodes at this depth as leaves.

        Returns:
            List of paths, each path is a list of FractalCells.
        """
        paths: List[List[FractalCell]] = []

        def _dfs(node: FractalCell, path: List[FractalCell]) -> None:
            path = path + [node]
            if max_depth is not None and node.depth >= max_depth:
                paths.append(path)
                return
            if not node.children:
                paths.append(path)
                return
            for child in node.children:
                _dfs(child, path)

        _dfs(root, [])
        return paths

    @staticmethod
    def get_cells_at_depth(root: FractalCell, target_depth: int) -> List[FractalCell]:
        """Return all cells at a specific depth level."""
        return [
            cell for cell in FractalTree.walk_breadth_first(root)
            if cell.depth == target_depth
        ]

    @staticmethod
    def flatten(root: FractalCell) -> Tuple[List[int], List[int], List[float]]:
        """Flatten a tree into parallel lists for tensor conversion.

        Returns:
            Tuple of (codes, depths, angles) lists.
        """
        codes: List[int] = []
        depths: List[int] = []
        angles: List[float] = []
        for cell in FractalTree.walk_depth_first(root):
            codes.append(cell.code)
            depths.append(cell.depth)
            angles.append(cell.angle)
        return codes, depths, angles

    @staticmethod
    def from_symbols(symbols: List[Symbol], branching_factor: int = 2) -> FractalCell:
        """Build a tree from Symbol objects."""
        codes = [s.code for s in symbols]
        if branching_factor == 1:
            return FractalTree.build_linear(codes)
        return FractalTree.build_branching(codes, branching_factor)
