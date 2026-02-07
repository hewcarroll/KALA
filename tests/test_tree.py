"""Tests for kala.fractal.tree -- FractalCell and FractalTree operations."""

import pytest

from kala.fractal.alphabet import OghamFutharkAlphabet
from kala.fractal.tree import FractalCell, FractalTree


class TestFractalCell:
    """Test basic FractalCell operations."""

    def test_create_root(self):
        cell = FractalCell(code=0)  # Fehu
        assert cell.depth == 0
        assert cell.is_root
        assert cell.is_leaf
        assert cell.num_children == 0

    def test_symbol_property(self):
        cell = FractalCell(code=0)  # Fehu
        assert cell.symbol is not None
        assert cell.symbol.name == "Fehu"

    def test_invalid_code_symbol(self):
        cell = FractalCell(code=24)  # invalid
        assert cell.symbol is None

    def test_add_child(self):
        root = FractalCell(code=0)
        child = root.add_child(1)  # Uruz
        assert root.num_children == 1
        assert not root.is_leaf
        assert child.depth == 1
        assert child.parent is root

    def test_child_position_nonzero(self):
        root = FractalCell(code=0)
        child = root.add_child(1)
        assert child.position != (0.0, 0.0)

    def test_to_path(self):
        root = FractalCell(code=0)
        c1 = root.add_child(1)
        c2 = c1.add_child(2)
        assert c2.to_path() == [0, 1, 2]

    def test_subtree_size(self):
        root = FractalCell(code=0)
        root.add_child(1)
        root.add_child(2)
        assert root.subtree_size() == 3

    def test_max_depth(self):
        root = FractalCell(code=0)
        c1 = root.add_child(1)
        c1.add_child(2)
        assert root.max_depth() == 2

    def test_bit_properties(self):
        cell = FractalCell(code=0b101001)
        assert cell.system_bit == 1
        assert cell.group_bits == 1
        assert cell.position_bits == 1


class TestBuildLinear:
    """Test linear tree construction."""

    def test_build_linear(self):
        codes = [0, 1, 2, 3]
        root = FractalTree.build_linear(codes)
        assert root.code == 0
        assert root.depth == 0
        assert root.subtree_size() == 4
        assert root.max_depth() == 3

    def test_linear_chain(self):
        codes = [0, 8, 16]
        root = FractalTree.build_linear(codes)
        # Root -> Child -> Grandchild
        assert root.num_children == 1
        assert root.children[0].num_children == 1
        assert root.children[0].children[0].is_leaf

    def test_empty_codes_raises(self):
        with pytest.raises(ValueError):
            FractalTree.build_linear([])

    def test_single_code(self):
        root = FractalTree.build_linear([0])
        assert root.is_leaf
        assert root.subtree_size() == 1


class TestBuildBranching:
    """Test branching tree construction."""

    def test_build_branching(self):
        codes = [0, 1, 2, 3, 4, 5, 6]
        root = FractalTree.build_branching(codes, branching_factor=2)
        assert root.code == 0
        assert root.num_children == 2
        assert root.subtree_size() == 7

    def test_branching_factor_3(self):
        codes = [0, 1, 2, 3]
        root = FractalTree.build_branching(codes, branching_factor=3)
        assert root.num_children == 3


class TestTraversal:
    """Test tree traversal methods."""

    def test_depth_first(self):
        codes = [0, 1, 2, 3, 4, 5, 6]
        root = FractalTree.build_branching(codes, branching_factor=2)
        visited = [c.code for c in FractalTree.walk_depth_first(root)]
        assert visited[0] == 0  # root first
        assert len(visited) == 7

    def test_breadth_first(self):
        codes = [0, 1, 2, 3, 4, 5, 6]
        root = FractalTree.build_branching(codes, branching_factor=2)
        visited = [c.code for c in FractalTree.walk_breadth_first(root)]
        assert visited[0] == 0
        assert visited[1] == 1
        assert visited[2] == 2
        assert len(visited) == 7

    def test_walk_paths(self):
        root = FractalCell(code=0)
        root.add_child(1)
        root.add_child(2)
        paths = FractalTree.walk_paths(root)
        assert len(paths) == 2
        assert len(paths[0]) == 2
        assert len(paths[1]) == 2

    def test_walk_paths_max_depth(self):
        codes = [0, 1, 2, 3]
        root = FractalTree.build_linear(codes)
        paths = FractalTree.walk_paths(root, max_depth=1)
        assert len(paths) == 1
        assert len(paths[0]) == 2  # root + depth-1 child

    def test_cells_at_depth(self):
        codes = [0, 1, 2, 3, 4, 5, 6]
        root = FractalTree.build_branching(codes, branching_factor=2)
        depth_1 = FractalTree.get_cells_at_depth(root, 1)
        assert len(depth_1) == 2


class TestFlatten:
    """Test tree flattening for tensor conversion."""

    def test_flatten_structure(self):
        root = FractalCell(code=0)
        root.add_child(1)
        root.add_child(8)
        codes, depths, angles = FractalTree.flatten(root)
        assert len(codes) == 3
        assert len(depths) == 3
        assert len(angles) == 3
        assert codes[0] == 0
        assert depths[0] == 0

    def test_flatten_depths_correct(self):
        codes = [0, 1, 2, 3]
        root = FractalTree.build_linear(codes)
        _, depths, _ = FractalTree.flatten(root)
        assert depths == [0, 1, 2, 3]


class TestFromSymbols:
    """Test building trees from Symbol objects."""

    def test_from_symbols_linear(self):
        symbols = OghamFutharkAlphabet.all_symbols()[:5]
        root = FractalTree.from_symbols(symbols, branching_factor=1)
        assert root.subtree_size() == 5
        assert root.max_depth() == 4

    def test_from_symbols_branching(self):
        symbols = OghamFutharkAlphabet.all_symbols()[:7]
        root = FractalTree.from_symbols(symbols, branching_factor=2)
        assert root.subtree_size() == 7
