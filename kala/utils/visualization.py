"""
Visualization utilities for fractal trees and QR matrices.

Provides text-based and data-export visualization methods.
Matplotlib-based plotting is optional (guarded by import check).

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under the Apache License, Version 2.0
"""

from typing import List, Optional, TextIO
import sys

from kala.fractal.tree import FractalCell, FractalTree

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def print_tree(
    root: FractalCell,
    indent: str = "",
    last: bool = True,
    file: TextIO = sys.stdout,
    max_depth: Optional[int] = None,
) -> None:
    """Print a fractal tree as an ASCII tree diagram.

    Args:
        root: Root FractalCell.
        indent: Current indentation (used in recursion).
        last: Whether this node is the last sibling.
        file: Output stream.
        max_depth: Maximum depth to display.
    """
    connector = "└── " if last else "├── "
    sym = root.symbol
    name = sym.name if sym else f"[{root.code:06b}]"
    group_info = f"g{root.group_bits}" if sym else ""

    print(
        f"{indent}{connector}{name} (d={root.depth}, {group_info}, "
        f"a={root.angle:.1f}°)",
        file=file,
    )

    if max_depth is not None and root.depth >= max_depth:
        if root.children:
            child_indent = indent + ("    " if last else "│   ")
            print(f"{child_indent}└── ... ({len(root.children)} children)", file=file)
        return

    child_indent = indent + ("    " if last else "│   ")
    for i, child in enumerate(root.children):
        is_last = (i == len(root.children) - 1)
        print_tree(child, child_indent, is_last, file, max_depth)


def tree_summary(root: FractalCell) -> str:
    """Generate a summary string of tree statistics.

    Returns:
        Multi-line summary with node count, depth, symbol distribution.
    """
    all_cells = list(FractalTree.walk_depth_first(root))
    total = len(all_cells)
    max_d = root.max_depth()
    leaves = sum(1 for c in all_cells if c.is_leaf)

    # Group distribution
    group_counts: dict = {}
    system_counts = {"Futhark": 0, "Ogham": 0}
    for cell in all_cells:
        sym = cell.symbol
        if sym:
            key = f"{sym.system.name}:g{sym.group}"
            group_counts[key] = group_counts.get(key, 0) + 1
            system_counts[sym.system.name] = system_counts.get(sym.system.name, 0) + 1

    lines = [
        f"Fractal Tree Summary",
        f"  Total nodes:  {total}",
        f"  Max depth:    {max_d}",
        f"  Leaf nodes:   {leaves}",
        f"  Systems:      {system_counts}",
        f"  Groups:       {group_counts}",
    ]
    return "\n".join(lines)


def export_tree_data(root: FractalCell) -> List[dict]:
    """Export tree data as a list of dictionaries (for JSON/CSV).

    Each dict contains: code, depth, angle, x, y, parent_code, symbol_name.
    """
    data = []
    for cell in FractalTree.walk_depth_first(root):
        sym = cell.symbol
        data.append({
            "code": cell.code,
            "code_binary": f"{cell.code:06b}",
            "depth": cell.depth,
            "angle": round(cell.angle, 2),
            "x": round(cell.position[0], 4),
            "y": round(cell.position[1], 4),
            "parent_code": cell.parent.code if cell.parent else None,
            "symbol_name": sym.name if sym else None,
            "system": sym.system.name if sym else None,
            "group": sym.group if sym else None,
            "position": sym.position if sym else None,
            "unicode": sym.unicode_char if sym else None,
            "is_leaf": cell.is_leaf,
            "num_children": cell.num_children,
        })
    return data


if HAS_MATPLOTLIB and HAS_NUMPY:

    def plot_fractal_tree(
        root: FractalCell,
        figsize: tuple = (12, 12),
        max_depth: Optional[int] = None,
        show_labels: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot a fractal tree using matplotlib.

        Nodes are placed at their geometric positions.
        Edges connect parent-child pairs. Node color indicates group.

        Args:
            root: Root FractalCell.
            figsize: Figure size in inches.
            max_depth: Maximum depth to display.
            show_labels: Whether to show symbol names.
            save_path: Path to save the figure (optional).
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Color map: system*4 + group -> color
        colors = [
            "#e74c3c", "#3498db", "#2ecc71", "#f39c12",  # Futhark groups 0-3
            "#9b59b6", "#1abc9c", "#e67e22", "#34495e",  # Ogham groups 0-3
        ]

        def _plot_cell(cell: FractalCell) -> None:
            if max_depth is not None and cell.depth > max_depth:
                return

            x, y = cell.position
            color_idx = cell.system_bit * 4 + cell.group_bits
            color = colors[color_idx % len(colors)]
            size = max(20, 200 / (cell.depth + 1))

            ax.scatter(x, y, c=color, s=size, zorder=3, edgecolors="black", linewidth=0.5)

            if show_labels and cell.depth <= 4:
                sym = cell.symbol
                label = sym.unicode_char if sym else f"{cell.code}"
                ax.annotate(
                    label, (x, y),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=max(6, 10 - cell.depth),
                )

            for child in cell.children:
                if max_depth is None or child.depth <= max_depth:
                    cx, cy = child.position
                    ax.plot(
                        [x, cx], [y, cy],
                        color="gray", linewidth=max(0.3, 1.5 - 0.2 * cell.depth),
                        alpha=0.5, zorder=1,
                    )
                    _plot_cell(child)

        _plot_cell(root)
        ax.set_aspect("equal")
        ax.set_title("Fractal Memory Tree", fontsize=14)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_qr_matrix(
        matrix: "np.ndarray",
        figsize: tuple = (8, 8),
        title: str = "Fractal QR Code",
        save_path: Optional[str] = None,
    ) -> None:
        """Plot a QR module matrix.

        Args:
            matrix: 2D numpy array from FractalQREncoder.
            figsize: Figure size.
            title: Plot title.
            save_path: Path to save figure.
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(matrix, cmap="binary", interpolation="nearest")
        ax.set_title(title, fontsize=14)
        ax.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
