"""
PyTorch modules for the fractal memory neural network.

FractalEmbedding converts 6-bit codes + depth indices into dense embeddings.
FractalMemoryNetwork stacks fractal attention layers to process the
memory tree structure.

The fractal memory is a pluggable component, enabled via config:
    memory.backend = "baseline" | "fractal_runic"

Initially treats the fractal tree as a flattened sequence with auxiliary
depth/angle features; upgrades to true tree-attention later.

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under the Apache License, Version 2.0
"""

from typing import List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError(
        "PyTorch is required for kala.models.fractal_memory. "
        "Install with: pip install torch"
    )

from .attention import FractalAttentionLayer
from kala.fractal.alphabet import NUM_SYMBOLS, VOCAB_SIZE
from kala.fractal.tree import FractalCell, FractalTree


class FractalEmbedding(nn.Module):
    """Embedding layer for fractal memory cells.

    Combines:
        - Symbol code embedding (6-bit code -> d_model)
        - Depth embedding (fractal depth level -> d_model)
        - Optional angle embedding via sinusoidal encoding

    Attributes:
        vocab_size: Number of possible 6-bit codes (64).
        d_model: Embedding dimension.
        max_depth: Maximum supported fractal depth.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        max_depth: int = 16,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_depth = max_depth

        self.code_embedding = nn.Embedding(vocab_size, d_model)
        self.depth_embedding = nn.Embedding(max_depth, d_model)

    def forward(
        self,
        codes: torch.LongTensor,
        depths: torch.LongTensor,
    ) -> torch.Tensor:
        """Embed fractal cell codes with depth information.

        Args:
            codes: [batch, seq_len] 6-bit symbol codes (0-63).
            depths: [batch, seq_len] depth of each cell in fractal tree.

        Returns:
            [batch, seq_len, d_model] combined embeddings.
        """
        code_emb = self.code_embedding(codes)
        depth_emb = self.depth_embedding(depths.clamp(max=self.max_depth - 1))
        return code_emb + depth_emb


class FractalMemoryNetwork(nn.Module):
    """Neural network operating on fractal tree structures.

    Stacks multiple FractalAttentionLayers with fractal geometry bias
    to process flattened tree representations. The network learns to
    leverage depth, group, and angular relationships between cells.

    Architecture:
        Input (codes + depths) -> FractalEmbedding -> N x FractalAttentionLayer -> Output projection

    Attributes:
        vocab_size: Number of symbols in the output space.
        d_model: Hidden dimension.
        n_heads: Attention heads per layer.
        n_layers: Number of fractal attention layers.
        max_depth: Maximum fractal depth supported.
    """

    def __init__(
        self,
        vocab_size: int = NUM_SYMBOLS,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_depth: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embed = FractalEmbedding(
            vocab_size=VOCAB_SIZE,  # 64 possible 6-bit codes
            d_model=d_model,
            max_depth=max_depth,
        )

        self.layers = nn.ModuleList([
            FractalAttentionLayer(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        codes: torch.LongTensor,
        depths: torch.LongTensor,
        groups: Optional[torch.LongTensor] = None,
        angles: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process a flattened fractal tree through the network.

        Args:
            codes: [batch, seq_len] 6-bit symbol codes.
            depths: [batch, seq_len] depth indices.
            groups: [batch, seq_len] group indices (optional).
            angles: [batch, seq_len] stemline angles (optional).
            mask: [batch, seq_len] padding mask (True = padded).

        Returns:
            [batch, seq_len, vocab_size] output logits.
        """
        x = self.embed(codes, depths)

        for layer in self.layers:
            x = layer(x, depths, groups, angles, mask)

        logits = self.output_proj(x)
        return logits

    @staticmethod
    def prepare_tree_input(
        root: FractalCell,
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.Tensor]:
        """Convert a FractalCell tree to tensor inputs.

        Args:
            root: Root of the fractal tree.

        Returns:
            Tuple of (codes, depths, groups, angles) tensors,
            each with shape [1, seq_len] (single-item batch).
        """
        codes_list, depths_list, angles_list = FractalTree.flatten(root)

        # Extract group bits from codes
        groups_list = [(c >> 3) & 0b11 for c in codes_list]

        codes = torch.tensor([codes_list], dtype=torch.long)
        depths = torch.tensor([depths_list], dtype=torch.long)
        groups = torch.tensor([groups_list], dtype=torch.long)
        angles = torch.tensor([angles_list], dtype=torch.float)

        return codes, depths, groups, angles

    @staticmethod
    def collate_trees(
        trees: List[FractalCell],
    ) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.Tensor, torch.Tensor]:
        """Collate multiple trees into a padded batch.

        Args:
            trees: List of FractalCell roots.

        Returns:
            Tuple of (codes, depths, groups, angles, mask) tensors.
        """
        all_codes = []
        all_depths = []
        all_groups = []
        all_angles = []
        max_len = 0

        for root in trees:
            codes, depths, angles = FractalTree.flatten(root)
            groups = [(c >> 3) & 0b11 for c in codes]
            all_codes.append(codes)
            all_depths.append(depths)
            all_groups.append(groups)
            all_angles.append(angles)
            max_len = max(max_len, len(codes))

        # Pad to max length
        batch_codes = []
        batch_depths = []
        batch_groups = []
        batch_angles = []
        batch_mask = []

        for i in range(len(trees)):
            pad_len = max_len - len(all_codes[i])
            batch_codes.append(all_codes[i] + [0] * pad_len)
            batch_depths.append(all_depths[i] + [0] * pad_len)
            batch_groups.append(all_groups[i] + [0] * pad_len)
            batch_angles.append(all_angles[i] + [0.0] * pad_len)
            batch_mask.append([False] * len(all_codes[i]) + [True] * pad_len)

        return (
            torch.tensor(batch_codes, dtype=torch.long),
            torch.tensor(batch_depths, dtype=torch.long),
            torch.tensor(batch_groups, dtype=torch.long),
            torch.tensor(batch_angles, dtype=torch.float),
            torch.tensor(batch_mask, dtype=torch.bool),
        )
