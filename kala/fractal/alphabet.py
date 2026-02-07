"""
Elder Futhark (24 runes) + Ogham (20 characters) = 44 unified symbols.

6-bit encoding layout:
    [system_bit (1)][group_bits (2)][position_bits (3)]

    Bit 5:   System selector (0=Futhark, 1=Ogham)
    Bits 3-4: Group (Aett/Aicme index, 0-3)
    Bits 0-2: Position within group (0-7)

This yields 64 possible codewords for 44 symbols, with 20 reserved codes
for control tokens, error flags, or structural markers.

Shannon optimum for 44 symbols: log2(44) = 5.46 bits/char.
6-bit encoding overhead: 0.54 bits/char -- near-optimal.

Copyright 2026 Hew Carroll / The Saelix Institute
Licensed under the Apache License, Version 2.0
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional


class RuneSystem(IntEnum):
    """Writing system selector (bit 5)."""
    FUTHARK = 0
    OGHAM = 1


class FutharkAett(IntEnum):
    """Elder Futhark aettir groups (bits 3-4 when system=0)."""
    FREYRS = 0      # Freyr's Aett: fertility, cattle, gods
    HEIMDALLS = 1   # Heimdall's Aett: hail, need, ice
    TYRS = 2        # Tyr's Aett: Tyr, birth, day


class OghamAicme(IntEnum):
    """Ogham aicmi groups (bits 3-4 when system=1)."""
    BEITHE = 0   # B-group: Birch, Rowan, Alder, Willow, Ash
    HUATH = 1    # H-group: Hawthorn, Oak, Holly, Hazel, Apple
    MUIN = 2     # M-group: Vine, Ivy, Reed, Blackthorn, Elder
    AILM = 3     # A-group: Fir/Pine, Gorse, Heather, Aspen, Yew


@dataclass(frozen=True)
class Symbol:
    """A single symbol in the 44-character Ogham-Futhark alphabet.

    Attributes:
        name: Traditional name of the rune/character.
        system: Writing system (FUTHARK or OGHAM).
        group: Aett/aicme group index (0-3).
        position: Position within group (0-7).
        unicode_char: Unicode representation.
        meaning: Traditional meaning or association.
    """
    name: str
    system: RuneSystem
    group: int
    position: int
    unicode_char: str
    meaning: str = ""

    @property
    def code(self) -> int:
        """Compute the 6-bit canonical code."""
        return (self.system << 5) | (self.group << 3) | self.position

    @property
    def system_bit(self) -> int:
        return (self.code >> 5) & 0b1

    @property
    def group_bits(self) -> int:
        return (self.code >> 3) & 0b11

    @property
    def position_bits(self) -> int:
        return self.code & 0b111


# ---------------------------------------------------------------------------
# Elder Futhark (24 runes in 3 aettir of 8)
# ---------------------------------------------------------------------------

FUTHARK_RUNES: List[Symbol] = [
    # Freyr's Aett (group 0, positions 0-7)
    Symbol("Fehu",     RuneSystem.FUTHARK, FutharkAett.FREYRS, 0, "\u16A0", "wealth/cattle"),
    Symbol("Uruz",     RuneSystem.FUTHARK, FutharkAett.FREYRS, 1, "\u16A2", "aurochs/strength"),
    Symbol("Thurisaz", RuneSystem.FUTHARK, FutharkAett.FREYRS, 2, "\u16A6", "thorn/giant"),
    Symbol("Ansuz",    RuneSystem.FUTHARK, FutharkAett.FREYRS, 3, "\u16A8", "god/mouth"),
    Symbol("Raido",    RuneSystem.FUTHARK, FutharkAett.FREYRS, 4, "\u16B1", "ride/journey"),
    Symbol("Kenaz",    RuneSystem.FUTHARK, FutharkAett.FREYRS, 5, "\u16B2", "torch/knowledge"),
    Symbol("Gebo",     RuneSystem.FUTHARK, FutharkAett.FREYRS, 6, "\u16B7", "gift/exchange"),
    Symbol("Wunjo",    RuneSystem.FUTHARK, FutharkAett.FREYRS, 7, "\u16B9", "joy/perfection"),

    # Heimdall's Aett (group 1, positions 0-7)
    Symbol("Hagalaz",  RuneSystem.FUTHARK, FutharkAett.HEIMDALLS, 0, "\u16BA", "hail/disruption"),
    Symbol("Nauthiz",  RuneSystem.FUTHARK, FutharkAett.HEIMDALLS, 1, "\u16BE", "need/constraint"),
    Symbol("Isa",      RuneSystem.FUTHARK, FutharkAett.HEIMDALLS, 2, "\u16C1", "ice/stillness"),
    Symbol("Jera",     RuneSystem.FUTHARK, FutharkAett.HEIMDALLS, 3, "\u16C3", "year/harvest"),
    Symbol("Eihwaz",   RuneSystem.FUTHARK, FutharkAett.HEIMDALLS, 4, "\u16C7", "yew/endurance"),
    Symbol("Perthro",  RuneSystem.FUTHARK, FutharkAett.HEIMDALLS, 5, "\u16C8", "lot cup/mystery"),
    Symbol("Algiz",    RuneSystem.FUTHARK, FutharkAett.HEIMDALLS, 6, "\u16C9", "elk sedge/protection"),
    Symbol("Sowilo",   RuneSystem.FUTHARK, FutharkAett.HEIMDALLS, 7, "\u16CA", "sun/victory"),

    # Tyr's Aett (group 2, positions 0-7)
    Symbol("Tiwaz",    RuneSystem.FUTHARK, FutharkAett.TYRS, 0, "\u16CF", "Tyr/justice"),
    Symbol("Berkano",  RuneSystem.FUTHARK, FutharkAett.TYRS, 1, "\u16D2", "birch/renewal"),
    Symbol("Ehwaz",    RuneSystem.FUTHARK, FutharkAett.TYRS, 2, "\u16D6", "horse/partnership"),
    Symbol("Mannaz",   RuneSystem.FUTHARK, FutharkAett.TYRS, 3, "\u16D7", "man/humanity"),
    Symbol("Laguz",    RuneSystem.FUTHARK, FutharkAett.TYRS, 4, "\u16DA", "lake/water"),
    Symbol("Ingwaz",   RuneSystem.FUTHARK, FutharkAett.TYRS, 5, "\u16DC", "Ing/fertility"),
    Symbol("Dagaz",    RuneSystem.FUTHARK, FutharkAett.TYRS, 6, "\u16DE", "day/awakening"),
    Symbol("Othala",   RuneSystem.FUTHARK, FutharkAett.TYRS, 7, "\u16DF", "heritage/homeland"),
]

# ---------------------------------------------------------------------------
# Ogham (20 characters in 4 aicmi of 5)
# ---------------------------------------------------------------------------

OGHAM_CHARACTERS: List[Symbol] = [
    # Beithe aicme (group 0, positions 0-4)
    Symbol("Beith",    RuneSystem.OGHAM, OghamAicme.BEITHE, 0, "\u1681", "birch/beginnings"),
    Symbol("Luis",     RuneSystem.OGHAM, OghamAicme.BEITHE, 1, "\u1682", "rowan/protection"),
    Symbol("Fearn",    RuneSystem.OGHAM, OghamAicme.BEITHE, 2, "\u1683", "alder/foundation"),
    Symbol("Sail",     RuneSystem.OGHAM, OghamAicme.BEITHE, 3, "\u1684", "willow/intuition"),
    Symbol("Nion",     RuneSystem.OGHAM, OghamAicme.BEITHE, 4, "\u1685", "ash/connection"),

    # Huath aicme (group 1, positions 0-4)
    Symbol("Huath",    RuneSystem.OGHAM, OghamAicme.HUATH, 0, "\u1686", "hawthorn/cleansing"),
    Symbol("Duir",     RuneSystem.OGHAM, OghamAicme.HUATH, 1, "\u1687", "oak/strength"),
    Symbol("Tinne",    RuneSystem.OGHAM, OghamAicme.HUATH, 2, "\u1688", "holly/challenge"),
    Symbol("Coll",     RuneSystem.OGHAM, OghamAicme.HUATH, 3, "\u1689", "hazel/wisdom"),
    Symbol("Quert",    RuneSystem.OGHAM, OghamAicme.HUATH, 4, "\u168A", "apple/beauty"),

    # Muin aicme (group 2, positions 0-4)
    Symbol("Muin",     RuneSystem.OGHAM, OghamAicme.MUIN, 0, "\u168B", "vine/harvest"),
    Symbol("Gort",     RuneSystem.OGHAM, OghamAicme.MUIN, 1, "\u168C", "ivy/tenacity"),
    Symbol("nGéadal",  RuneSystem.OGHAM, OghamAicme.MUIN, 2, "\u168D", "reed/direction"),
    Symbol("Straif",   RuneSystem.OGHAM, OghamAicme.MUIN, 3, "\u168E", "blackthorn/discipline"),
    Symbol("Ruis",     RuneSystem.OGHAM, OghamAicme.MUIN, 4, "\u168F", "elder/transition"),

    # Ailm aicme (group 3, positions 0-4)
    Symbol("Ailm",     RuneSystem.OGHAM, OghamAicme.AILM, 0, "\u1690", "fir/clarity"),
    Symbol("Onn",      RuneSystem.OGHAM, OghamAicme.AILM, 1, "\u1691", "gorse/gathering"),
    Symbol("Úr",       RuneSystem.OGHAM, OghamAicme.AILM, 2, "\u1692", "heather/passion"),
    Symbol("Eadhadh",  RuneSystem.OGHAM, OghamAicme.AILM, 3, "\u1693", "aspen/endurance"),
    Symbol("Iodhadh",  RuneSystem.OGHAM, OghamAicme.AILM, 4, "\u1694", "yew/rebirth"),
]

# ---------------------------------------------------------------------------
# Combined alphabet and lookup tables
# ---------------------------------------------------------------------------

ALPHABET: List[Symbol] = FUTHARK_RUNES + OGHAM_CHARACTERS
"""All 44 symbols in canonical order."""

# Code -> Symbol lookup (only valid codes)
_CODE_TO_SYMBOL: Dict[int, Symbol] = {s.code: s for s in ALPHABET}

# Name -> Symbol lookup (case-insensitive)
_NAME_TO_SYMBOL: Dict[str, Symbol] = {s.name.lower(): s for s in ALPHABET}

# Valid code set for fast membership testing
VALID_CODES = frozenset(_CODE_TO_SYMBOL.keys())

# Total vocabulary size (2^6 = 64 possible codewords)
VOCAB_SIZE = 64

# Number of actual symbols
NUM_SYMBOLS = len(ALPHABET)  # 44


class OghamFutharkAlphabet:
    """Interface for encoding/decoding the 44-symbol alphabet."""

    @staticmethod
    def encode(symbol: Symbol) -> int:
        """Encode a Symbol to its 6-bit code."""
        return symbol.code

    @staticmethod
    def decode(code: int) -> Symbol:
        """Decode a 6-bit code to its Symbol.

        Raises:
            KeyError: If the code does not map to a valid symbol.
        """
        if code not in _CODE_TO_SYMBOL:
            raise KeyError(
                f"Code {code} (0b{code:06b}) is not a valid symbol. "
                f"Valid codes: {sorted(VALID_CODES)}"
            )
        return _CODE_TO_SYMBOL[code]

    @staticmethod
    def lookup_by_name(name: str) -> Symbol:
        """Find a symbol by its traditional name (case-insensitive).

        Raises:
            KeyError: If no symbol has that name.
        """
        key = name.lower()
        if key not in _NAME_TO_SYMBOL:
            raise KeyError(f"No symbol named '{name}'")
        return _NAME_TO_SYMBOL[key]

    @staticmethod
    def get_group_neighbors(symbol: Symbol) -> List[Symbol]:
        """Return all symbols in the same aett/aicme group."""
        return [
            s for s in ALPHABET
            if s.system == symbol.system and s.group == symbol.group
        ]

    @staticmethod
    def is_valid_code(code: int) -> bool:
        """Check whether a 6-bit code maps to a valid symbol."""
        return code in VALID_CODES

    @staticmethod
    def extract_system(code: int) -> int:
        """Extract the system bit (bit 5) from a 6-bit code."""
        return (code >> 5) & 0b1

    @staticmethod
    def extract_group(code: int) -> int:
        """Extract the group bits (bits 3-4) from a 6-bit code."""
        return (code >> 3) & 0b11

    @staticmethod
    def extract_position(code: int) -> int:
        """Extract the position bits (bits 0-2) from a 6-bit code."""
        return code & 0b111

    @staticmethod
    def encode_text(text: str) -> List[int]:
        """Encode a string of symbol names (space-separated) to codes."""
        codes = []
        for name in text.strip().split():
            sym = OghamFutharkAlphabet.lookup_by_name(name)
            codes.append(sym.code)
        return codes

    @staticmethod
    def decode_sequence(codes: List[int]) -> List[Symbol]:
        """Decode a list of 6-bit codes to symbols."""
        return [OghamFutharkAlphabet.decode(c) for c in codes]

    @staticmethod
    def all_symbols() -> List[Symbol]:
        """Return all 44 symbols."""
        return list(ALPHABET)

    @staticmethod
    def futhark_symbols() -> List[Symbol]:
        """Return only the 24 Elder Futhark runes."""
        return list(FUTHARK_RUNES)

    @staticmethod
    def ogham_symbols() -> List[Symbol]:
        """Return only the 20 Ogham characters."""
        return list(OGHAM_CHARACTERS)
