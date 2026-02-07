"""Tests for kala.fractal.alphabet -- 44-symbol Ogham-Futhark encoding."""

import pytest

from kala.fractal.alphabet import (
    ALPHABET,
    FUTHARK_RUNES,
    OGHAM_CHARACTERS,
    NUM_SYMBOLS,
    VALID_CODES,
    VOCAB_SIZE,
    FutharkAett,
    OghamAicme,
    OghamFutharkAlphabet,
    RuneSystem,
    Symbol,
)


class TestAlphabetStructure:
    """Test the alphabet definition."""

    def test_total_symbols(self):
        assert NUM_SYMBOLS == 44
        assert len(ALPHABET) == 44

    def test_futhark_count(self):
        assert len(FUTHARK_RUNES) == 24

    def test_ogham_count(self):
        assert len(OGHAM_CHARACTERS) == 20

    def test_vocab_size(self):
        assert VOCAB_SIZE == 64  # 2^6

    def test_all_codes_unique(self):
        codes = [s.code for s in ALPHABET]
        assert len(codes) == len(set(codes)), "Duplicate codes found"

    def test_all_names_unique(self):
        names = [s.name for s in ALPHABET]
        assert len(names) == len(set(names)), "Duplicate names found"

    def test_codes_fit_in_6_bits(self):
        for s in ALPHABET:
            assert 0 <= s.code <= 63, f"{s.name} has code {s.code} outside 6-bit range"


class TestFutharkAettir:
    """Test Elder Futhark aett groupings."""

    def test_freyrs_aett(self):
        freyrs = [s for s in FUTHARK_RUNES if s.group == FutharkAett.FREYRS]
        assert len(freyrs) == 8
        assert freyrs[0].name == "Fehu"
        assert freyrs[7].name == "Wunjo"

    def test_heimdalls_aett(self):
        heimdalls = [s for s in FUTHARK_RUNES if s.group == FutharkAett.HEIMDALLS]
        assert len(heimdalls) == 8
        assert heimdalls[0].name == "Hagalaz"
        assert heimdalls[7].name == "Sowilo"

    def test_tyrs_aett(self):
        tyrs = [s for s in FUTHARK_RUNES if s.group == FutharkAett.TYRS]
        assert len(tyrs) == 8
        assert tyrs[0].name == "Tiwaz"
        assert tyrs[7].name == "Othala"


class TestOghamAicmi:
    """Test Ogham aicme groupings."""

    def test_beithe_aicme(self):
        beithe = [s for s in OGHAM_CHARACTERS if s.group == OghamAicme.BEITHE]
        assert len(beithe) == 5
        assert beithe[0].name == "Beith"

    def test_huath_aicme(self):
        huath = [s for s in OGHAM_CHARACTERS if s.group == OghamAicme.HUATH]
        assert len(huath) == 5

    def test_muin_aicme(self):
        muin = [s for s in OGHAM_CHARACTERS if s.group == OghamAicme.MUIN]
        assert len(muin) == 5

    def test_ailm_aicme(self):
        ailm = [s for s in OGHAM_CHARACTERS if s.group == OghamAicme.AILM]
        assert len(ailm) == 5


class TestBitEncoding:
    """Test the 6-bit encoding structure."""

    def test_system_bit_futhark(self):
        for s in FUTHARK_RUNES:
            assert s.system_bit == 0, f"{s.name} should have system_bit=0"

    def test_system_bit_ogham(self):
        for s in OGHAM_CHARACTERS:
            assert s.system_bit == 1, f"{s.name} should have system_bit=1"

    def test_group_bits_range(self):
        for s in ALPHABET:
            assert 0 <= s.group_bits <= 3

    def test_position_bits_range(self):
        for s in ALPHABET:
            assert 0 <= s.position_bits <= 7

    def test_code_composition(self):
        """Code = (system << 5) | (group << 3) | position."""
        for s in ALPHABET:
            expected = (s.system << 5) | (s.group << 3) | s.position
            assert s.code == expected, f"{s.name}: expected {expected}, got {s.code}"

    def test_fehu_encoding(self):
        """Fehu: system=0, group=0, position=0 -> code 0b000000 = 0."""
        fehu = OghamFutharkAlphabet.lookup_by_name("Fehu")
        assert fehu.code == 0

    def test_beith_encoding(self):
        """Beith: system=1, group=0, position=0 -> code 0b100000 = 32."""
        beith = OghamFutharkAlphabet.lookup_by_name("Beith")
        assert beith.code == 32


class TestEncodeDecodeRoundTrip:
    """Test encoding and decoding round-trips."""

    def test_all_symbols_round_trip(self):
        for s in ALPHABET:
            code = OghamFutharkAlphabet.encode(s)
            decoded = OghamFutharkAlphabet.decode(code)
            assert decoded.name == s.name
            assert decoded.code == s.code

    def test_invalid_code_raises(self):
        # Code 24 is system=0, group=3, position=0 -- Futhark has no group 3
        with pytest.raises(KeyError):
            OghamFutharkAlphabet.decode(24)

    def test_name_lookup_case_insensitive(self):
        assert OghamFutharkAlphabet.lookup_by_name("fehu").name == "Fehu"
        assert OghamFutharkAlphabet.lookup_by_name("FEHU").name == "Fehu"

    def test_name_lookup_invalid(self):
        with pytest.raises(KeyError):
            OghamFutharkAlphabet.lookup_by_name("InvalidRune")


class TestGroupNeighbors:
    """Test group neighbor queries."""

    def test_futhark_group_size(self):
        fehu = OghamFutharkAlphabet.lookup_by_name("Fehu")
        neighbors = OghamFutharkAlphabet.get_group_neighbors(fehu)
        assert len(neighbors) == 8  # Freyr's Aett has 8 runes

    def test_ogham_group_size(self):
        beith = OghamFutharkAlphabet.lookup_by_name("Beith")
        neighbors = OghamFutharkAlphabet.get_group_neighbors(beith)
        assert len(neighbors) == 5  # Beithe aicme has 5 characters

    def test_same_group_members(self):
        fehu = OghamFutharkAlphabet.lookup_by_name("Fehu")
        wunjo = OghamFutharkAlphabet.lookup_by_name("Wunjo")
        neighbors = OghamFutharkAlphabet.get_group_neighbors(fehu)
        assert wunjo in neighbors


class TestValidCodes:
    """Test code validity checking."""

    def test_valid_code_count(self):
        assert len(VALID_CODES) == 44

    def test_is_valid(self):
        assert OghamFutharkAlphabet.is_valid_code(0)  # Fehu
        assert OghamFutharkAlphabet.is_valid_code(32)  # Beith

    def test_is_invalid(self):
        assert not OghamFutharkAlphabet.is_valid_code(24)  # unused slot
        assert not OghamFutharkAlphabet.is_valid_code(63)  # unused slot

    def test_extract_bits(self):
        code = 0b101001  # system=1, group=01, position=001
        assert OghamFutharkAlphabet.extract_system(code) == 1
        assert OghamFutharkAlphabet.extract_group(code) == 1
        assert OghamFutharkAlphabet.extract_position(code) == 1
