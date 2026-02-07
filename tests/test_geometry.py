"""Tests for kala.fractal.geometry -- Golden Ratio geometric primitives."""

import math

import pytest

from kala.fractal.geometry import (
    GOLDEN_ANGLE_DEG,
    GOLDEN_ANGLE_RAD,
    PHI,
    GoldenRatioGeometry,
)


class TestConstants:
    """Test fundamental constants."""

    def test_phi_value(self):
        assert abs(PHI - 1.618033988749895) < 1e-12

    def test_phi_property(self):
        """phi^2 = phi + 1."""
        assert abs(PHI * PHI - PHI - 1) < 1e-12

    def test_golden_angle_degrees(self):
        expected = 360.0 / (PHI * PHI)
        assert abs(GOLDEN_ANGLE_DEG - expected) < 1e-6
        assert abs(GOLDEN_ANGLE_DEG - 137.508) < 0.001

    def test_golden_angle_radians(self):
        assert abs(GOLDEN_ANGLE_RAD - math.radians(GOLDEN_ANGLE_DEG)) < 1e-12


class TestBranchAngle:
    """Test branch angle calculations."""

    def test_first_child_angle(self):
        angle = GoldenRatioGeometry.branch_angle(0)
        assert angle == 0.0

    def test_second_child_angle(self):
        angle = GoldenRatioGeometry.branch_angle(1)
        assert abs(angle - GOLDEN_ANGLE_DEG) < 1e-6

    def test_angles_wrap_around(self):
        angle = GoldenRatioGeometry.branch_angle(100)
        assert 0 <= angle < 360.0


class TestScaleFactor:
    """Test depth-based scaling."""

    def test_root_scale(self):
        assert GoldenRatioGeometry.scale_factor(0) == 1.0

    def test_depth_1_scale(self):
        expected = 1.0 / PHI  # ~0.618
        assert abs(GoldenRatioGeometry.scale_factor(1) - expected) < 1e-12

    def test_monotonically_decreasing(self):
        prev = GoldenRatioGeometry.scale_factor(0)
        for d in range(1, 10):
            curr = GoldenRatioGeometry.scale_factor(d)
            assert curr < prev
            prev = curr


class TestStemlinePosition:
    """Test stemline position calculations."""

    def test_root_position(self):
        x, y = GoldenRatioGeometry.stemline_position(0, 0.0)
        # At depth 0, scale=1.0, angle=0 -> (1.0, 0.0)
        assert abs(x - 1.0) < 1e-12
        assert abs(y - 0.0) < 1e-12

    def test_position_with_parent(self):
        x, y = GoldenRatioGeometry.stemline_position(0, 0.0, (5.0, 5.0))
        assert abs(x - 6.0) < 1e-12
        assert abs(y - 5.0) < 1e-12

    def test_90_degree_angle(self):
        x, y = GoldenRatioGeometry.stemline_position(0, 90.0)
        assert abs(x - 0.0) < 1e-10
        assert abs(y - 1.0) < 1e-10


class TestBranchPositions:
    """Test rune/ogham branch position layouts."""

    def test_rune_branches(self):
        branches = GoldenRatioGeometry.rune_branch_positions()
        assert branches["above"] < 0
        assert branches["center"] == 0
        assert branches["below"] > 0

    def test_ogham_branches(self):
        branches = GoldenRatioGeometry.ogham_branch_positions()
        assert branches["left"] < 0
        assert branches["stemline"] == 0
        assert branches["right"] > 0


class TestDistanceFunctions:
    """Test distance and similarity metrics."""

    def test_distance_zero(self):
        assert GoldenRatioGeometry.distance((0, 0), (0, 0)) == 0.0

    def test_distance_unit(self):
        assert abs(GoldenRatioGeometry.distance((0, 0), (1, 0)) - 1.0) < 1e-12

    def test_angular_distance_same(self):
        assert GoldenRatioGeometry.angular_distance(45.0, 45.0) == 0.0

    def test_angular_distance_opposite(self):
        assert abs(GoldenRatioGeometry.angular_distance(0.0, 180.0) - 180.0) < 1e-10

    def test_angular_distance_wraparound(self):
        assert abs(GoldenRatioGeometry.angular_distance(10.0, 350.0) - 20.0) < 1e-10

    def test_angular_similarity_same(self):
        sim = GoldenRatioGeometry.angular_similarity(45.0, 45.0)
        assert abs(sim - 1.0) < 1e-12

    def test_angular_similarity_opposite(self):
        sim = GoldenRatioGeometry.angular_similarity(0.0, 180.0)
        assert abs(sim - (-1.0)) < 1e-10
