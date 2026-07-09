"""
Tests for libssim.transport.geometry (Herrera 2008: chord geometry of
Eq. 5-48, p. 119; parabolic profiles Eqs. 5-36/5-37 and Saha closure
Eq. 5-38, p. 116).
"""

from __future__ import annotations

import numpy as np
import pytest

from libssim.transport.base import PathSegment
from libssim.transport.geometry import SphericalOnion


@pytest.fixture()
def two_zone(make_state):
    """Factory: hot-core / cool-shell onion with adjustable radii."""

    def _build(r_core: float = 1.0e-3, r_outer: float = 2.0e-3):
        core = make_state(12000.0, 1.0e22)
        shell = make_state(6000.0, 1.0e21)
        return SphericalOnion(
            zones=(core, shell), boundaries_m=(r_core, r_outer)
        )

    return _build


class TestPathSegments:
    def test_single_zone_central_chord(self, make_state):
        onion = SphericalOnion(
            zones=(make_state(1.0e4, 1.0e22),), boundaries_m=(2.0e-3,)
        )
        segments = onion.path_segments(0.0)
        assert segments == (PathSegment(0, 4.0e-3),)  # full diameter 2R

    def test_two_zone_central_chord_ordering(self, two_zone):
        onion = two_zone(r_core=1.0e-3, r_outer=2.0e-3)
        segments = onion.path_segments(0.0)
        # far shell -> central core (merged double-length) -> near shell
        assert [s.zone_index for s in segments] == [1, 0, 1]
        assert segments[0].length_m == pytest.approx(1.0e-3, rel=1e-12)
        assert segments[1].length_m == pytest.approx(2.0e-3, rel=1e-12)
        assert segments[2].length_m == pytest.approx(1.0e-3, rel=1e-12)

    def test_chord_missing_the_core(self, two_zone):
        # p between R_core and R: only the outer shell is traversed.
        onion = two_zone(r_core=1.0e-3, r_outer=2.0e-3)
        p = 1.5e-3
        segments = onion.path_segments(p)
        assert [s.zone_index for s in segments] == [1]
        assert segments[0].length_m == pytest.approx(
            2.0 * np.sqrt((2.0e-3) ** 2 - p**2), rel=1e-12
        )

    @pytest.mark.parametrize("p_frac", [0.0, 0.13, 0.499, 0.62, 0.87, 0.999])
    def test_lengths_sum_to_full_chord(self, make_state, p_frac):
        # Telescoping identity: sum of segment lengths = 2*sqrt(R^2-p^2).
        radii = (0.5e-3, 0.9e-3, 1.4e-3, 2.0e-3)
        zones = tuple(
            make_state(1.0e4 - 1.0e3 * i, 1.0e22) for i in range(len(radii))
        )
        onion = SphericalOnion(zones=zones, boundaries_m=radii)
        p = p_frac * onion.outer_radius_m
        total = sum(s.length_m for s in onion.path_segments(p))
        assert total == pytest.approx(
            2.0 * np.sqrt(onion.outer_radius_m**2 - p**2), rel=1e-12
        )

    def test_mirror_symmetry_of_zone_sequence(self, make_state):
        radii = (0.5e-3, 0.9e-3, 1.4e-3, 2.0e-3)
        zones = tuple(make_state(1.0e4, 1.0e22) for _ in radii)
        onion = SphericalOnion(zones=zones, boundaries_m=radii)
        indices = [s.zone_index for s in onion.path_segments(0.2e-3)]
        assert indices == indices[::-1]  # palindromic in/out sequence

    def test_grazing_boundary_emits_no_zero_length_segment(self, two_zone):
        # p exactly on the inner boundary: the core is tangent, not
        # crossed — every emitted segment must have positive length.
        onion = two_zone(r_core=1.0e-3, r_outer=2.0e-3)
        segments = onion.path_segments(1.0e-3)
        assert [s.zone_index for s in segments] == [1]
        assert all(s.length_m > 0.0 for s in segments)

    @pytest.mark.parametrize("p", [-1.0e-6, 2.0e-3, 5.0e-3, np.nan])
    def test_invalid_impact_parameter_rejected(self, two_zone, p):
        onion = two_zone(r_core=1.0e-3, r_outer=2.0e-3)
        with pytest.raises(ValueError):
            onion.path_segments(p)


class TestConstruction:
    def test_mismatched_lengths_rejected(self, make_state):
        with pytest.raises(ValueError, match="same length"):
            SphericalOnion(
                zones=(make_state(1e4, 1e22),), boundaries_m=(1e-3, 2e-3)
            )

    def test_non_increasing_boundaries_rejected(self, make_state):
        zones = (make_state(1e4, 1e22), make_state(9e3, 1e22))
        with pytest.raises(ValueError, match="strictly increasing"):
            SphericalOnion(zones=zones, boundaries_m=(2e-3, 1e-3))

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="at least one zone"):
            SphericalOnion(zones=(), boundaries_m=())

    def test_zone_radius_field_is_not_consulted(self, make_state):
        # The documented contract: geometry boundaries are authoritative,
        # PlasmaState.radius_m is ignored by transport.
        zone = make_state(1e4, 1e22)  # radius_m = 1e-3 inside the state
        onion = SphericalOnion(zones=(zone,), boundaries_m=(5.0e-3,))
        assert onion.outer_radius_m == 5.0e-3
        assert onion.path_segments(0.0)[0].length_m == pytest.approx(1.0e-2)

    def test_path_segment_validation(self):
        with pytest.raises(ValueError):
            PathSegment(-1, 1.0)
        with pytest.raises(ValueError):
            PathSegment(0, 0.0)


class TestParabolicFactory:
    def test_profiles_and_zone_count(self, saha_solver):
        R = 2.0e-3
        onion = SphericalOnion.from_parabolic_profiles(
            center_temperature_K=12000.0,
            temperature_gradient_k1=0.5 / R**2,
            center_densities_m3={"Fe": 7.0e22, "Al": 3.0e22},
            density_gradient_k2=0.8 / R**2,
            outer_radius_m=R,
            n_zones=5,
            saha_solver=saha_solver,
        )
        assert onion.n_zones == 5
        assert onion.outer_radius_m == pytest.approx(R)
        temperatures = [z.temperature_K for z in onion.zones]
        assert np.all(np.diff(temperatures) < 0)  # Eq. 5-36: cooler outward
        # Eq. 5-36 evaluated at the first shell midpoint (r = R/10).
        r_mid = 0.1 * R
        assert temperatures[0] == pytest.approx(
            12000.0 * (1 - 0.5 / R**2 * r_mid**2), rel=1e-12
        )

    def test_zone_electron_density_satisfies_charge_equilibrium(
        self, saha_solver
    ):
        # Eq. 5-38 closure per zone: n_e equals the summed ion density
        # of a Saha balance evaluated at the zone's own conditions.
        R = 2.0e-3
        onion = SphericalOnion.from_parabolic_profiles(
            center_temperature_K=11000.0,
            temperature_gradient_k1=0.3 / R**2,
            center_densities_m3={"Fe": 5.0e22},
            density_gradient_k2=0.4 / R**2,
            outer_radius_m=R,
            n_zones=3,
            saha_solver=saha_solver,
        )
        for zone in onion.zones:
            heavies = {
                el: frac * (zone.total_density_m3 - zone.electron_density_m3)
                for el, frac in zone.composition.items()
            }
            balance = saha_solver.balance(
                zone.temperature_K, zone.electron_density_m3, heavies
            )
            assert balance.total_ion_density_m3 == pytest.approx(
                zone.electron_density_m3, rel=1e-10
            )

    def test_profile_positivity_enforced(self, saha_solver):
        R = 2.0e-3
        with pytest.raises(ValueError, match="stay positive"):
            SphericalOnion.from_parabolic_profiles(
                center_temperature_K=1.0e4,
                temperature_gradient_k1=1.0 / R**2,  # k1*R^2 = 1: T(R) = 0
                center_densities_m3={"Fe": 1.0e22},
                density_gradient_k2=0.0,
                outer_radius_m=R,
                n_zones=3,
                saha_solver=saha_solver,
            )
