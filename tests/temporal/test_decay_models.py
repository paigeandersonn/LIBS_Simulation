"""
Tests for libssim.temporal.decay_models (profiles + evolutions;
Herrera 2008: Eqs. 5-36/5-37/5-38 p. 116; App. B Eq. B-9 p. 269).
"""

from __future__ import annotations

import numpy as np
import pytest

from libssim.temporal.decay_models import (
    Constant,
    CustomEvolution,
    ExpandingOnionEvolution,
    ExponentialDecay,
    PowerLawDecay,
    UniformPlasmaEvolution,
)
from libssim.transport.geometry import SphericalOnion


class TestProfiles:
    def test_constant(self):
        profile = Constant(5000.0)
        assert profile(0.0) == 5000.0
        assert profile(1.0e-5) == 5000.0

    def test_exponential_decay_values(self):
        profile = ExponentialDecay(1.0e4, decay_time_s=1.0e-6)
        assert profile(0.0) == pytest.approx(1.0e4)
        assert profile(1.0e-6) == pytest.approx(1.0e4 / np.e, rel=1e-12)

    def test_exponential_clamped_before_start(self):
        profile = ExponentialDecay(1.0e4, 1.0e-6, start_time_s=5.0e-7)
        assert profile(0.0) == 1.0e4
        assert profile(5.0e-7) == 1.0e4
        assert profile(1.5e-6) == pytest.approx(1.0e4 / np.e, rel=1e-12)

    def test_power_law_values(self):
        # v(t_ref) = v0; v(2 t_ref) = v0 * 2^-b (Aguilera & Aragon form).
        profile = PowerLawDecay(1.0e4, reference_time_s=1.0e-6, exponent=0.7)
        assert profile(1.0e-6) == pytest.approx(1.0e4)
        assert profile(2.0e-6) == pytest.approx(1.0e4 * 2.0**-0.7, rel=1e-12)

    def test_power_law_undefined_at_zero(self):
        profile = PowerLawDecay(1.0e4, 1.0e-6, 0.7)
        with pytest.raises(ValueError, match="t > 0"):
            profile(0.0)

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: Constant(np.inf),
            lambda: ExponentialDecay(1.0, decay_time_s=0.0),
            lambda: ExponentialDecay(1.0, 1.0, start_time_s=-1.0),
            lambda: PowerLawDecay(1.0, reference_time_s=0.0, exponent=1.0),
            lambda: PowerLawDecay(1.0, 1.0, exponent=-0.5),
        ],
    )
    def test_invalid_construction_rejected(self, factory):
        with pytest.raises(ValueError):
            factory()

    def test_negative_time_rejected(self):
        with pytest.raises(ValueError):
            Constant(1.0)(-1.0e-9)


class TestUniformPlasmaEvolution:
    def test_prescribed_electron_density(self):
        evolution = UniformPlasmaEvolution(
            temperature_K=ExponentialDecay(1.2e4, 2.0e-6),
            heavy_density_m3=Constant(1.0e22),
            composition={"Fe": 1.0},
            radius_m=1.5e-3,
            electron_density_m3=Constant(1.0e21),
        )
        geometry = evolution.geometry_at(2.0e-6)
        assert isinstance(geometry, SphericalOnion)
        assert geometry.n_zones == 1
        zone = geometry.zones[0]
        assert zone.temperature_K == pytest.approx(1.2e4 / np.e, rel=1e-12)
        assert zone.electron_density_m3 == 1.0e21
        assert zone.total_density_m3 == pytest.approx(1.0e22 + 1.0e21)
        assert zone.time_s == 2.0e-6
        assert geometry.outer_radius_m == 1.5e-3

    def test_saha_closure(self, saha_solver):
        evolution = UniformPlasmaEvolution(
            temperature_K=Constant(1.0e4),
            heavy_density_m3=Constant(5.0e22),
            composition={"Fe": 1.0},
            saha_solver=saha_solver,
        )
        zone = evolution.geometry_at(1.0e-6).zones[0]
        expected = saha_solver.solve_electron_density(1.0e4, {"Fe": 5.0e22})
        assert zone.electron_density_m3 == pytest.approx(expected, rel=1e-12)

    def test_exactly_one_electron_source_required(self, saha_solver):
        kwargs = dict(
            temperature_K=Constant(1.0e4),
            heavy_density_m3=Constant(1.0e22),
            composition={"Fe": 1.0},
        )
        with pytest.raises(ValueError, match="exactly one"):
            UniformPlasmaEvolution(**kwargs)  # neither
        with pytest.raises(ValueError, match="exactly one"):
            UniformPlasmaEvolution(
                **kwargs,
                electron_density_m3=Constant(1e20),
                saha_solver=saha_solver,
            )  # both

    def test_unphysical_profile_value_reported_with_time(self):
        evolution = UniformPlasmaEvolution(
            temperature_K=lambda t: 1.0e4 - 2.0e10 * t,  # goes negative
            heavy_density_m3=Constant(1.0e22),
            composition={"Fe": 1.0},
            electron_density_m3=Constant(1e20),
        )
        with pytest.raises(ValueError, match="temperature"):
            evolution.geometry_at(1.0e-6)


class TestCustomEvolution:
    def test_passthrough(self, make_state):
        onion = SphericalOnion(
            zones=(make_state(1e4, 1e22),), boundaries_m=(1e-3,)
        )
        evolution = CustomEvolution(geometry_factory=lambda t: onion)
        assert evolution.geometry_at(1.0e-6) is onion

    def test_wrong_return_type_rejected(self):
        evolution = CustomEvolution(geometry_factory=lambda t: "not a geometry")
        with pytest.raises(TypeError, match="PlasmaGeometry"):
            evolution.geometry_at(0.0)


class TestExpandingOnionEvolution:
    R0 = 1.0e-3

    def build(self, saha_solver, radius_profile):
        return ExpandingOnionEvolution(
            center_temperature_K=Constant(1.1e4),
            radius_m=radius_profile,
            initial_radius_m=self.R0,
            temperature_gradient_k1=0.3 / self.R0**2,
            density_gradient_k2=0.5 / self.R0**2,
            center_densities_m3={"Fe": 5.0e22},
            n_zones=4,
            saha_solver=saha_solver,
        )

    def test_matches_static_factory_at_initial_radius(self, saha_solver):
        # R(t) = R0: Eq. B-9 reduces to the initial profile, so the
        # snapshot must equal the Phase 3 factory output exactly.
        evolution = self.build(saha_solver, Constant(self.R0))
        snapshot = evolution.geometry_at(0.0)
        reference = SphericalOnion.from_parabolic_profiles(
            center_temperature_K=1.1e4,
            temperature_gradient_k1=0.3 / self.R0**2,
            center_densities_m3={"Fe": 5.0e22},
            density_gradient_k2=0.5 / self.R0**2,
            outer_radius_m=self.R0,
            n_zones=4,
            saha_solver=saha_solver,
        )
        for got, want in zip(snapshot.zones, reference.zones):
            assert got.temperature_K == pytest.approx(
                want.temperature_K, rel=1e-12
            )
            assert got.total_density_m3 == pytest.approx(
                want.total_density_m3, rel=1e-12
            )
        assert snapshot.boundaries_m == reference.boundaries_m

    def test_expansion_dilutes_and_cools_geometry(self, saha_solver):
        # Doubling R: center densities scale by (R0/R)^3 = 1/8
        # (Eq. B-9) and boundaries scale with R.
        evolution = self.build(
            saha_solver, lambda t: self.R0 * (1.0 + 1.0e6 * t)
        )
        early = evolution.geometry_at(0.0)
        late = evolution.geometry_at(1.0e-6)  # R = 2 R0
        assert late.outer_radius_m == pytest.approx(2 * self.R0, rel=1e-12)
        heavy_early = early.zones[0].total_density_m3 - early.zones[
            0
        ].electron_density_m3
        heavy_late = late.zones[0].total_density_m3 - late.zones[
            0
        ].electron_density_m3
        # Innermost-zone profile factor (1 - k2' r_mid^2) is invariant
        # in the self-similar coordinate, so the heavy density scales
        # exactly by 1/8.
        assert heavy_late == pytest.approx(heavy_early / 8.0, rel=1e-12)

    def test_positivity_invariant_under_large_expansion(self, saha_solver):
        # k2 * R^2 is invariant (module docs): a 50x expansion must not
        # trip the positivity guard of the Phase 3 factory.
        evolution = self.build(saha_solver, Constant(50 * self.R0))
        geometry = evolution.geometry_at(0.0)
        assert all(z.temperature_K > 0 for z in geometry.zones)

    def test_initial_positivity_enforced(self, saha_solver):
        with pytest.raises(ValueError, match="stay positive"):
            ExpandingOnionEvolution(
                center_temperature_K=Constant(1.0e4),
                radius_m=Constant(self.R0),
                initial_radius_m=self.R0,
                temperature_gradient_k1=1.0 / self.R0**2,
                density_gradient_k2=0.0,
                center_densities_m3={"Fe": 1.0e22},
                n_zones=3,
                saha_solver=saha_solver,
            )
