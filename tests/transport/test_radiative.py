"""
Tests for libssim.transport.radiative (Herrera 2008: Eq. 5-48 p. 119;
Eq. 3-10 p. 55; self-reversal pp. 53-54).

Covers the Phase 3 acceptance criteria (implementation_plan.md):
- Optically thick resonance lines produce visible self-reversal
- Increasing optical depth decreases transmitted intensity in the core
- Line-of-sight integration is numerically stable
"""

from __future__ import annotations

import numpy as np
import pytest

from libssim.core.spectrum import Spectrum
from libssim.physics.emission import blackbody_spectral_radiance_hz
from libssim.core.constants import C
from libssim.transport.emissivity import LTESpectralModel
from libssim.transport.geometry import SphericalOnion
from libssim.transport.radiative import (
    disk_integrated_radiance,
    emergent_radiance,
    emergent_spectrum,
    optical_depth,
)

from conftest import ATOMIC_MASSES_KG, RESONANCE_WAVELENGTH_M

N_LAMBDA = 5  # synthetic tests use small flat spectral arrays


def uniform_sphere(make_state, radius=2.0e-3):
    return SphericalOnion(
        zones=(make_state(1.0e4, 1.0e22),), boundaries_m=(radius,)
    )


def flat(value: float) -> np.ndarray:
    return np.full((1, N_LAMBDA), value)


class TestAnalyticLimits:
    """Synthetic epsilon/kappa arrays: exact solutions, no physics."""

    @pytest.mark.parametrize("p_frac", [0.0, 0.3, 0.75, 0.99])
    def test_uniform_sphere_matches_eq_3_10(self, make_state, p_frac):
        # I = S * (1 - exp(-kappa * L)) with L the chord length —
        # Eq. 3-10, p. 55 with the source as the blackbody radiance.
        onion = uniform_sphere(make_state)
        p = p_frac * onion.outer_radius_m
        eps0, kap0 = 7.5e3, 2.0e2
        chord = 2.0 * np.sqrt(onion.outer_radius_m**2 - p**2)
        radiance = emergent_radiance(
            onion.path_segments(p), flat(eps0), flat(kap0)
        )
        expected = (eps0 / kap0) * (1.0 - np.exp(-kap0 * chord))
        assert np.allclose(radiance, expected, rtol=1e-14)

    def test_optically_thin_limit(self, make_state):
        # tau ~ 1e-9: I -> epsilon * L (relative deviation ~ tau/2).
        onion = uniform_sphere(make_state)
        length = 2.0 * onion.outer_radius_m
        kappa0 = 1.0e-9 / length
        radiance = emergent_radiance(
            onion.path_segments(0.0), flat(3.0e4), flat(kappa0)
        )
        assert np.allclose(radiance, 3.0e4 * length, rtol=1e-8)

    def test_transparent_zones_accumulate_pure_emission(self, make_state):
        # kappa identically 0 must be exact, not a divide-by-zero hole.
        onion = uniform_sphere(make_state)
        radiance = emergent_radiance(
            onion.path_segments(0.0), flat(3.0e4), flat(0.0)
        )
        assert np.allclose(
            radiance, 3.0e4 * 2.0 * onion.outer_radius_m, rtol=1e-15
        )

    def test_optically_thick_saturates_at_source(self, make_state):
        onion = uniform_sphere(make_state)
        eps0, kap0 = 7.5e3, 1.0e7  # tau ~ 4e4
        radiance = emergent_radiance(
            onion.path_segments(0.0), flat(eps0), flat(kap0)
        )
        assert np.allclose(radiance, eps0 / kap0, rtol=1e-12)

    @pytest.mark.parametrize("kappa0", [0.0, 1e-12, 1.0, 1e6, 1e12])
    def test_stable_over_extreme_optical_depths(self, make_state, kappa0):
        # Acceptance criterion: numerically stable — never NaN/Inf, and
        # bounded by the source function for any kappa.
        onion = uniform_sphere(make_state)
        radiance = emergent_radiance(
            onion.path_segments(0.0), flat(7.5e3), flat(kappa0)
        )
        assert np.all(np.isfinite(radiance))
        if kappa0 > 0.0:
            assert np.all(radiance <= 7.5e3 / kappa0 + 1e-30)

    def test_two_zone_cold_absorber_attenuation(self, make_state):
        # Non-emitting outer shell: the core radiance is attenuated by
        # exactly exp(-kappa_shell * one-way shell path) — increasing
        # optical depth monotonically dims the transmitted core.
        core = make_state(1.2e4, 1.0e22)
        shell = make_state(6.0e3, 1.0e21)
        onion = SphericalOnion(
            zones=(core, shell), boundaries_m=(1.0e-3, 1.6e-3)
        )
        eps = np.vstack([np.full(N_LAMBDA, 5.0e3), np.zeros(N_LAMBDA)])
        core_only = emergent_radiance(
            onion.path_segments(0.0), eps, np.zeros_like(eps)
        )
        previous = np.inf
        for kappa_shell in [1.0e2, 1.0e3, 4.0e3]:
            kap = np.vstack(
                [np.zeros(N_LAMBDA), np.full(N_LAMBDA, kappa_shell)]
            )
            radiance = emergent_radiance(onion.path_segments(0.0), eps, kap)
            expected = core_only * np.exp(-kappa_shell * 0.6e-3)
            assert np.allclose(radiance, expected, rtol=1e-12)
            assert radiance[0] < previous
            previous = radiance[0]

    def test_optical_depth_sums_kappa_length(self, make_state):
        core = make_state(1.2e4, 1.0e22)
        shell = make_state(6.0e3, 1.0e21)
        onion = SphericalOnion(
            zones=(core, shell), boundaries_m=(1.0e-3, 2.0e-3)
        )
        kap = np.vstack([np.full(N_LAMBDA, 10.0), np.full(N_LAMBDA, 2.0)])
        tau = optical_depth(onion.path_segments(0.0), kap)
        # central chord: 2 mm through the core, 2 x 1 mm through the shell
        assert np.allclose(tau, 10.0 * 2.0e-3 + 2.0 * 2.0e-3, rtol=1e-12)

    def test_disk_average_of_thin_uniform_sphere(self, make_state):
        # Analytic: <I> = (2/R^2) int eps*2*sqrt(R^2-p^2) p dp = eps*4R/3.
        onion = uniform_sphere(make_state)
        eps0 = 3.0e4
        average = disk_integrated_radiance(
            onion, flat(eps0), flat(0.0), n_impact=96
        )
        expected = eps0 * 4.0 * onion.outer_radius_m / 3.0
        assert np.allclose(average, expected, rtol=1e-3)

    def test_zone_index_out_of_range_rejected(self, make_state):
        onion = uniform_sphere(make_state)
        segments = onion.path_segments(0.0)
        with pytest.raises(ValueError, match="zone rows"):
            emergent_radiance(
                segments, np.zeros((0, N_LAMBDA)), np.zeros((0, N_LAMBDA))
            )


@pytest.fixture(scope="module")
def line_grid():
    fwhm = 3.5e-12
    return np.linspace(
        RESONANCE_WAVELENGTH_M - 10 * fwhm,
        RESONANCE_WAVELENGTH_M + 10 * fwhm,
        3001,
    )


@pytest.fixture(scope="module")
def model(saha_solver, resonance_transition, line_grid):
    return LTESpectralModel(
        saha_solver=saha_solver,
        wavelength_m=line_grid,
        transitions=(resonance_transition,),
        atomic_masses_kg=ATOMIC_MASSES_KG,
        include_continuum=False,
    )


class TestSelfAbsorptionPhysics:
    """Full-physics acceptance tests: hot core + cool shell."""

    def build_onion(self, make_state, shell_heavy_density: float):
        # Hot, partially ionized core; cool dense shell of the same
        # element -> optically thick resonance-line envelope.
        core = make_state(8000.0, 1.0e22, electron_density_m3=1.0e22)
        shell = make_state(
            5000.0, shell_heavy_density, electron_density_m3=1.0e19
        )
        return SphericalOnion(
            zones=(core, shell), boundaries_m=(1.0e-3, 1.6e-3)
        )

    def test_optically_thick_line_self_reverses(
        self, make_state, model, line_grid
    ):
        # Acceptance criterion: visible self-reversal (Fig. 3-3 dip,
        # pp. 53-54) — the saturated core approaches B_nu(T_shell),
        # while the wings escape from the hot core.
        onion = self.build_onion(make_state, shell_heavy_density=2.0e21)
        epsilon, kappa = model.geometry_properties(onion)
        radiance = emergent_radiance(
            onion.path_segments(0.0), epsilon, kappa
        )
        center = np.argmin(np.abs(line_grid - RESONANCE_WAVELENGTH_M))
        peak = np.argmax(radiance)
        assert peak != center                       # maxima off-center
        assert radiance[peak] > 1.5 * radiance[center]  # visible dip
        # The reversed core saturates toward (below) the shell blackbody.
        planck_shell = blackbody_spectral_radiance_hz(
            C / RESONANCE_WAVELENGTH_M, 5000.0
        )
        assert radiance[center] <= planck_shell * (1 + 1e-9)

    def test_core_intensity_decreases_with_optical_depth(
        self, make_state, model, line_grid
    ):
        # Acceptance criterion: raising the absorbing column strictly
        # dims the transmitted line core.
        center = np.argmin(np.abs(line_grid - RESONANCE_WAVELENGTH_M))
        core_values = []
        taus = []
        for scale in [0.02, 0.1, 0.3, 1.0]:
            onion = self.build_onion(
                make_state, shell_heavy_density=scale * 2.0e21
            )
            epsilon, kappa = model.geometry_properties(onion)
            segments = onion.path_segments(0.0)
            core_values.append(emergent_radiance(segments, epsilon, kappa)[center])
            taus.append(optical_depth(segments, kappa)[center])
        assert np.all(np.diff(taus) > 0)          # tau grows with density
        assert np.all(np.diff(core_values) < 0)   # core intensity falls

    def test_emergent_spectrum_container(self, make_state, model, line_grid):
        onion = self.build_onion(make_state, shell_heavy_density=2.0e21)
        spectrum = emergent_spectrum(onion, model, impact_parameter_m=0.0)
        assert isinstance(spectrum, Spectrum)
        assert spectrum.wavelength_m.shape == line_grid.shape
        assert spectrum.metadata["observation_mode"].startswith(
            "impact_parameter"
        )
        # Per-wavelength conversion: I_lambda = I_nu * c / lambda^2.
        epsilon, kappa = model.geometry_properties(onion)
        radiance_nu = emergent_radiance(
            onion.path_segments(0.0), epsilon, kappa
        )
        assert np.allclose(
            spectrum.intensity, radiance_nu * C / line_grid**2, rtol=1e-13
        )

    def test_disk_integrated_spectrum_smaller_than_central_ray(
        self, make_state, model, line_grid
    ):
        # Limb chords are shorter/dimmer, so the disk average of the
        # line peak lies below the central-ray value.
        onion = self.build_onion(make_state, shell_heavy_density=2.0e21)
        disk = emergent_spectrum(onion, model, n_impact=32)
        ray = emergent_spectrum(onion, model, impact_parameter_m=0.0)
        assert disk.metadata["observation_mode"] == "disk-integrated"
        assert disk.intensity.max() < ray.intensity.max()
