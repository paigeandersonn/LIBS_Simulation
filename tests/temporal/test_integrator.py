"""
Tests for libssim.temporal.integrator (Herrera 2008: t_delay/t_gate,
pp. 46-47).

Covers the Phase 4 acceptance criterion (implementation_plan.md):
- Integrated intensity scales correctly with gate width.
"""

from __future__ import annotations

import numpy as np
import pytest

from libssim.temporal.decay_models import (
    Constant,
    ExponentialDecay,
    UniformPlasmaEvolution,
)
from libssim.temporal.integrator import GateIntegrator
from libssim.transport.emissivity import LTESpectralModel

TAU = 8.0e-7  # heavy-density e-folding time (s)


@pytest.fixture(scope="module")
def line_grid(resonance_wavelength_m):
    fwhm = 4.0e-12
    return np.linspace(
        resonance_wavelength_m - 8 * fwhm,
        resonance_wavelength_m + 8 * fwhm,
        201,
    )


@pytest.fixture(scope="module")
def model(saha_solver, resonance_transition, line_grid, atomic_masses_kg):
    return LTESpectralModel(
        saha_solver=saha_solver,
        wavelength_m=line_grid,
        transitions=(resonance_transition,),
        atomic_masses_kg=atomic_masses_kg,
        include_continuum=False,
    )


def make_integrator(model, heavy_profile) -> GateIntegrator:
    """Optically thin uniform plasma with fixed T and n_e: the radiance
    is exactly proportional to the heavy density, giving analytic time
    integrals to test against."""
    evolution = UniformPlasmaEvolution(
        temperature_K=Constant(1.0e4),
        heavy_density_m3=heavy_profile,
        composition={"Fe": 1.0},
        radius_m=1.0e-3,
        electron_density_m3=Constant(1.0e20),
    )
    return GateIntegrator(
        spectral_model=model, evolution=evolution, impact_parameter_m=0.0
    )


class TestGateIntegration:
    def test_constant_plasma_scales_exactly_with_gate_width(self, model):
        # Acceptance criterion: for a constant plasma the quadrature of
        # a constant is exact, so E is proportional to t_gate identically.
        integrator = make_integrator(model, Constant(1.0e16))
        one = integrator.gate_integrated(1.0e-6, 1.0e-6)
        two = integrator.gate_integrated(1.0e-6, 2.0e-6)
        assert np.allclose(two.intensity, 2.0 * one.intensity, rtol=1e-12)
        # ... and equals width x instantaneous radiance.
        snap = integrator.snapshot(1.0e-6)
        assert np.allclose(
            one.intensity, 1.0e-6 * snap.intensity, rtol=1e-12
        )

    def test_exponential_decay_matches_analytic_integral(self, model):
        # Optically thin + fixed (T, n_e): I(t) = I(0) e^(-t/tau), so
        # E = I(0) * tau * (e^(-td/tau) - e^(-(td+tw)/tau)) analytically.
        integrator = make_integrator(
            model, ExponentialDecay(1.0e16, decay_time_s=TAU)
        )
        delay, width = 2.0e-7, 1.0e-6
        gate = integrator.gate_integrated(delay, width, n_time_nodes=16)
        reference = integrator.snapshot(0.0).intensity
        factor = TAU * (
            np.exp(-delay / TAU) - np.exp(-(delay + width) / TAU)
        )
        expected = reference * factor
        scale = expected.max()
        assert np.allclose(
            gate.intensity, expected, rtol=1e-6, atol=1e-9 * scale
        )

    def test_trapezoid_converges_to_gauss(self, model):
        integrator = make_integrator(
            model, ExponentialDecay(1.0e16, decay_time_s=TAU)
        )
        gauss = integrator.gate_integrated(2.0e-7, 1.0e-6, n_time_nodes=16)
        trapezoid = integrator.gate_integrated(
            2.0e-7, 1.0e-6, n_time_nodes=129, quadrature="trapezoid"
        )
        assert np.allclose(
            trapezoid.intensity, gauss.intensity, rtol=1e-4,
            atol=1e-8 * gauss.intensity.max(),
        )

    def test_metadata_records_gate_and_history(self, model):
        integrator = make_integrator(
            model, ExponentialDecay(1.0e16, decay_time_s=TAU)
        )
        gate = integrator.gate_integrated(5.0e-7, 2.0e-6, n_time_nodes=8)
        md = gate.metadata
        assert md["gate_delay_s"] == 5.0e-7
        assert md["gate_width_s"] == 2.0e-6
        assert md["n_time_nodes"] == 8
        assert len(md["time_nodes_s"]) == 8
        assert len(md["zone_temperatures_K_per_node"]) == 8
        assert "radiant exposure" in md["intensity_units"]
        assert md["observation_mode"].startswith("impact_parameter")

    @pytest.mark.parametrize(
        "delay, width, nodes, quadrature",
        [
            (-1.0e-6, 1.0e-6, 8, "gauss"),   # negative delay
            (0.0, 0.0, 8, "gauss"),           # zero width
            (0.0, 1.0e-6, 1, "gauss"),        # too few nodes
            (0.0, 1.0e-6, 8, "simpson"),      # unknown quadrature
        ],
    )
    def test_invalid_gate_parameters_rejected(
        self, model, delay, width, nodes, quadrature
    ):
        integrator = make_integrator(model, Constant(1.0e16))
        with pytest.raises(ValueError):
            integrator.gate_integrated(
                delay, width, n_time_nodes=nodes, quadrature=quadrature
            )


class TestTimeResolved:
    def test_snapshots_decay_in_time(self, model):
        integrator = make_integrator(
            model, ExponentialDecay(1.0e16, decay_time_s=TAU)
        )
        spectra = integrator.time_resolved([0.0, 5.0e-7, 2.0e-6])
        peaks = [s.intensity.max() for s in spectra]
        assert peaks[0] > peaks[1] > peaks[2]
        assert [s.metadata["time_s"] for s in spectra] == [0.0, 5.0e-7, 2.0e-6]

    def test_snapshot_units_and_zone_summary(self, model):
        integrator = make_integrator(model, Constant(1.0e16))
        snap = integrator.snapshot(1.0e-6)
        assert "spectral radiance" in snap.metadata["intensity_units"]
        assert snap.metadata["zone_temperatures_K"] == (1.0e4,)
        assert snap.metadata["n_zones"] == 1
