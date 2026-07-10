"""
Tests for libssim.analysis (Herrera 2008: Eq. 5-56, p. 122) — CSV
import, resampling, correlation, sweeps and plotting.
"""

from __future__ import annotations

import numpy as np
import pytest

from libssim.analysis import (
    correlation_coefficient,
    load_spectrum_csv,
    plot_spectra,
    resample,
    sweep,
    sweep_gate_delay,
)
from libssim.core.spectrum import Spectrum


def make_spectrum(intensity, grid=None) -> Spectrum:
    intensity = np.asarray(intensity, dtype=float)
    if grid is None:
        grid = 400e-9 + np.arange(intensity.size) * 1e-12
    return Spectrum(wavelength_m=np.asarray(grid), intensity=intensity)


class TestCsvImport:
    def test_round_trip_nm(self, tmp_path):
        path = tmp_path / "spectrum.csv"
        path.write_text(
            "# wavelength_nm,intensity\n400.0,10.0\n400.5,20.0\n401.0,15.0\n"
        )
        spectrum = load_spectrum_csv(path)
        assert np.allclose(
            spectrum.wavelength_m, [400.0e-9, 400.5e-9, 401.0e-9]
        )
        assert np.allclose(spectrum.intensity, [10.0, 20.0, 15.0])
        assert spectrum.metadata["source"] == str(path)

    def test_sorts_and_converts_angstrom(self, tmp_path):
        path = tmp_path / "spectrum.csv"
        path.write_text("4010,1.0\n4000,2.0\n")
        spectrum = load_spectrum_csv(path, wavelength_unit="angstrom")
        assert np.allclose(spectrum.wavelength_m, [400.0e-9, 401.0e-9])
        assert np.allclose(spectrum.intensity, [2.0, 1.0])  # reordered

    def test_bad_unit_and_columns_rejected(self, tmp_path):
        path = tmp_path / "spectrum.csv"
        path.write_text("400,1\n401,2\n")
        with pytest.raises(ValueError, match="wavelength_unit"):
            load_spectrum_csv(path, wavelength_unit="furlong")
        with pytest.raises(ValueError, match="columns"):
            load_spectrum_csv(path, intensity_column=5)


class TestResampleAndCorrelation:
    def test_resample_interpolates(self):
        spectrum = make_spectrum([0.0, 1.0, 0.0])
        fine = resample(
            spectrum, np.linspace(spectrum.wavelength_m[0],
                                  spectrum.wavelength_m[-1], 5)
        )
        assert fine.intensity == pytest.approx([0.0, 0.5, 1.0, 0.5, 0.0])
        assert fine.metadata["resampled"] is True

    def test_correlation_of_identical_shapes_is_one(self):
        a = make_spectrum(np.random.default_rng(0).random(100))
        # Scale- and offset-invariance of Eq. 5-56.
        b = make_spectrum(3.0 * a.intensity + 42.0, grid=a.wavelength_m)
        assert correlation_coefficient(a, b) == pytest.approx(1.0, abs=1e-12)

    def test_correlation_matches_numpy(self):
        rng = np.random.default_rng(1)
        x, y = rng.random(200), rng.random(200)
        a, b = make_spectrum(x), make_spectrum(y)
        assert correlation_coefficient(a, b) == pytest.approx(
            np.corrcoef(x, y)[0, 1], rel=1e-12
        )

    def test_anticorrelation(self):
        a = make_spectrum([0.0, 1.0, 2.0, 3.0])
        b = make_spectrum([3.0, 2.0, 1.0, 0.0])
        assert correlation_coefficient(a, b) == pytest.approx(-1.0)

    def test_grid_mismatch_rejected(self):
        a = make_spectrum(np.ones(4) * [1, 2, 3, 4])
        b = make_spectrum([1, 2, 3, 4],
                          grid=500e-9 + np.arange(4) * 1e-12)
        with pytest.raises(ValueError, match="same wavelength grid"):
            correlation_coefficient(a, b)

    def test_constant_spectrum_rejected(self):
        a = make_spectrum([1.0, 2.0, 3.0])
        b = make_spectrum([5.0, 5.0, 5.0])
        with pytest.raises(ValueError, match="zero variance"):
            correlation_coefficient(a, b)


class TestSweeps:
    def test_generic_sweep_tags_metadata(self):
        spectra = sweep(
            [1.0, 2.0],
            lambda v: make_spectrum([v, 2 * v]),
            label="scale",
        )
        assert [s.metadata["scale"] for s in spectra] == [1.0, 2.0]
        assert np.allclose(spectra[1].intensity, [2.0, 4.0])

    def test_sweep_rejects_non_spectrum(self):
        with pytest.raises(TypeError):
            sweep([1], lambda v: v)

    def test_gate_delay_sweep_decays(
        self, saha_solver, resonance_transition, atomic_masses_kg,
        resonance_wavelength_m,
    ):
        # Full pipeline: later gates on a decaying plasma record less.
        from libssim.temporal.decay_models import (
            Constant,
            ExponentialDecay,
            UniformPlasmaEvolution,
        )
        from libssim.temporal.integrator import GateIntegrator
        from libssim.transport.emissivity import LTESpectralModel

        fwhm = 4.0e-12
        grid = np.linspace(
            resonance_wavelength_m - 8 * fwhm,
            resonance_wavelength_m + 8 * fwhm,
            101,
        )
        model = LTESpectralModel(
            saha_solver=saha_solver,
            wavelength_m=grid,
            transitions=(resonance_transition,),
            atomic_masses_kg=atomic_masses_kg,
            include_continuum=False,
        )
        integrator = GateIntegrator(
            spectral_model=model,
            evolution=UniformPlasmaEvolution(
                temperature_K=Constant(1.0e4),
                heavy_density_m3=ExponentialDecay(1.0e16, 5.0e-7),
                composition={"Fe": 1.0},
                electron_density_m3=Constant(1.0e20),
            ),
            impact_parameter_m=0.0,
        )
        spectra = sweep_gate_delay(
            integrator, [0.0, 5.0e-7, 1.5e-6], 2.0e-7, n_time_nodes=4
        )
        peaks = [s.intensity.max() for s in spectra]
        assert peaks[0] > peaks[1] > peaks[2]
        assert [s.metadata["gate_delay_s"] for s in spectra] == [
            0.0, 5.0e-7, 1.5e-6,
        ]


class TestPlotting:
    def test_plot_returns_axes_with_lines(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        a = make_spectrum([1.0, 2.0, 1.0])
        b = Spectrum(
            wavelength_m=a.wavelength_m,
            intensity=np.array([2.0, 1.0, 2.0]),
            metadata={"gate_delay_s": 1e-6},
        )
        ax = plot_spectra(a, b, normalize=True, title="test")
        assert len(ax.get_lines()) == 2
        labels = [line.get_label() for line in ax.get_lines()]
        assert labels[0] == "spectrum 0"
        assert labels[1].startswith("t_d =")
        assert ax.get_ylabel() == "normalized intensity"
        plt.close(ax.figure)

    def test_plot_validation(self):
        pytest.importorskip("matplotlib")
        a = make_spectrum([1.0, 2.0])
        with pytest.raises(ValueError, match="at least one"):
            plot_spectra()
        with pytest.raises(ValueError, match="labels"):
            plot_spectra(a, labels=["one", "two"])
        with pytest.raises(ValueError, match="wavelength_unit"):
            plot_spectra(a, wavelength_unit="cubit")
