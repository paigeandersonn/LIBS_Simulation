"""
Tests for libssim.validation — atomic data sanity, preprocessing,
metrics and the end-to-end workflow (surrogate round trip).
"""

from __future__ import annotations

import numpy as np
import pytest

from libssim.core.constants import C, EV, H
from libssim.core.spectrum import Spectrum
from libssim.validation import (
    InstrumentSettings,
    PlasmaConditions,
    ValidationCase,
    aluminum_setup,
    compute_metrics,
    crop,
    find_peak_positions,
    intensity_ratio,
    match_peaks,
    noise_study,
    normalize,
    sodium_setup,
    subtract_background,
    surrogate_experiment,
)
from libssim.temporal.decay_models import PowerLawDecay


class TestAtomicData:
    def test_sodium_partition_functions(self):
        setup = sodium_setup()
        provider = setup.saha_solver.partition_provider
        # Improved (n <= 6) list: U_NaI(10 kK) ~ 3.3; Na II closed
        # shell U = 1.
        assert 2.8 < provider.partition_function("Na", 1, 1.0e4) < 3.5
        assert provider.partition_function("Na", 2, 1.0e4) == pytest.approx(
            1.0, abs=1e-12
        )

    def test_aluminum_partition_functions(self):
        setup = aluminum_setup()
        provider = setup.saha_solver.partition_provider
        # Literature: U_AlI(10 kK) ~ 5.8-6.1.
        assert 5.5 < provider.partition_function("Al", 1, 1.0e4) < 6.6
        assert provider.partition_function("Al", 2, 1.0e4) == pytest.approx(
            1.0, rel=0.05
        )

    @pytest.mark.parametrize("setup_fn", [sodium_setup, aluminum_setup])
    def test_air_wavelengths_consistent_with_level_energies(self, setup_fn):
        # E_upper - E_lower vs h*c/lambda_air differ only by the air
        # refractive index (~2.7e-4) — catches transcription errors.
        for tr in setup_fn().transitions:
            photon_ev = H * C / tr.wavelength_m / EV
            delta_ev = tr.energy_upper_ev - tr.energy_lower_ev
            assert photon_ev == pytest.approx(delta_ev, rel=5e-4)

    def test_doublet_structure(self):
        na = sodium_setup().transitions
        assert [t.g_upper for t in na] == [4, 2]  # D2, D1
        al = aluminum_setup().transitions
        # Shared upper level (4s 2S1/2) for the Al doublet.
        assert al[0].energy_upper_ev == al[1].energy_upper_ev


class TestImprovedAtomicData:
    def test_extended_levels_raise_high_temperature_u(self):
        from libssim.physics.partition import partition_function_from_levels
        from libssim.validation.atomic_data import (
            NA_I_LEVELS,
            NA_I_LEVELS_MINIMAL,
        )

        def u(levels, T):
            return partition_function_from_levels(
                [g for _, g in levels], [e for e, _ in levels], T
            )

        # Cold: identical (added levels frozen out). Hot: improved > minimal.
        assert u(NA_I_LEVELS, 3000.0) == pytest.approx(
            u(NA_I_LEVELS_MINIMAL, 3000.0), rel=1e-6
        )
        assert u(NA_I_LEVELS, 1.5e4) > 1.02 * u(NA_I_LEVELS_MINIMAL, 1.5e4)

    def test_load_levels_csv_ev_and_cm1(self, tmp_path):
        from libssim.validation import load_levels_csv

        path_ev = tmp_path / "levels_ev.csv"
        path_ev.write_text("# E(eV),g\n2.10,4\n0.0,2\n3.19,2\n")
        levels = load_levels_csv(path_ev)
        assert levels[0] == (0.0, 2.0)          # sorted by energy
        assert levels[1] == (2.10, 4.0)

        path_cm = tmp_path / "levels_cm.csv"
        path_cm.write_text("0.0,2\n16956.17,2\n")  # Na 3p 2P1/2 in cm^-1
        levels_cm = load_levels_csv(path_cm, energy_unit="cm-1")
        assert levels_cm[1][0] == pytest.approx(2.1023, rel=1e-3)

        with pytest.raises(ValueError, match="energy_unit"):
            load_levels_csv(path_ev, energy_unit="joules")

    def test_hydrogenic_tail_levels(self):
        from libssim.validation import hydrogenic_tail_levels

        tail = hydrogenic_tail_levels(5.139077, 7, 8)
        assert tail[0][1] == 98.0                      # 2 n^2 at n = 7
        assert tail[0][0] == pytest.approx(
            5.139077 - 13.605693 / 49.0, rel=1e-9
        )
        assert tail[1][1] == 128.0
        with pytest.raises(ValueError, match="n_min"):
            hydrogenic_tail_levels(5.139, 1, 3)
        with pytest.raises(ValueError, match="below the ground state"):
            hydrogenic_tail_levels(1.0, 2, 3)  # chi - R/4 < 0

    def test_fit_irwin_polynomial_roundtrip(self):
        from libssim.physics.partition import partition_function_from_levels
        from libssim.validation import fit_irwin_polynomial
        from libssim.validation.atomic_data import NA_I_LEVELS

        poly = fit_irwin_polynomial(NA_I_LEVELS)
        for T in (2000.0, 8000.0, 15000.0, 25000.0):
            exact = partition_function_from_levels(
                [g for _, g in NA_I_LEVELS],
                [e for e, _ in NA_I_LEVELS],
                T,
            )
            # <~0.3% residual: limited by the freeze-out knee (fitter
            # docstring), fine for a beyond-table fallback.
            assert poly(T) == pytest.approx(exact, rel=4e-3)

    def test_polynomial_fallback_extends_narrow_table(self):
        # Table only to 12 kK; the registered Irwin-form fallback must
        # cover the rest of its validity range through the provider.
        from libssim.validation import SODIUM_DATA, build_setup

        setup = build_setup(
            SODIUM_DATA, temperature_grid_K=np.linspace(1000.0, 12000.0, 100)
        )
        provider = setup.saha_solver.partition_provider
        u_hot = provider.partition_function("Na", 1, 2.0e4)  # beyond table
        assert np.isfinite(u_hot) and u_hot > 3.5

        bare = build_setup(
            SODIUM_DATA,
            temperature_grid_K=np.linspace(1000.0, 12000.0, 100),
            include_polynomial_fallback=False,
        )
        with pytest.raises(ValueError, match="undefined at"):
            bare.saha_solver.partition_provider.partition_function(
                "Na", 1, 2.0e4
            )

    def test_setup_carries_provenance_and_verify_notes(self):
        setup = sodium_setup()
        assert "NIST" in setup.provenance
        assert any("Rydberg" in note for note in setup.verify_notes)


class TestPreprocessing:
    GRID = 588.0e-9 + np.arange(200) * 1.0e-11

    def spectrum_with_baseline(self, slope=0.0, offset=100.0):
        line = 1000.0 * np.exp(
            -0.5 * ((self.GRID - 589.0e-9) / 3e-11) ** 2
        )
        baseline = offset + slope * (self.GRID - self.GRID[0])
        return Spectrum(wavelength_m=self.GRID, intensity=line + baseline)

    def test_constant_background_removed_exactly(self):
        spectrum = self.spectrum_with_baseline(offset=250.0)
        windows = [(self.GRID[0], self.GRID[20]), (self.GRID[-21], self.GRID[-1])]
        cleaned = subtract_background(spectrum, windows, fit="constant")
        # Window regions are line-free: baseline recovered exactly.
        assert np.allclose(cleaned.intensity[:10], 0.0, atol=1e-9)
        assert cleaned.metadata["background_fit"] == "constant"

    def test_linear_background_removed_exactly(self):
        spectrum = self.spectrum_with_baseline(slope=5.0e12, offset=80.0)
        windows = [(self.GRID[0], self.GRID[20]), (self.GRID[-21], self.GRID[-1])]
        cleaned = subtract_background(spectrum, windows, fit="linear")
        assert np.allclose(cleaned.intensity[:10], 0.0, atol=1e-6)
        assert np.allclose(cleaned.intensity[-10:], 0.0, atol=1e-6)

    def test_normalize_modes(self):
        spectrum = self.spectrum_with_baseline(offset=0.0)
        peak = normalize(spectrum, "peak")
        assert peak.intensity.max() == pytest.approx(1.0)
        area = normalize(spectrum, "area")
        from scipy.integrate import trapezoid

        assert trapezoid(area.intensity, area.wavelength_m) == pytest.approx(
            1.0, rel=1e-12
        )

    def test_crop_bounds_and_validation(self):
        spectrum = self.spectrum_with_baseline()
        cropped = crop(spectrum, 588.5e-9, 589.5e-9)
        assert cropped.wavelength_m[0] >= 588.5e-9
        assert cropped.wavelength_m[-1] <= 589.5e-9
        with pytest.raises(ValueError, match="fewer than 2"):
            crop(spectrum, 700e-9, 701e-9)
        with pytest.raises(ValueError):
            subtract_background(spectrum, [], fit="constant")


class TestMetrics:
    def two_line_spectrum(self, ratio=2.0):
        grid = 588.0e-9 + np.arange(400) * 5.0e-12
        intensity = ratio * np.exp(
            -0.5 * ((grid - 588.995e-9) / 2e-11) ** 2
        ) + np.exp(-0.5 * ((grid - 589.592e-9) / 2e-11) ** 2)
        return Spectrum(wavelength_m=grid, intensity=intensity)

    def test_peak_finding_and_matching(self):
        spectrum = self.two_line_spectrum()
        found = find_peak_positions(spectrum, min_relative_height=0.1)
        assert len(found) == 2
        matches = match_peaks(
            found, [588.995e-9, 589.592e-9], tolerance_m=2e-11
        )
        assert all(m.matched for m in matches)
        offsets = [abs(m.offset_m) for m in matches]
        assert max(offsets) < 5e-12  # grid-resolution accuracy

    def test_unmatched_peak_reported(self):
        matches = match_peaks([500e-9], [600e-9], tolerance_m=1e-10)
        assert not matches[0].matched
        assert matches[0].offset_m is None

    def test_intensity_ratio(self):
        spectrum = self.two_line_spectrum(ratio=2.0)
        ratio = intensity_ratio(
            spectrum, 588.995e-9, 589.592e-9, half_window_m=5e-11
        )
        # Grid sampling offsets each peak by up to ~0.8% (step/sigma).
        assert ratio == pytest.approx(2.0, rel=2e-2)

    def test_compute_metrics_perfect_match(self):
        spectrum = self.two_line_spectrum()
        metrics = compute_metrics(
            spectrum, spectrum, [588.995e-9, 589.592e-9]
        )
        assert metrics.r_correlation == pytest.approx(1.0, abs=1e-12)
        assert metrics.rms_residual == 0.0
        assert metrics.n_matched == 2
        assert "R (Eq. 5-56)" in metrics.summary()


@pytest.fixture(scope="module")
def sodium_case():
    setup = sodium_setup()
    grid = np.linspace(587.5e-9, 591.0e-9, 701)
    conditions = PlasmaConditions(
        temperature_K=PowerLawDecay(1.05e4, 1.0e-6, 0.4),
        heavy_density_m3=5.0e20,
        electron_density_m3=PowerLawDecay(1.0e23, 1.0e-6, 1.0),
        gate_delay_s=1.0e-6,
        gate_width_s=1.0e-6,
        n_time_nodes=6,
    )
    instrument = InstrumentSettings(
        slit_width_um=50.0, read_noise_rms_counts=10.0, dark_mean_counts=100.0
    )
    return ValidationCase(
        name="Na I D doublet (test)",
        setup=setup,
        conditions=conditions,
        instrument=instrument,
        wavelength_m=grid,
        n_impact=16,
    )


class TestWorkflow:
    def test_surrogate_round_trip_high_correlation(self, sodium_case):
        # Matched conditions: only noise separates the two spectra.
        surrogate = surrogate_experiment(sodium_case, seed=20260711)
        assert surrogate.metadata["surrogate"] is True
        result = sodium_case.validate(surrogate)
        assert result.is_surrogate
        assert result.metrics.r_correlation > 0.98
        assert result.metrics.n_matched == 2
        assert "SURROGATE" in result.report()

    def test_wrong_temperature_degrades_correlation(self, sodium_case):
        from dataclasses import replace

        surrogate = surrogate_experiment(sodium_case, seed=20260711)
        matched = sodium_case.validate(surrogate).metrics.r_correlation
        colder = replace(
            sodium_case,
            conditions=replace(
                sodium_case.conditions,
                temperature_K=PowerLawDecay(7.0e3, 1.0e-6, 0.4),
            ),
        )
        mismatched = colder.validate(surrogate).metrics.r_correlation
        assert matched > mismatched

    def test_noisy_synthetic_reproducible(self, sodium_case):
        a = sodium_case.synthetic_noisy(seed=5)
        b = sodium_case.synthetic_noisy(seed=5)
        assert np.array_equal(a.intensity, b.intensity)

    def test_nonuniform_grid_rejected(self, sodium_case):
        from dataclasses import replace

        bad_grid = np.concatenate(
            [np.linspace(588e-9, 589e-9, 50),
             np.linspace(589.01e-9, 591e-9, 30)]
        )
        with pytest.raises(ValueError, match="uniform"):
            replace(sodium_case, wavelength_m=bad_grid)

    D2_M, D1_M = 588.99509e-9, 589.59237e-9
    BG_WINDOWS = ((587.5e-9, 588.2e-9), (590.4e-9, 591.0e-9))

    def test_realistic_surrogate_with_background(self, sodium_case):
        # Low-SNR surrogate + pedestal: R must be meaningfully below 1
        # yet high, and the background must be removed by the pipeline.
        surrogate = surrogate_experiment(
            sodium_case,
            seed=7,
            peak_counts=4.0e3,
            read_noise_rms_counts=15.0,
            background_mean_counts=250.0,
        )
        # Raw surrogate edges sit on dark (100) + background (250).
        assert surrogate.intensity[:30].mean() > 300.0
        result = sodium_case.validate(
            surrogate, background_windows=self.BG_WINDOWS
        )
        # Peak-normalized, background-subtracted edges scatter around 0.
        assert abs(float(result.experimental.intensity[:30].mean())) < 0.05
        assert 0.9 < result.metrics.r_correlation < 0.9999

    def test_describe_lists_all_conditions(self, sodium_case):
        text = sodium_case.describe()
        for token in (
            "gate delay",
            "gate width",
            "slit width",
            "PowerLawDecay",
            "instrumental FWHM",
            "Na",
        ):
            assert token in text

    def test_noise_study_statistics(self, sodium_case):
        study = noise_study(
            sodium_case,
            seeds=[1, 2, 3],
            line_a_m=self.D2_M,
            line_b_m=self.D1_M,
            half_window_m=1.5e-10,
            peak_counts=4.0e3,
            read_noise_rms_counts=15.0,
            background_mean_counts=250.0,
            background_windows=self.BG_WINDOWS,
        )
        assert len(study.r_values) == 3
        assert len(study.ratio_values) == 3
        assert 0.9 < study.r_mean < 1.0
        assert study.ratio_std >= 0.0
        assert study.clean_ratio > 1.0  # D2 stronger than D1
        assert "R (Eq. 5-56)" in study.summary()

    def test_noise_study_requires_two_seeds(self, sodium_case):
        with pytest.raises(ValueError, match="at least 2 seeds"):
            noise_study(
                sodium_case,
                seeds=[1],
                line_a_m=self.D2_M,
                line_b_m=self.D1_M,
                half_window_m=1.5e-10,
            )

    def test_self_absorption_reduces_doublet_ratio(self):
        # Optically thin -> D2/D1 ~ g_u ratio = 2; raising the Na column
        # saturates D2 first, pulling the ratio toward 1 (Fig. 3-2
        # behaviour) — the qualitative self-absorption check.
        setup = sodium_setup()
        grid = np.linspace(587.5e-9, 591.0e-9, 701)
        instrument = InstrumentSettings(slit_width_um=50.0)

        def ratio_at(heavy: float) -> float:
            case = ValidationCase(
                name="ratio",
                setup=setup,
                conditions=PlasmaConditions(
                    temperature_K=1.0e4,
                    heavy_density_m3=heavy,
                    electron_density_m3=1.0e23,
                    gate_delay_s=1.0e-6,
                    gate_width_s=5.0e-7,
                    n_time_nodes=2,
                ),
                instrument=instrument,
                wavelength_m=grid,
                include_continuum=False,
                n_impact=8,
            )
            return intensity_ratio(
                case.synthetic_noise_free(),
                588.99509e-9,
                589.59237e-9,
                half_window_m=1.5e-10,
            )

        thin = ratio_at(1.0e18)
        thick = ratio_at(5.0e22)
        assert thin == pytest.approx(2.0, rel=0.1)
        assert thick < thin  # self-absorption compresses the doublet
