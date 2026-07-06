"""
tests/unit/test_state.py
========================
Comprehensive unit tests for libssim.core.state.PlasmaState

These tests verify:
- Immutability (frozen dataclass)
- Physical validations
- Composition normalization (stoichiometric ablation)
- Convenience properties
- Correct behavior matching Herrera (2008) MC-LIBS / CF-LIBS usage
"""

import pytest
from dataclasses import FrozenInstanceError

from libssim.core.state import PlasmaState


def make_state(**overrides):
    base = dict(
        temperature_K=10000.0,
        electron_density_m3=1e23,
        total_density_m3=2e23,
        radius_m=1e-3,
        time_s=1e-6,
        composition={"Ce": 2.0, "O": 1.0},
    )
    base.update(overrides)
    return PlasmaState(**base)


def test_state_is_immutable():
    s = make_state()
    with pytest.raises(FrozenInstanceError):
        s.temperature_K = 20000.0


def test_composition_is_normalized_to_sum_1():
    s = make_state(composition={"A": 2.0, "B": 2.0})
    assert abs(sum(s.composition.values()) - 1.0) < 1e-12
    assert s.composition["A"] == pytest.approx(0.5)
    assert s.composition["B"] == pytest.approx(0.5)


def test_composition_must_sum_positive():
    with pytest.raises(ValueError, match="must sum to a positive value"):
        make_state(composition={"A": 0.0, "B": 0.0})


def test_composition_negative_value_rejected():
    with pytest.raises(ValueError, match="must be >= 0"):
        make_state(composition={"A": -1.0, "B": 3.0})


def test_empty_composition_allowed_but_empty():
    s = make_state(composition={})
    assert s.composition == {}


def test_electron_density_cannot_exceed_total_density():
    with pytest.raises(ValueError, match="cannot exceed total_density_m3"):
        make_state(electron_density_m3=3e23, total_density_m3=2e23)


def test_scalar_validations():
    with pytest.raises(ValueError, match="temperature_K must be > 0"):
        make_state(temperature_K=-1.0)
    with pytest.raises(ValueError, match="radius_m must be > 0"):
        make_state(radius_m=0.0)
    with pytest.raises(ValueError, match="time_s must be >= 0"):
        make_state(time_s=-1e-6)


def test_aliases_and_species():
    s = make_state(composition={"Al": 0.9, "Mg": 0.1})
    assert s.n_e == s.electron_density_m3
    assert s.n_tot == s.total_density_m3
    assert s.species == ["Al", "Mg"]


def test_repr_is_readable():
    s = make_state(temperature_K=12000.0, radius_m=0.5e-3, time_s=5e-6)
    r = repr(s)
    assert "T=12000.0 K" in r
    assert "R=0.50 mm" in r


def test_state_with_unit_conversion():
    from libssim.core.units import ev_to_k, nm_to_m
    s = make_state(temperature_K=ev_to_k(1.0))
    assert s.temperature_K == pytest.approx(11604.518, rel=1e-4)
    assert nm_to_m(656.272) == pytest.approx(6.56272e-7)


def test_single_species_plasma():
    s = make_state(composition={"Pu": 1.0})
    assert s.composition == {"Pu": 1.0}


def test_high_ionization_plasma():
    s = make_state(electron_density_m3=1.999e23, total_density_m3=2e23)
    assert s.n_e < s.n_tot