"""
tests/unit/test_atomic.py
=========================
Unit tests for Phase 1: Atomic Database Abstraction.
"""

import io
from pathlib import Path

import pandas as pd
import pytest

from libssim.atomic import (
    Transition,
    AtomicDatabase,
    parse_nist_csv,
    CSVAtomicDatabase,
)


# ============================================================
# Sample test data (in-memory CSV)
# ============================================================

SAMPLE_CSV = """Spectrum,Wavelength (nm),Ei(eV),Ek(eV),Aki(s^-1),gi,gk
Ce I,500.123,0.000,2.479,1.23e7,1,3
Ce II,510.456,0.500,2.930,4.56e6,2,4
Ce II,520.789,1.200,3.580,8.90e6,4,6
Ce I,999.999,0.000,1.240,,,,   # bad row - missing A_ki and g values
Fe I,400.000,0.000,3.100,2.50e7,1,3
"""


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    """Create a temporary CSV file for testing."""
    csv_file = tmp_path / "test_cerium.csv"
    csv_file.write_text(SAMPLE_CSV)
    return csv_file


# ============================================================
# Transition Dataclass Tests
# ============================================================

class TestTransition:
    def test_valid_transition_creation(self):
        t = Transition(
            element="Ce",
            ion_stage=2,
            wavelength_m=5.10123e-7,
            energy_lower_ev=0.5,
            energy_upper_ev=2.93,
            a_ki=4.56e6,
            g_lower=2,
            g_upper=4,
        )
        assert t.element == "Ce"
        assert t.ion_stage == 2
        assert t.wavelength_nm == pytest.approx(510.123, abs=0.01)
        assert t.energy_diff_ev == pytest.approx(2.43)

    def test_transition_is_frozen(self):
        t = Transition("Ce", 1, 5e-7, 0, 2, 1e7, 1, 3)
        with pytest.raises(AttributeError):
            t.element = "Fe"  # Should fail because frozen=True

    def test_transition_validation_errors(self):
        # Negative wavelength
        with pytest.raises(ValueError, match="wavelength_m must be positive"):
            Transition("Ce", 1, -1e-7, 0, 2, 1e7, 1, 3)

        # Negative A_ki
        with pytest.raises(ValueError, match="a_ki must be non-negative"):
            Transition("Ce", 1, 5e-7, 0, 2, -100, 1, 3)

        # Upper energy lower than lower energy
        with pytest.raises(ValueError, match="energy_upper_ev must be >="):
            Transition("Ce", 1, 5e-7, 3, 1, 1e7, 1, 3)

    def test_transition_repr(self):
        t = Transition("Ce", 2, 5.10456e-7, 0.5, 2.93, 4.56e6, 2, 4)
        repr_str = repr(t)
        assert "Ce 2" in repr_str
        assert "510.4560 nm" in repr_str          # Match actual formatting
        assert "A=4.56e+06" in repr_str           # Also check A coefficient


# ============================================================
# parse_nist_csv Tests
# ============================================================

class TestParseNistCsv:
    def test_basic_parsing(self, sample_csv_path):
        transitions = parse_nist_csv(sample_csv_path)
        assert len(transitions) == 4  # 1 bad row should be dropped

    def test_filters_by_element_and_ion_stage(self, sample_csv_path):
        ce_ii = parse_nist_csv(sample_csv_path, element="Ce", ion_stage=2)
        assert len(ce_ii) == 2
        assert all(t.element == "Ce" and t.ion_stage == 2 for t in ce_ii)

    def test_drops_rows_with_missing_critical_fields(self, sample_csv_path):
        transitions = parse_nist_csv(sample_csv_path)
        # The row with missing A_ki and g values should be dropped
        assert len(transitions) == 4

    def test_handles_spectrum_column(self, sample_csv_path):
        ce_ii = parse_nist_csv(sample_csv_path, element="Ce", ion_stage=2)
        assert len(ce_ii) == 2

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_nist_csv("nonexistent_file.csv")

    def test_returns_clean_transitions_no_none_values(self, sample_csv_path):
        transitions = parse_nist_csv(sample_csv_path)
        for t in transitions:
            assert t.wavelength_m is not None
            assert t.energy_lower_ev is not None
            assert t.a_ki is not None
            assert t.g_lower is not None


# ============================================================
# CSVAtomicDatabase Tests
# ============================================================

class TestCSVAtomicDatabase:
    def test_get_transitions(self, sample_csv_path):
        db = CSVAtomicDatabase(sample_csv_path)
        transitions = db.get_transitions("Ce", ion_stage=2)
        assert len(transitions) == 2

    def test_caching(self, sample_csv_path, mocker):
        """Test that CSVAtomicDatabase caches parsed data and doesn't re-parse on every call."""
        import libssim.atomic.parsers as parsers

        db = CSVAtomicDatabase(sample_csv_path)

        # Spy on the parse function to count how many times it's called
        parse_spy = mocker.spy(parsers, "parse_nist_csv")

        # First call — should parse the file
        result1 = db.get_transitions("Ce")
        assert len(result1) > 0

        # Second call — should use cache (parse should not be called again)
        result2 = db.get_transitions("Ce")
        assert len(result2) > 0

        # The parse function should only have been called **once**
        assert parse_spy.call_count == 1

    def test_raises_when_no_results(self, sample_csv_path):
        db = CSVAtomicDatabase(sample_csv_path)
        with pytest.raises(ValueError, match="No transitions found"):
            db.get_transitions("Uranium")  # Not in test data


# ============================================================
# Abstract Base Class Contract Test
# ============================================================

def test_atomic_database_is_abstract():
    with pytest.raises(TypeError):
        AtomicDatabase()  # Should not be instantiable