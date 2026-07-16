"""Robust parser for NIST-style CSV exports.

Includes defensive cleaning that guarantees the Phase 1 acceptance
criteria: clean list of Transition objects with no missing critical fields.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .base import AtomicDatabase
from .transition import Transition

logger = logging.getLogger(__name__)


def _normalize_element(name: str) -> str:
    name = name.strip().capitalize()
    mapping = {"Cerium": "Ce", "Ce": "Ce"}
    return mapping.get(name, name[:2].capitalize())


def parse_nist_csv(
    csv_path: str | Path,
    element: Optional[str] = None,
    ion_stage: Optional[int] = None,
) -> List[Transition]:
    """
    Parse a NIST-style transition CSV into clean Transition objects.
    This is the core function for Phase 1 acceptance criteria.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(
        csv_path,
        comment="#",
        skipinitialspace=True,
        na_values=["", " ", "nan", "NaN", "N/A", "-"],
        engine="python",
    )
    df = df.dropna(how="all")
    logger.info(f"Loaded {len(df)} rows from {csv_path.name}")

    # Column normalization for common NIST export variations
    col_map = {
        "Wavelength (nm)": "wavelength_nm",
        "Wavelength": "wavelength_nm",
        "Ei(eV)": "energy_lower_ev",
        "Ek(eV)": "energy_upper_ev",
        "Aki(s^-1)": "a_ki",
        "Aki": "a_ki",
        "gi": "g_lower",
        "gk": "g_upper",
        "Spectrum": "spectrum",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Handle "Ce II" style spectrum column
    if "spectrum" in df.columns:
        def parse_spectrum(s):
            if pd.isna(s):
                return None, None
            s = str(s).strip()
            parts = s.split()
            if len(parts) >= 2:
                elem = _normalize_element(parts[0])
                roman = parts[1].upper()
                ion = {"I": 1, "II": 2, "III": 3, "IV": 4}.get(roman, 1)
                return elem, ion
            return _normalize_element(s), 1

        df[["parsed_element", "parsed_ion"]] = df["spectrum"].apply(
            lambda x: pd.Series(parse_spectrum(x))
        )
        if "element" not in df.columns:
            df["element"] = df["parsed_element"]
        if "ion_stage" not in df.columns:
            df["ion_stage"] = df["parsed_ion"]

    required = ["wavelength_nm", "energy_lower_ev", "energy_upper_ev", "a_ki", "g_lower", "g_upper"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    # === CLEANING (BUG FIXED) ===
    initial_count = len(df)

    # Only convert numeric columns — do NOT touch "element"
    numeric_cols = required + ["ion_stage"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df_clean = df.dropna(subset=required)

    dropped = initial_count - len(df_clean)
    if dropped > 0:
        logger.warning(
            f"Dropped {dropped} rows ({100 * dropped / initial_count:.1f}%) "
            "due to missing critical fields."
        )

    # Apply optional filters
    if element is not None:
        elem_norm = _normalize_element(element)
        df_clean = df_clean[df_clean["element"].str.upper() == elem_norm.upper()]

    if ion_stage is not None:
        df_clean = df_clean[df_clean["ion_stage"] == ion_stage]

    # Build Transition objects
    transitions: List[Transition] = []
    for _, row in df_clean.iterrows():
        try:
            t = Transition(
                element=str(row.get("element", "Unknown")).strip(),
                ion_stage=int(row["ion_stage"]),
                wavelength_m=float(row["wavelength_nm"]) * 1e-9,
                energy_lower_ev=float(row["energy_lower_ev"]),
                energy_upper_ev=float(row["energy_upper_ev"]),
                a_ki=float(row["a_ki"]),
                g_lower=int(row["g_lower"]),
                g_upper=int(row["g_upper"]),
            )
            transitions.append(t)
        except (ValueError, TypeError):
            pass

    logger.info(f"Successfully parsed {len(transitions)} clean Transition objects.")
    return transitions


class CSVAtomicDatabase(AtomicDatabase):
    """Concrete implementation backed by a local CSV file."""

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self._cache: List[Transition] | None = None

    def get_transitions(
        self, element: str, ion_stage: Optional[int] = None
    ) -> List[Transition]:
        if self._cache is None:
            self._cache = parse_nist_csv(self.csv_path)

        elem_norm = _normalize_element(element)
        result = [
            t for t in self._cache
            if t.element.upper() == elem_norm.upper()
            and (ion_stage is None or t.ion_stage == ion_stage)
        ]
        if not result:
            raise ValueError(
                f"No transitions found for element={element}, ion_stage={ion_stage}"
            )
        return result

    def __repr__(self) -> str:
        return f"CSVAtomicDatabase({self.csv_path.name})"