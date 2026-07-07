"""
libssim.atomic
==============
Atomic data abstraction layer (Phase 1).

Provides:
- Transition dataclass (immutable core data structure)
- AtomicDatabase abstract base class
- CSV parser + concrete implementation (ready for NIST/Blaise extensions)
"""

from .transition import Transition
from .base import AtomicDatabase
from .parsers import parse_nist_csv, CSVAtomicDatabase

__all__ = [
    "Transition",
    "AtomicDatabase",
    "parse_nist_csv",
    "CSVAtomicDatabase",
]