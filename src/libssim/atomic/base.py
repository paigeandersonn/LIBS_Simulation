"""Abstract base class defining the interface for atomic line
databases.

This abstraction keeps the physics engine agnostic to the data source
(CSV, NIST web, Blaise, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from .transition import Transition


class AtomicDatabase(ABC):
    """Abstract interface for atomic transition databases."""

    @abstractmethod
    def get_transitions(
        self,
        element: str,
        ion_stage: Optional[int] = None,
    ) -> List[Transition]:
        """
        Return clean Transition objects for the requested element.
        Implementations must guarantee no None values in critical fields.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"