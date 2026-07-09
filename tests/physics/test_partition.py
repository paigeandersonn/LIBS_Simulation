"""
Tests for libssim.physics.partition (Herrera 2008: U(T) of Eqs. 5-1,
5-2 p. 98; 5-8 pp. 103-104; D-1 p. 274; direct-sum definition p. 260).
"""

from __future__ import annotations

import numpy as np
import pytest

from libssim.core.constants import KB_EV
from libssim.physics.partition import (
    PartitionFunctionPolynomial,
    PartitionFunctionProvider,
    PartitionFunctionTable,
    partition_function_from_levels,
)

G = [2.0, 4.0, 6.0]
E_EV = [0.0, 1.0, 3.0]


class TestDirectSummation:
    def test_matches_hand_computation(self):
        T = 10000.0
        expected = sum(
            g * np.exp(-e / (KB_EV * T)) for g, e in zip(G, E_EV)
        )
        assert partition_function_from_levels(G, E_EV, T) == pytest.approx(
            expected, rel=1e-12
        )

    def test_low_temperature_limit_is_ground_degeneracy(self):
        # U(T -> 0) -> g_0 (validation identity, module docs)
        assert partition_function_from_levels(G, E_EV, 50.0) == pytest.approx(
            G[0], abs=1e-12
        )

    def test_monotonically_increasing_in_temperature(self):
        U = partition_function_from_levels(G, E_EV, np.linspace(2e3, 2e4, 30))
        assert np.all(np.diff(U) > 0)

    def test_vectorized_shape_preserved(self):
        T = np.array([[5.0e3, 1.0e4], [1.5e4, 2.0e4]])
        U = partition_function_from_levels(G, E_EV, T)
        assert U.shape == T.shape

    def test_scalar_input_returns_float(self):
        assert isinstance(partition_function_from_levels(G, E_EV, 1e4), float)

    @pytest.mark.parametrize(
        "g, e_ev, T",
        [
            ([2.0], [0.0, 1.0], 1e4),   # mismatched lengths
            ([0.0, 2.0], [0.0, 1.0], 1e4),  # non-positive weight
            ([2.0, 4.0], [0.0, -1.0], 1e4),  # negative energy
            (G, E_EV, -5.0),            # non-positive temperature
            (G, E_EV, np.nan),          # non-finite temperature
        ],
    )
    def test_invalid_inputs_rejected(self, g, e_ev, T):
        with pytest.raises(ValueError):
            partition_function_from_levels(g, e_ev, T)


@pytest.fixture(scope="module")
def table() -> PartitionFunctionTable:
    grid = np.linspace(1000.0, 20000.0, 400)
    return PartitionFunctionTable.from_levels(G, E_EV, grid)


@pytest.fixture(scope="module")
def polynomial() -> PartitionFunctionPolynomial:
    # Irwin-form fit ln U = poly(ln T) to the exact direct sum.
    grid = np.linspace(1000.0, 20000.0, 400)
    U = partition_function_from_levels(G, E_EV, grid)
    coeffs = np.polynomial.polynomial.polyfit(np.log(grid), np.log(U), 7)
    return PartitionFunctionPolynomial(
        coefficients=tuple(coeffs), temperature_range_K=(1000.0, 30000.0)
    )


@pytest.fixture(scope="module")
def provider(
    table: PartitionFunctionTable, polynomial: PartitionFunctionPolynomial
) -> PartitionFunctionProvider:
    return PartitionFunctionProvider(
        tables={("Ce", 1): table},
        polynomials={("ce", 1): polynomial},  # case-insensitive key merge
    )


class TestPartitionFunctionTable:
    def test_interpolation_matches_direct_sum(self, table):
        # Linear-interpolation error is bounded by the grid spacing:
        # ~1e-5 on this 400-point grid (module docs: refine the grid,
        # not the method), so 1e-4 has an order-of-magnitude margin.
        T = np.linspace(1500.0, 19500.0, 57)
        exact = partition_function_from_levels(G, E_EV, T)
        rel = np.max(np.abs(table(T) - exact) / exact)
        assert rel < 1e-4

    def test_exact_on_grid_nodes(self, table):
        T_node = float(table.temperatures_K[123])
        exact = partition_function_from_levels(G, E_EV, T_node)
        assert table(T_node) == pytest.approx(exact, rel=1e-14)

    def test_rejects_extrapolation(self, table):
        with pytest.raises(ValueError, match="outside tabulated range"):
            table(25000.0)
        with pytest.raises(ValueError, match="outside tabulated range"):
            table(500.0)

    def test_arrays_are_read_only(self, table):
        with pytest.raises(ValueError):
            table.values[0] = 99.0

    @pytest.mark.parametrize(
        "temps, values",
        [
            ([1000.0], [5.0]),                    # fewer than 2 points
            ([1000.0, 900.0], [5.0, 5.0]),        # non-increasing grid
            ([1000.0, 2000.0], [5.0, -1.0]),      # non-positive U
            ([1000.0, 2000.0], [5.0, np.inf]),    # non-finite U
        ],
    )
    def test_invalid_construction_rejected(self, temps, values):
        with pytest.raises(ValueError):
            PartitionFunctionTable(
                temperatures_K=np.asarray(temps), values=np.asarray(values)
            )


class TestPartitionFunctionPolynomial:
    def test_roundtrip_against_direct_sum(self, polynomial):
        # Bounds the degree-7 ln-ln fit residual (~4e-4 for this toy
        # species), i.e. the intrinsic accuracy of the fallback path.
        T = np.linspace(1500.0, 19500.0, 57)
        exact = partition_function_from_levels(G, E_EV, T)
        rel = np.max(np.abs(polynomial(T) - exact) / exact)
        assert rel < 1e-3

    def test_validity_range_enforced(self, polynomial):
        with pytest.raises(ValueError, match="validity range"):
            polynomial(50000.0)

    @pytest.mark.parametrize(
        "coeffs, rng",
        [
            ((), (1e3, 2e4)),               # empty coefficients
            ((1.0, np.nan), (1e3, 2e4)),    # non-finite coefficient
            ((1.0,), (2e4, 1e3)),           # inverted range
            ((1.0,), (-1.0, 1e3)),          # non-positive lower bound
        ],
    )
    def test_invalid_construction_rejected(self, coeffs, rng):
        with pytest.raises(ValueError):
            PartitionFunctionPolynomial(
                coefficients=coeffs, temperature_range_K=rng
            )


class TestPartitionFunctionProvider:
    def test_table_is_primary_polynomial_is_fallback(self, provider):
        table = provider.tables[("CE", 1)]
        poly = provider.polynomials[("CE", 1)]
        U = provider.partition_function("CE", 1, np.array([5000.0, 25000.0]))
        assert U[0] == pytest.approx(table(5000.0), rel=1e-14)   # in grid
        assert U[1] == pytest.approx(poly(25000.0), rel=1e-14)   # fallback

    def test_scalar_returns_float(self, provider):
        assert isinstance(provider.partition_function("Ce", 1, 1e4), float)

    def test_case_insensitive_species(self, provider):
        a = provider.partition_function("ce", 1, 1e4)
        b = provider.partition_function("CE", 1, 1e4)
        assert a == b
        assert provider.has_species("cE", 1)

    def test_unknown_species_raises_keyerror(self, provider):
        with pytest.raises(KeyError):
            provider.partition_function("Al", 2, 1e4)

    def test_uncovered_temperature_raises_valueerror(self, provider):
        with pytest.raises(ValueError, match="undefined at"):
            provider.partition_function("Ce", 1, 50000.0)

    def test_ion_stage_bounds(self, provider):
        with pytest.raises(ValueError, match="ion_stage"):
            provider.partition_function("Ce", 4, 1e4)

    def test_with_table_returns_new_provider(self, provider):
        grid = np.linspace(1000.0, 20000.0, 50)
        extra = PartitionFunctionTable.from_levels([1.0], [0.0], grid)
        newer = provider.with_table("Al", 2, extra)
        assert newer.has_species("Al", 2)
        assert not provider.has_species("Al", 2)  # original untouched
