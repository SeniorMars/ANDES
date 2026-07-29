"""Expression ranking helpers for empirical ranked ANDES workflows.

The ranking statistic is the ordinary least-squares t statistic for a
two-group predictor with an intercept. The vectorized form matches
``sm.OLS(y, add_constant(condition))`` for every gene and computes a phenotype
permutation batch with one ``Y @ C`` matrix multiplication.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.intp]


@dataclass(frozen=True, slots=True)
class ExpressionDataset:
    """Validated gene-by-sample expression data for a binary condition."""

    genes: tuple[str, ...]
    values: FloatArray
    labels: FloatArray

    @classmethod
    def from_dataframe(
        cls,
        data,
        condition,
        *,
        sample_columns=None,
    ):
        labels = encode_two_group_condition(condition)
        if data.ndim != 2 or data.shape[0] == 0:
            raise ValueError("expression data must contain a two-dimensional table")

        if sample_columns is None:
            if data.shape[1] != labels.size:
                raise ValueError(
                    "sample_columns is required unless every expression "
                    "column corresponds to one condition label"
                )
            selected = data
        else:
            sample_columns = tuple(sample_columns)
            if len(sample_columns) != labels.size:
                raise ValueError(
                    "sample_columns length must match the condition length"
                )
            missing = [
                column for column in sample_columns if column not in data.columns
            ]
            if missing:
                raise ValueError(
                    "expression sample columns are missing: "
                    + ", ".join(str(column) for column in missing)
                )
            selected = data.loc[:, sample_columns]

        try:
            values = selected.to_numpy(dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("expression sample columns must be numeric") from exc
        if not np.isfinite(values).all():
            bad_row, bad_column = np.argwhere(~np.isfinite(values))[0]
            raise ValueError(
                "expression data contains a non-finite sample value at "
                f"row {int(bad_row)}, column {int(bad_column)}"
            )
        genes = tuple(data.index.astype(str))
        if any(not gene for gene in genes):
            raise ValueError("expression gene identifiers must be non-empty")
        if len(set(genes)) != len(genes):
            raise ValueError("expression gene identifiers must be unique")
        values = np.ascontiguousarray(values, dtype=np.float64)
        labels = np.ascontiguousarray(labels, dtype=np.float64)
        values.setflags(write=False)
        labels.setflags(write=False)
        return cls(genes=genes, values=values, labels=labels)


def encode_two_group_condition(condition) -> FloatArray:
    """Return a float64 0/1 vector for a finite two-level condition.

    The larger numeric label is coded as one. Historical 0/1 input is unchanged.
    Other numeric two-level labels retain the OLS slope sign from the original
    labels.
    """
    values = np.asarray(condition, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("condition must be one-dimensional")
    if values.size < 4:
        raise ValueError("binary OLS requires at least two samples in each group")
    if not np.isfinite(values).all():
        raise ValueError("condition contains a non-finite value")

    levels = np.unique(values)
    if levels.size != 2:
        raise ValueError(
            f"condition must contain exactly two groups; found {levels.size}"
        )
    encoded = (values == levels[-1]).astype(np.float64)
    n_case = int(encoded.sum())
    n_control = int(encoded.size - n_case)
    if n_case < 2 or n_control < 2:
        raise ValueError("condition must contain at least two samples in each group")
    return encoded


def prepare_expression_data(data, condition, *, sample_columns=None):
    """Extract a validated gene-by-sample float64 matrix from a DataFrame.

    Without ``sample_columns``, every DataFrame column must correspond to one
    condition label. Callers loading tables with metadata columns must identify
    their sample columns explicitly.

    Returns
    -------
    values : ndarray, shape (genes, samples), float64
    genes : ndarray, shape (genes,), str
    encoded_condition : ndarray, shape (samples,), float64
    """
    dataset = ExpressionDataset.from_dataframe(
        data,
        condition,
        sample_columns=sample_columns,
    )
    return (
        dataset.values,
        np.asarray(dataset.genes, dtype=str),
        dataset.labels,
    )


def _binary_ols_from_summaries(Y, C, total, total_ss):
    """Core one-GEMM OLS calculation for validated arrays and fixed Y."""
    n = Y.shape[1]
    n1 = C.sum(axis=0, dtype=np.float64)
    n0 = float(n) - n1
    if np.any(n1 < 2.0) or np.any(n0 < 2.0):
        raise ValueError(
            "every condition column must contain at least two samples in each group"
        )

    sum1 = Y @ C
    mean1 = sum1 / n1[None, :]
    mean0 = (total - sum1) / n0[None, :]
    beta = mean1 - mean0

    between_ss = beta * beta * (n0 * n1 / float(n))[None, :]
    residual_ss = total_ss - between_ss

    # Roundoff can put a theoretically zero residual a few ULPs below zero.
    scale = np.maximum(total_ss, between_ss)
    tolerance = 128.0 * np.finfo(np.float64).eps * np.maximum(scale, 1.0)
    if np.any(residual_ss < -tolerance):
        raise FloatingPointError(
            "negative OLS residual sum-of-squares exceeds roundoff tolerance"
        )
    np.maximum(residual_ss, 0.0, out=residual_ss)

    mse = residual_ss / float(n - 2)
    variance = mse * (1.0 / n0 + 1.0 / n1)[None, :]
    standard_error = np.sqrt(variance)

    statistics = np.empty_like(beta)
    positive_se = standard_error > 0.0
    np.divide(beta, standard_error, out=statistics, where=positive_se)
    zero_se = ~positive_se
    statistics[zero_se & (beta == 0.0)] = 0.0
    nonzero_perfect = zero_se & (beta != 0.0)
    statistics[nonzero_perfect] = np.copysign(np.inf, beta[nonzero_perfect])
    return statistics


def _validate_expression_values(expression_values):
    Y = np.asarray(expression_values, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError("expression_values must have shape (genes, samples)")
    if Y.shape[0] == 0:
        raise ValueError("expression_values contains no genes")
    if Y.shape[1] < 3:
        raise ValueError("OLS with an intercept requires at least three samples")
    if not np.isfinite(Y).all():
        raise ValueError("expression_values contains a non-finite value")
    return np.ascontiguousarray(Y)


def _expression_summaries(Y):
    total = Y.sum(axis=1, dtype=np.float64)[:, None]
    centered = Y - (total / float(Y.shape[1]))
    total_ss = np.einsum("ij,ij->i", centered, centered)[:, None]
    return total, total_ss


def binary_ols_t_statistics(expression_values, conditions):
    """Vectorized OLS slope t statistics for one or many two-group designs.

    Parameters
    ----------
    expression_values : array-like, shape (genes, samples)
        Finite expression values.
    conditions : array-like, shape (samples,) or (samples, permutations)
        Binary 0/1 design columns. Every column must contain at least two
        samples in each group.

    Notes
    -----
    With an intercept and a binary predictor, the slope is the difference of
    group means. Total centered sum-of-squares is invariant to phenotype
    permutations, so all permutation-specific sufficient statistics come from
    one GEMM, ``expression_values @ conditions``.

    A constant gene receives t=0. A zero-residual, nonzero group difference
    receives signed infinity. Other non-finite inputs are rejected.
    """
    Y = _validate_expression_values(expression_values)
    C_input = np.asarray(conditions)
    return_vector = C_input.ndim == 1
    if return_vector:
        C_input = C_input[:, None]
    if C_input.ndim != 2 or C_input.shape[0] != Y.shape[1]:
        raise ValueError(
            "conditions must have shape (samples,) or (samples, permutations)"
        )

    C = np.asarray(C_input, dtype=np.float64)
    if not np.isfinite(C).all():
        raise ValueError("conditions contains a non-finite value")
    if np.any((C != 0.0) & (C != 1.0)):
        raise ValueError("conditions must contain only binary 0/1 values")

    total, total_ss = _expression_summaries(Y)
    statistics = _binary_ols_from_summaries(Y, C, total, total_ss)
    if return_vector:
        return statistics[:, 0]
    return statistics


def stable_rank_orders(t_statistics):
    """Return stable descending gene-row orders for one or many columns."""
    statistics = np.asarray(t_statistics, dtype=np.float64)
    if statistics.ndim == 1:
        return np.argsort(-statistics, kind="stable")
    if statistics.ndim != 2:
        raise ValueError("t_statistics must be one- or two-dimensional")
    return np.argsort(-statistics, axis=0, kind="stable")


def permute_labels(labels, rng: np.random.Generator) -> FloatArray:
    """Return an independently permuted copy using the caller's generator."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    encoded = encode_two_group_condition(labels)
    return np.asarray(rng.permutation(encoded), dtype=np.float64)


def permuted_condition_matrix(
    condition,
    seeds: Sequence[int],
) -> FloatArray:
    """Build deterministic, independently shuffled condition columns."""
    encoded = encode_two_group_condition(condition)
    seeds = list(seeds)
    result = np.empty((encoded.size, len(seeds)), dtype=np.float64)
    for column, seed in enumerate(seeds):
        result[:, column] = permute_labels(
            encoded,
            np.random.default_rng(int(seed)),
        )
    return result


def iter_label_shuffled_rank_orders(
    expression_values,
    condition,
    n_permutations,
    *,
    seed=0,
    batch_size=16,
) -> Iterator[tuple[int, IndexArray]]:
    """Yield stable gene-row orders for phenotype permutations in GEMM batches.

    Each yielded order matrix has shape ``(n_genes, batch_permutations)``.
    Permutation ``i`` uses seed ``seed + i``, matching the former CLI.
    """
    if int(n_permutations) < 1:
        raise ValueError("n_permutations must be >= 1")
    if int(batch_size) < 1:
        raise ValueError("batch_size must be >= 1")

    encoded = encode_two_group_condition(condition)
    values = _validate_expression_values(expression_values)
    if values.ndim != 2 or values.shape[1] != encoded.size:
        raise ValueError("expression_values must have one column per condition sample")
    total, total_ss = _expression_summaries(values)

    for start in range(0, int(n_permutations), int(batch_size)):
        end = min(start + int(batch_size), int(n_permutations))
        seeds = [int(seed) + i for i in range(start, end)]
        condition_batch = permuted_condition_matrix(
            encoded,
            seeds,
        )
        statistics = _binary_ols_from_summaries(
            values, condition_batch, total, total_ss
        )
        yield start, stable_rank_orders(statistics)
