"""Tiny scalar/reference definitions used only for numerical equivalence."""

import numpy as np

_RANKED_TIE_RTOL = 8.0 * np.finfo(np.float32).eps
_RANKED_TIE_ATOL = 8.0 * np.finfo(np.float32).eps


def reference_bma(similarity, left_indices, right_indices):
    block = np.asarray(similarity)[
        np.ix_(
            np.asarray(left_indices, dtype=np.int32),
            np.asarray(right_indices, dtype=np.int32),
        )
    ]
    return float(
        (block.max(axis=1).sum() + block.max(axis=0).sum())
        / (block.shape[0] + block.shape[1])
    )


def reference_ranked_es(similarity, gene_set_indices, ranked_indices):
    block = np.asarray(similarity)[
        np.ix_(
            np.asarray(gene_set_indices, dtype=np.int32),
            np.asarray(ranked_indices, dtype=np.int32),
        )
    ]
    best = block.max(axis=0)
    mean = best.sum(dtype=np.float64) / best.size
    running = 0.0
    maximum = 0.0
    selected = 0.0
    for value in best:
        running += float(value) - mean
        candidate = abs(running)
        scale = max(1.0, candidate, maximum)
        tolerance = _RANKED_TIE_ATOL + _RANKED_TIE_RTOL * scale
        if (maximum == 0.0 and candidate > 0.0) or candidate > maximum + tolerance:
            maximum = candidate
            selected = running
    return selected
