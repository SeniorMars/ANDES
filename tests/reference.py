"""Tiny scalar/reference definitions used only for numerical equivalence."""

import numpy as np


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
    running = np.cumsum(best - best.mean())
    return float(running[int(np.abs(running).argmax())])
