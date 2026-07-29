import tempfile
from pathlib import Path

import numpy as np
import pytest

from andes import bma, data, nulls, ranked
from tests.reference import reference_bma, reference_ranked_es


def _embedding(seed=91):
    rng = np.random.default_rng(seed)
    return data.EmbeddingSpace.from_arrays(
        rng.normal(size=(12, 5)).astype(np.float32),
        [f"g{i}" for i in range(12)],
    ).vectors


def _embedding_space(seed=91):
    vectors = _embedding(seed)
    return data.EmbeddingSpace.from_arrays(
        vectors,
        [f"g{i}" for i in range(vectors.shape[0])],
        normalize=False,
    )


def _partial_fisher_yates(size, take, rng):
    permutation = np.arange(size, dtype=np.int32)
    sample = np.empty(take, dtype=np.int32)
    swaps = np.empty(take, dtype=np.int64)
    for position in range(take):
        target = int(rng.integers(position, size))
        swaps[position] = target
        permutation[position], permutation[target] = (
            permutation[target],
            permutation[position],
        )
        sample[position] = permutation[position]
    for position in range(take - 1, -1, -1):
        target = int(swaps[position])
        permutation[position], permutation[target] = (
            permutation[target],
            permutation[position],
        )
    return sample


def test_prefix_bma_null_matches_explicit_monte_carlo_reference():
    embedding = _embedding()
    similarity = embedding @ embedding.T
    left_population = np.arange(0, 10, dtype=np.int32)
    right_population = np.arange(2, 12, dtype=np.int32)
    pairs = {(2, 3), (4, 2)}
    iterations = 9
    seed = 37

    observed = bma.build_prefix_null(
        embedding,
        left_population,
        pairs,
        iterations=iterations,
        seed=seed,
        population2=right_population,
    )

    rng = np.random.default_rng(seed)
    samples = {pair: [] for pair in pairs}
    max_left = max(left for left, _ in pairs)
    max_right = max(right for _, right in pairs)
    for _ in range(iterations):
        left = left_population[rng.permutation(len(left_population))[:max_left]]
        right = right_population[rng.permutation(len(right_population))[:max_right]]
        for pair in pairs:
            left_size, right_size = pair
            samples[pair].append(
                reference_bma(
                    similarity,
                    left[:left_size],
                    right[:right_size],
                )
            )

    for pair, values in samples.items():
        expected = (float(np.mean(values)), float(np.std(values, ddof=1)))
        np.testing.assert_allclose(
            observed[pair],
            expected,
            rtol=2e-6,
            atol=2e-6,
        )


def test_null_resolver_rejects_sizes_that_overflow_int32():
    embedding = _embedding_space()
    population = np.arange(len(embedding.genes), dtype=np.int32)
    wrapped_one = np.asarray([2**32 + 1], dtype=np.uint64)

    with pytest.raises(ValueError, match="supported int32"):
        nulls.inspect_bma_null(
            embedding,
            population,
            population,
            wrapped_one,
            [1],
        )


def test_prefix_ranked_null_matches_explicit_monte_carlo_reference():
    embedding = _embedding()
    similarity = embedding @ embedding.T
    population = np.arange(1, 11, dtype=np.int32)
    ranked_indices = np.array([11, 0, 8, 2, 6, 4, 9], dtype=np.int32)
    ranked_embedding = ranked.compute_ranked_emb(embedding, ranked_indices)
    sizes = [2, 4]
    iterations = 8
    seed = 43

    observed = ranked.build_ranked_null(
        embedding,
        population,
        sizes,
        ranked_embedding,
        iterations=iterations,
        seed=seed,
    )

    samples = {size: [] for size in sizes}
    for iteration in range(iterations):
        rng = np.random.default_rng(np.random.SeedSequence([seed, iteration]))
        local = _partial_fisher_yates(
            len(population),
            max(sizes),
            rng,
        )
        selected = population[local]
        for size in sizes:
            samples[size].append(
                reference_ranked_es(
                    similarity,
                    selected[:size],
                    ranked_indices,
                )
            )

    for size, values in samples.items():
        expected = (float(np.mean(values)), float(np.std(values, ddof=1)))
        np.testing.assert_allclose(
            observed[size],
            expected,
            rtol=2e-5,
            atol=2e-5,
        )


def test_ranked_null_runtime_is_cpu_iteration_and_memory_bounded():
    dimensions = {
        "iterations": 10,
        "max_size": 30,
        "ranked_length": 700,
        "embedding_dimensions": 16,
        "population_size": 900,
        "n_sizes": 12,
        "cpu_count": 32,
    }
    roomy = ranked.plan_ranked_null_runtime(
        requested_workers=0,
        total_workspace_bytes=1_000_000_000,
        **dimensions,
    )
    assert roomy.workers == 8

    minimum = roomy.minimum_workspace_bytes_per_worker
    constrained = ranked.plan_ranked_null_runtime(
        requested_workers=0,
        total_workspace_bytes=3 * minimum,
        **dimensions,
    )
    assert constrained.workers == 3
    assert constrained.workspace_bytes_per_worker == minimum
    assert (
        constrained.workers * constrained.workspace_bytes_per_worker
        <= constrained.total_workspace_bytes
    )

    with np.testing.assert_raises_regex(
        ValueError,
        "need at least",
    ):
        ranked.plan_ranked_null_runtime(
            requested_workers=0,
            total_workspace_bytes=minimum - 1,
            **dimensions,
        )
    with np.testing.assert_raises_regex(
        ValueError,
        "workers need at least",
    ):
        ranked.plan_ranked_null_runtime(
            requested_workers=4,
            total_workspace_bytes=3 * minimum,
            **dimensions,
        )

    explicit = ranked.plan_ranked_null_runtime(
        requested_workers=4,
        total_workspace_bytes=4 * minimum,
        **dimensions,
    )
    assert explicit.workers == 4
    assert (
        explicit.workers * explicit.workspace_bytes_per_worker
        <= explicit.total_workspace_bytes
    )

    with np.testing.assert_raises_regex(ValueError, "max_size"):
        ranked.plan_ranked_null_runtime(
            requested_workers=0,
            total_workspace_bytes=1_000_000,
            **(dimensions | {"max_size": 0}),
        )


def test_precomputed_ranked_null_matches_direct_gemm_statistics():
    embedding = _embedding(109)
    population = np.arange(1, 11, dtype=np.int32)
    ranking = np.array([11, 0, 8, 2, 6, 4, 9], dtype=np.int32)
    ranked_embedding = ranked.compute_ranked_emb(embedding, ranking)
    direct = ranked.build_ranked_null(
        embedding,
        population,
        [2, 4],
        ranked_embedding,
        iterations=9,
        seed=71,
        worker_workspace_bytes=1_000_000,
    )
    precomputed = ranked.build_ranked_null_parallel(
        embedding,
        population,
        [2, 4],
        ranked_embedding,
        iterations=9,
        seed=71,
        workers=2,
        worker_workspace_bytes=1_000_000,
        precompute_similarities=True,
    )

    for size in (2, 4):
        np.testing.assert_allclose(
            precomputed[size],
            direct[size],
            rtol=2e-6,
            atol=2e-6,
        )


def test_ranked_null_builders_reject_less_than_one_iteration_workspace():
    embedding = _embedding(151)
    population = np.arange(1, 11, dtype=np.int32)
    ranking = np.array([11, 0, 8, 2, 6, 4, 9], dtype=np.int32)
    ranked_embedding = ranked.compute_ranked_emb(embedding, ranking)

    with pytest.raises(ValueError, match="too small for one iteration"):
        ranked.build_ranked_null(
            embedding,
            population,
            [2, 4],
            ranked_embedding,
            iterations=2,
            worker_workspace_bytes=1,
        )

    with pytest.raises(ValueError, match="too small for one iteration"):
        ranked.build_ranked_null_parallel(
            embedding,
            population,
            [2, 4],
            ranked_embedding,
            iterations=2,
            workers=1,
            worker_workspace_bytes=1,
            precompute_similarities=True,
        )


def test_bma_null_resolution_builds_reuses_and_extends_one_typed_artifact():
    vectors = _embedding(211)
    embedding = data.EmbeddingSpace.from_arrays(
        vectors,
        [f"g{i}" for i in range(vectors.shape[0])],
        normalize=False,
    )
    population = np.arange(vectors.shape[0], dtype=np.int32)
    with tempfile.TemporaryDirectory() as root:
        artifact = Path(root) / "bma.null"
        built = nulls.resolve_bma_null(
            embedding,
            population,
            population,
            [2],
            [3],
            path=artifact,
            iterations=5,
            seed=17,
        )
        assert built.built
        assert built.added_entries == 1
        assert isinstance(built.model, nulls.BmaNullModel)
        inspected = nulls.inspect_bma_null(
            embedding,
            population,
            population,
            [2, 4],
            [3],
            path=artifact,
            iterations=5,
            seed=17,
        )
        assert inspected.status == "extend"
        assert inspected.requested_entries == 2
        assert inspected.missing_entries == 1
        original = (
            built.model.means[2, 3],
            built.model.stds[2, 3],
        )

        reused = nulls.resolve_bma_null(
            embedding,
            population,
            population,
            [2],
            [3],
            path=artifact,
            iterations=5,
            seed=17,
            no_build=True,
        )
        assert not reused.built
        np.testing.assert_array_equal(reused.model.means, built.model.means)

        extended = nulls.resolve_bma_null(
            embedding,
            population,
            population,
            [2, 4],
            [3],
            path=artifact,
            iterations=5,
            seed=17,
        )
        assert extended.built
        assert extended.added_entries == 1
        assert extended.model.present[4, 3]
        assert (extended.model.means[2, 3], extended.model.stds[2, 3]) == original
        inspected = nulls.inspect_bma_null(
            embedding,
            population,
            population,
            [2, 4],
            [3],
            path=artifact,
            iterations=5,
            seed=17,
        )
        assert inspected.status == "reuse"
        assert inspected.missing_entries == 0

        with np.testing.assert_raises_regex(ValueError, "scientific identity"):
            nulls.resolve_bma_null(
                embedding,
                population,
                population,
                [2],
                [3],
                path=artifact,
                iterations=5,
                seed=18,
                no_build=True,
            )

        before = {
            path.relative_to(Path(root)): path.read_bytes()
            for path in Path(root).rglob("*")
            if path.is_file()
        }
        with np.testing.assert_raises_regex(ValueError, "--rebuild-cache"):
            nulls.resolve_bma_null(
                embedding,
                population,
                population,
                [2],
                [3],
                path=artifact,
                iterations=5,
                seed=18,
            )
        after = {
            path.relative_to(Path(root)): path.read_bytes()
            for path in Path(root).rglob("*")
            if path.is_file()
        }
        assert after == before

        rebuilt = nulls.resolve_bma_null(
            embedding,
            population,
            population,
            [2],
            [3],
            path=artifact,
            iterations=5,
            seed=18,
            rebuild=True,
        )
        assert rebuilt.built
        assert rebuilt.model.spec.seed == 18


def test_ranked_null_resolution_returns_the_same_typed_model_on_reuse():
    vectors = _embedding(223)
    embedding = data.EmbeddingSpace.from_arrays(
        vectors,
        [f"g{i}" for i in range(vectors.shape[0])],
        normalize=False,
    )
    population = np.arange(vectors.shape[0], dtype=np.int32)
    ranking = np.arange(vectors.shape[0] - 1, -1, -1, dtype=np.int32)
    ranked_embedding = ranked.compute_ranked_emb(vectors, ranking)
    plan = ranked.plan_ranked_null_runtime(
        requested_workers=1,
        total_workspace_bytes=10_000_000,
        iterations=4,
        max_size=3,
        ranked_length=len(ranking),
        embedding_dimensions=vectors.shape[1],
        population_size=len(population),
        n_sizes=2,
    )
    with tempfile.TemporaryDirectory() as root:
        artifact = Path(root) / "ranked.null"
        built = nulls.resolve_ranked_null(
            embedding,
            population,
            [2, 3],
            ranked_embedding,
            runtime_plan=plan,
            path=artifact,
            iterations=4,
            seed=29,
        )
        reused = nulls.resolve_ranked_null(
            embedding,
            population,
            [2, 3],
            ranked_embedding,
            runtime_plan=plan,
            path=artifact,
            iterations=4,
            seed=29,
            no_build=True,
        )

        assert built.built
        assert not reused.built
        assert isinstance(reused.model, nulls.RankedNullModel)
        assert reused.model.spec == built.model.spec
        np.testing.assert_array_equal(reused.model.means, built.model.means)
        np.testing.assert_array_equal(reused.model.stds, built.model.stds)
        inspected = nulls.inspect_ranked_null(
            embedding,
            population,
            [2, 3],
            ranked_embedding,
            path=artifact,
            iterations=4,
            seed=29,
        )
        assert inspected.status == "reuse"
        assert inspected.missing_entries == 0

        before = {
            path.relative_to(Path(root)): path.read_bytes()
            for path in Path(root).rglob("*")
            if path.is_file()
        }
        with pytest.raises(ValueError, match="--rebuild-cache"):
            nulls.resolve_ranked_null(
                embedding,
                population,
                [2, 3],
                ranked_embedding,
                runtime_plan=plan,
                path=artifact,
                iterations=4,
                seed=30,
            )
        after = {
            path.relative_to(Path(root)): path.read_bytes()
            for path in Path(root).rglob("*")
            if path.is_file()
        }
        assert after == before


def test_required_bma_null_reads_do_not_mutate_cache_root():
    vectors = _embedding(227)
    embedding = data.EmbeddingSpace.from_arrays(
        vectors,
        [f"g{i}" for i in range(vectors.shape[0])],
        normalize=False,
    )
    population = np.arange(vectors.shape[0], dtype=np.int32)
    with tempfile.TemporaryDirectory() as root_name:
        root = Path(root_name)
        artifact = root / "bma.null"
        built = nulls.resolve_bma_null(
            embedding,
            population,
            population,
            [2],
            [3],
            path=artifact,
            iterations=3,
            seed=41,
        )
        lock_path = root / ".bma.null.lock"
        before = {
            path.relative_to(root): (path.stat().st_mtime_ns, path.read_bytes())
            for path in root.rglob("*")
            if path.is_file()
        }
        root.chmod(0o555)
        try:
            loaded = nulls.resolve_bma_null(
                embedding,
                population,
                population,
                [2],
                [3],
                path=artifact,
                iterations=3,
                seed=41,
                no_build=True,
            )
            with pytest.raises(FileNotFoundError, match="does not exist"):
                nulls.resolve_bma_null(
                    embedding,
                    population,
                    population,
                    [2],
                    [3],
                    path=root / "missing.null",
                    iterations=3,
                    seed=41,
                    no_build=True,
                )
        finally:
            root.chmod(0o755)

        assert loaded.model.spec == built.model.spec
        after = {
            path.relative_to(root): (path.stat().st_mtime_ns, path.read_bytes())
            for path in root.rglob("*")
            if path.is_file()
        }
        assert after == before

        lock_path.unlink()
        with pytest.raises(FileNotFoundError, match="republish or adopt"):
            nulls.resolve_bma_null(
                embedding,
                population,
                population,
                [2],
                [3],
                path=artifact,
                iterations=3,
                seed=41,
                no_build=True,
            )
        assert not lock_path.exists()
