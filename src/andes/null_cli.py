"""Build, list, and verify content-addressed BMA and ranked null artifacts."""

import argparse
import glob
import time

from . import data as ld
from . import index as index_api
from .nulls import (
    load_null_model,
    null_cache_dir,
    resolve_bma_null,
    resolve_cache_root,
    resolve_null_seed,
    resolve_ranked_null,
)
from .ranked import (
    compute_ranked_emb,
    plan_ranked_null_runtime,
)
from .runtime import blas_runtime_info, format_blas_runtime, query_blas_context


def load_union_database(gmt_paths, embedding, min_size, max_size):
    """Return one canonical database spanning several GMT files."""
    return ld.load_gene_set_databases(
        gmt_paths,
        embedding,
        min_size=min_size,
        max_size=max_size,
        sort_terms=True,
    )


def parse_size_spec(value):
    """Parse positive sizes such as ``1:300,350`` into sorted unique values."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("query-size specification must not be empty")
    sizes = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            raise ValueError(f"invalid query-size specification {value!r}")
        if ":" not in token:
            try:
                size = int(token)
            except ValueError as exc:
                raise ValueError(f"invalid query size {token!r}") from exc
            if size < 1:
                raise ValueError("query sizes must be positive")
            sizes.add(size)
            continue

        bounds = token.split(":")
        if len(bounds) != 2:
            raise ValueError(f"invalid query-size range {token!r}")
        try:
            start, stop = (int(bound) for bound in bounds)
        except ValueError as exc:
            raise ValueError(f"invalid query-size range {token!r}") from exc
        if start < 1 or stop < start:
            raise ValueError(f"invalid query-size range {token!r}")
        sizes.update(range(start, stop + 1))
    return sorted(sizes)


def _matched_gmt_paths(patterns):
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError("no GMT files matched the supplied patterns")
    return paths


def _load_raw_database(args):
    gmt_paths = _matched_gmt_paths(args.gmt)
    embedding = ld.load_embedding_space(args.emb, args.genelist)
    database = load_union_database(gmt_paths, embedding, args.min, args.max)
    return embedding, database, gmt_paths


def _load_ranked_inputs(args):
    if args.index:
        index = index_api.load_andes_index(args.index, mmap=True)
        return (
            index.embedding_space(),
            index.gene_set_database(),
            f"Index: {args.index}",
        )
    embedding, database, gmt_paths = _load_raw_database(args)
    return (
        embedding,
        database,
        f"GMT files ({len(gmt_paths)}):\n"
        + "".join(f"  {path}\n" for path in gmt_paths),
    )


def _load_bma_inputs(args):
    if args.index:
        left = index_api.load_andes_index(args.index, mmap=True)
        right = (
            index_api.load_andes_index(args.index2, mmap=True) if args.index2 else left
        )
        left.assert_compatible_with(right)
        embedding = left.embedding_space()
        left_background = left.background
        right_background = right.background
        column_sizes = sorted({int(size) for size in right.sizes})
        row_sizes = (
            parse_size_spec(args.query_sizes)
            if args.query_sizes
            else sorted({int(size) for size in left.sizes})
        )
        source = (
            f"Index: {args.index}\n"
            + (f"Second index: {args.index2}\n" if args.index2 else "")
            + f"Terms: {len(left.terms)} x {len(right.terms)}\n"
            + "Backgrounds: "
            + f"{len(left_background)} x {len(right_background)} genes"
        )
    else:
        embedding, database, gmt_paths = _load_raw_database(args)
        left_background = database.background
        right_background = database.background
        column_sizes = sorted({int(size) for size in database.sizes})
        row_sizes = (
            parse_size_spec(args.query_sizes) if args.query_sizes else column_sizes
        )
        source = (
            f"GMT files ({len(gmt_paths)}):\n"
            + "".join(f"  {path}\n" for path in gmt_paths)
            + f"Terms: {len(database.terms)}\n"
            + f"Background: {len(left_background)} genes"
        )
    if max(row_sizes) > len(left_background):
        raise ValueError(
            "query sizes cannot exceed the indexed null background "
            f"({len(left_background)} genes)"
        )
    if max(column_sizes) > len(right_background):
        raise ValueError(
            "term sizes cannot exceed the indexed null background "
            f"({len(right_background)} genes)"
        )
    return (
        embedding,
        left_background,
        right_background,
        row_sizes,
        column_sizes,
        source,
    )


def cmd_bma(args):
    print("Building BMA null artifact")

    (
        embedding,
        left_background,
        right_background,
        row_sizes,
        column_sizes,
        source,
    ) = _load_bma_inputs(args)
    print(source)
    print(
        f"\nEmbedding: {embedding.vectors.shape}  "
        f"({embedding.vectors.nbytes / 1e6:.1f} MB)"
    )
    size_pairs = {(m, k) for m in row_sizes for k in column_sizes}
    print(
        f"Query sizes: {len(row_sizes)}, "
        f"term sizes: {len(column_sizes)}, "
        f"{len(size_pairs)} (m,k) pairs"
    )

    args.seed = resolve_null_seed(args.seed)
    cache_dir = null_cache_dir("bma", args.cache_root or None)
    t0 = time.perf_counter()
    with query_blas_context(args.worker_blas_threads):
        print(
            "Null BLAS: "
            f"{args.worker_blas_threads} "
            f"({format_blas_runtime(blas_runtime_info())})"
        )
        resolution = resolve_bma_null(
            embedding,
            left_background,
            right_background,
            row_sizes,
            column_sizes,
            base_dir=cache_dir,
            iterations=args.ite,
            seed=args.seed,
            rebuild=args.rebuild,
            blas_threads=args.worker_blas_threads,
        )
    elapsed = time.perf_counter() - t0
    action = "Built" if resolution.built else "Reused"
    print(f"{action} {len(size_pairs)} entries in {elapsed:.1f}s")
    cache_path = resolution.path
    print(f"Null artifact: {cache_path}")


def cmd_es(args):
    print("Building ranked null artifact")

    embedding, database, source = _load_ranked_inputs(args)
    sizes = sorted({int(size) for size in database.sizes})
    print(source)
    print(f"Terms: {len(database.terms)}, sizes: {len(sizes)}")

    ranked_idx = ld.load_ranked_indices(
        args.ranked,
        embedding.gene_to_index,
    )
    print(f"Ranked list: {len(ranked_idx)} genes")

    ranked_emb = compute_ranked_emb(embedding.vectors, ranked_idx)
    args.seed = resolve_null_seed(args.seed)
    cache_dir = null_cache_dir("ranked", args.cache_root or None)

    t0 = time.perf_counter()
    null_plan = plan_ranked_null_runtime(
        requested_workers=args.workers,
        total_workspace_bytes=int(args.null_memory_mb * 1e6),
        iterations=args.ite,
        max_size=max(sizes),
        ranked_length=len(ranked_idx),
        embedding_dimensions=embedding.vectors.shape[1],
        population_size=len(database.background),
        n_sizes=len(sizes),
    )
    print(
        f"Runtime: {null_plan.workers} workers, {null_plan.strategy}, "
        f"{null_plan.total_workspace_bytes / 1e6:.0f} MB total workspace"
    )
    resolution = resolve_ranked_null(
        embedding,
        database.background,
        sizes,
        ranked_emb,
        runtime_plan=null_plan,
        base_dir=cache_dir,
        iterations=args.ite,
        seed=args.seed,
        rebuild=args.rebuild,
        worker_blas_threads=args.worker_blas_threads,
    )
    elapsed = time.perf_counter() - t0
    action = "Built" if resolution.built else "Reused"
    print(f"{action} in {elapsed:.1f}s")
    cache_path = resolution.path
    print(f"Null artifact: {cache_path}")


def cmd_list(args):
    cache_root = resolve_cache_root(args.cache_root or None)
    print(f"Cache root: {cache_root.resolve()}")
    for kind in ("bma", "ranked"):
        artifacts = sorted((cache_root / kind).glob("*.null"))
        print(f"\n[{kind}] {len(artifacts)} cache(s)")
        for path in artifacts:
            size_kb = None
            try:
                size_kb = (
                    sum(
                        child.stat().st_size
                        for child in path.iterdir()
                        if child.is_file()
                    )
                    / 1024
                )
                model = load_null_model(path)
            except (OSError, TypeError, ValueError) as exc:
                size = "" if size_kb is None else f"  ({size_kb:.1f} KB)"
                print(f"  {path.name}{size}  INVALID: {exc}")
                continue
            print(f"  {path.name}  ({size_kb:.1f} KB)")
            print(f"    iterations/seed : {model.spec.iterations}/{model.spec.seed}")
            print(f"    stored entries  : {int(model.present.sum())}")


def cmd_verify(args):
    """Report which (m,k) or m sizes would be missing for a given query setup."""
    kind = args.kind
    if kind == "bma":
        (
            embedding,
            left_background,
            right_background,
            row_sizes,
            column_sizes,
            _,
        ) = _load_bma_inputs(args)
        size_pairs = {(m, k) for m in row_sizes for k in column_sizes}
        resolution = resolve_bma_null(
            embedding,
            left_background,
            right_background,
            row_sizes,
            column_sizes,
            base_dir=null_cache_dir("bma", args.cache_root or None),
            iterations=args.ite,
            seed=resolve_null_seed(args.seed),
            no_build=True,
        )
        model = resolution.model
        print(
            f"Verified {len(size_pairs)} required (m,k) pairs in "
            f"{int(model.present.sum())} stored entries"
        )
        return

    if not args.ranked:
        raise ValueError("--ranked is required when verifying an ES null")
    embedding, database, _ = _load_ranked_inputs(args)
    sizes = sorted({int(size) for size in database.sizes})
    ranked_idx = ld.load_ranked_indices(args.ranked, embedding.gene_to_index)
    ranked_emb = compute_ranked_emb(embedding.vectors, ranked_idx)
    runtime_plan = plan_ranked_null_runtime(
        requested_workers=1,
        total_workspace_bytes=128_000_000,
        iterations=args.ite,
        max_size=max(sizes),
        ranked_length=len(ranked_idx),
        embedding_dimensions=embedding.vectors.shape[1],
        population_size=len(database.background),
        n_sizes=len(sizes),
    )
    resolution = resolve_ranked_null(
        embedding,
        database.background,
        sizes,
        ranked_emb,
        runtime_plan=runtime_plan,
        base_dir=null_cache_dir("ranked", args.cache_root or None),
        iterations=args.ite,
        seed=resolve_null_seed(args.seed),
        no_build=True,
    )
    print(
        f"Verified {len(sizes)} required sizes in "
        f"{int(resolution.model.present.sum())} stored entries"
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="andes null",
        description="Build and inspect reusable ANDES null artifacts",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_cache_root(s):
        s.add_argument(
            "--cache-root",
            default="",
            help=(
                "content-addressed artifact root; defaults to ANDES_CACHE_ROOT "
                "or cache/"
            ),
        )

    def add_identity(s, *, rebuild):
        s.add_argument(
            "--min",
            type=int,
            default=10,
            help="minimum mapped term size (default: 10)",
        )
        s.add_argument(
            "--max",
            type=int,
            default=300,
            help="maximum mapped term size (default: 300)",
        )
        s.add_argument(
            "--ite",
            type=int,
            default=1000,
            help="Monte Carlo null iterations (default: 1000)",
        )
        s.add_argument(
            "--seed",
            type=int,
            default=12345,
            help="random seed; -1 uses OS entropy",
        )
        add_cache_root(s)
        if rebuild:
            s.add_argument(
                "--rebuild",
                action="store_true",
                help="replace an existing matching null artifact",
            )

    def add_raw_source(s, *, required):
        s.add_argument(
            "--emb",
            required=required,
            help="embedding CSV or NPY",
        )
        s.add_argument(
            "--genelist",
            required=required,
            help="embedding gene list",
        )
        s.add_argument(
            "--gmt",
            required=required,
            nargs="+",
            help="one or more GMT files or globs",
        )

    def add_bma_source(s):
        s.add_argument(
            "--index",
            default="",
            help="persistent index for query-size nulls",
        )
        s.add_argument(
            "--index2",
            default="",
            help="compatible column index for index-to-index nulls",
        )
        add_raw_source(s, required=False)
        s.add_argument(
            "--query-sizes",
            default="",
            help=(
                "positive sizes or inclusive ranges, for example 1:300,350; "
                "required for single-index queries"
            ),
        )

    def add_parallel(s):
        s.add_argument(
            "--workers",
            type=int,
            default=0,
            help="worker count; 0 selects a memory-aware count (default: 0)",
        )
        s.add_argument(
            "--worker-blas-threads",
            type=int,
            default=1,
            help="BLAS threads per null worker (default: 1)",
        )
        s.add_argument(
            "--null-memory-mb",
            type=float,
            default=128,
            help="total workspace across ranked-null workers in MB (default: 128)",
        )

    sb = sub.add_parser("bma", help="build a BMA null artifact")
    add_bma_source(sb)
    add_identity(sb, rebuild=True)
    sb.add_argument(
        "--worker-blas-threads",
        type=int,
        default=1,
        help="BLAS threads for BMA null construction (default: 1)",
    )
    sb.set_defaults(func=cmd_bma)

    se = sub.add_parser("es", help="build a ranked null artifact")
    se.add_argument(
        "--index",
        default="",
        help="persistent index for ranked null construction",
    )
    add_raw_source(se, required=False)
    add_identity(se, rebuild=True)
    add_parallel(se)
    se.add_argument("--ranked", required=True, help="ranked gene list")
    se.set_defaults(func=cmd_es)

    sl = sub.add_parser("list", help="list null artifacts")
    add_cache_root(sl)
    sl.set_defaults(func=cmd_list)

    sv = sub.add_parser("verify", help="check null artifact coverage")
    sv.add_argument("kind", choices=["bma", "es"], help="null method")
    add_bma_source(sv)
    add_identity(sv, rebuild=False)
    sv.add_argument("--ranked", help="ranked gene list for ES verification")
    sv.set_defaults(func=cmd_verify)

    args = p.parse_args(argv)
    if hasattr(args, "ite") and args.ite < 2:
        p.error("--ite must be at least 2")
    if hasattr(args, "workers") and args.workers < 0:
        p.error("--workers must be non-negative")
    if hasattr(args, "worker_blas_threads") and args.worker_blas_threads < 1:
        p.error("--worker-blas-threads must be positive")
    if hasattr(args, "null_memory_mb") and args.null_memory_mb <= 0:
        p.error("--null-memory-mb must be positive")
    if args.cmd in {"bma", "verify"} and (args.cmd == "bma" or args.kind == "bma"):
        raw_values = (args.emb, args.genelist, args.gmt)
        if args.index2 and not args.index:
            p.error("--index2 requires --index")
        if args.index:
            if any(raw_values):
                p.error("--index cannot be combined with --emb, --genelist, or --gmt")
            if not args.index2 and not args.query_sizes:
                p.error("--query-sizes is required with a single --index")
        else:
            missing = [
                flag
                for flag, value in zip(
                    ("--emb", "--genelist", "--gmt"),
                    raw_values,
                    strict=True,
                )
                if not value
            ]
            if missing:
                p.error("BMA null construction requires " + ", ".join(missing))
        if args.query_sizes:
            try:
                parse_size_spec(args.query_sizes)
            except ValueError as exc:
                p.error(str(exc))
    if args.cmd == "verify" and args.kind == "es" and (args.index2 or args.query_sizes):
        p.error("--index2 and --query-sizes apply only to BMA nulls")
    if args.cmd == "es" or (args.cmd == "verify" and args.kind == "es"):
        raw_values = (args.emb, args.genelist, args.gmt)
        if args.index:
            if any(raw_values):
                p.error("--index cannot be combined with --emb, --genelist, or --gmt")
        else:
            missing = [
                flag
                for flag, value in zip(
                    ("--emb", "--genelist", "--gmt"),
                    raw_values,
                    strict=True,
                )
                if not value
            ]
            if missing:
                p.error("ES null construction requires " + ", ".join(missing))
        if not args.ranked:
            p.error("ES null construction requires --ranked")
    return args


def main(argv=None):
    args = parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
