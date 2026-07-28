"""
validate_drug_repurposing.py

End-to-end pipeline to reproduce the drug-repurposing figure from the original
ANDES paper using the optimized code, and to validate that new ANDES gives the
same biological conclusions as old ANDES.

What it does
------------
1. For each (GSE dataset, disease) pair, score every drug in DrugBank by
   running new ANDES in ranked-GSEA mode against the disease's ranked
   expression list.
2. Optionally re-run old ANDES on the same inputs and report per-dataset
   AUPRC differences.
3. Load existing GSEA and GSPA NES scores from the paper's results
   directory.
4. Compute hypergeometric scores (drug targets vs top differentially
   expressed disease genes).
5. For each method and each disease, compute AUPRC against the
   drug-indication ground truth from DrugBank.
6. Save panel CSVs and render the three-panel figure.

Validation output
-----------------
    auprc_new_vs_old.csv       per-GSE AUPRC for old and new ANDES
    panel_A.csv                hyper, ANDES per-GSE AUPRC
    panel_B.csv                GSEA, GSPA, ANDES per-GSE AUPRC
    panel_C.csv                corrected GSEA, GSPA, ANDES per-GSE AUPRC
    figure_drug_repurposing.pdf  the three-panel figure

Expected directory layout (matches helper.py paths from the paper repo)
-----------------------------------------------------------------------
    data/
        embedding/
            node2vec_consensus.csv
            consensus_node.txt
        gene_sets/
            hsa_experimental_eval_BP_propagated.gmt   (background)
            drugbank_targets.gmt                      (drugs -> target genes)
        expression/
            <GSE_id>.txt           expression matrix (label row + data)
            <GSE_id>_rank.txt      pre-ranked gene list, one gene id per row
        ground_truth/
            drug_indications.csv   index = drug id; columns = disease labels
                                   matching gse_to_disease_csv labels;
                                   values 0/1
    results/
        enrichment_analysis/
            GSEA/<GSE_id>_result.csv             with column 'NES'
            GSPA/<GSE_id>_GSPA_results_our_ppi.csv  with column 'NES'

If your layout differs, override paths via CLI flags.

Usage
-----
    # Full pipeline (build cache, score everything, plot):
    python validate_drug_repurposing.py \
        --emb data/embedding/node2vec_consensus.csv \
        --genelist data/embedding/consensus_node.txt \
        --drug-gmt data/gene_sets/drugbank_targets.gmt \
        --bg-gmt data/gene_sets/hsa_experimental_eval_BP_propagated.gmt \
        --gse-list configs/gse_to_disease.csv \
        --rank-dir data/expression \
        --truth data/ground_truth/drug_indications.csv \
        --gsea-dir results/enrichment_analysis/GSEA \
        --gspa-dir results/enrichment_analysis/GSPA \
        --out-dir results/validation \
        --workers 8

    # Validation mode: also re-run old ANDES for direct comparison.
    # Adds ~30 minutes for ~30 GSE datasets. Off by default.
    python validate_drug_repurposing.py ... --run-old-andes

    # Skip the ANDES run and just plot from cached panel CSVs (fast iteration
    # on the figure):
    python validate_drug_repurposing.py ... --plot-only
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Pin BLAS BEFORE numpy import
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import hypergeom
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from andes import bma as func_new
from andes import data as ld
from andes.ranked import (
    RankedNullBuilder,
    compute_es_score,
    compute_ranked_emb,
)
from experiments.legacy import set_analysis_func as func_old


# ════════════════════════════════════════════════════════════════════════════
# Plotting (style matches the published figure)
# ════════════════════════════════════════════════════════════════════════════

PLT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.85",
}

METHOD_STYLE = {
    "hypergeometric":   dict(marker="D", color="#9b9b9b", filled=True),
    "ANDES":            dict(marker="o", color="#a83232", filled=True),
    "GSEA":             dict(marker="<", color="#2f7d3a", filled=True),
    "GSPA":             dict(marker=">", color="#1f4e79", filled=True),
    "corrected GSEA":   dict(marker="<", color="#2f7d3a", filled=False),
    "corrected GSPA":   dict(marker=">", color="#1f4e79", filled=False),
    "corrected ANDES":  dict(marker="o", color="#a83232", filled=False),
}


def _plot_panel(ax, df, sort_by, title):
    df = df.sort_values(sort_by, ascending=True).copy()  # ascending so top = highest
    y = np.arange(len(df))
    for col in df.columns:
        if col not in METHOD_STYLE:
            continue
        st = METHOD_STYLE[col]
        ax.plot(
            df[col].values, y,
            marker=st["marker"], linestyle="None",
            markersize=6,
            markeredgecolor=st["color"],
            markerfacecolor=st["color"] if st["filled"] else "white",
            markeredgewidth=0.0 if st["filled"] else 1.0,
            label=col,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=8)
    ax.set_xlabel("AUPRC")
    ax.set_xlim(-0.02, 1.0)
    ax.set_title(title, loc="left", x=-0.35, y=1.01)
    ax.grid(axis="x", alpha=0.15, linewidth=0.5)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right")


def make_figure(panel_a, panel_b, panel_c, out_path):
    plt.rcParams.update(PLT_STYLE)
    height = max(0.16 * max(len(panel_a), len(panel_b), len(panel_c)) + 1.2, 6)
    fig, axes = plt.subplots(1, 3, figsize=(15, height))
    _plot_panel(axes[0], panel_a, "ANDES",          "A")
    _plot_panel(axes[1], panel_b, "ANDES",          "B")
    _plot_panel(axes[2], panel_c, "corrected ANDES", "C")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# DrugBank x OMIM z-score validation
# ════════════════════════════════════════════════════════════════════════════

ATC_COLORS = {
    "A": "#8dd3c7",
    "B": "#ffffb3",
    "C": "#bebada",
    "D": "#fb8072",
    "G": "#80b1d3",
    "H": "#fdb462",
    "J": "#b3de69",
    "L": "#fccde5",
    "M": "#d9d9d9",
    "N": "#bc80bd",
    "P": "#ccebc5",
    "R": "#ffed6f",
    "S": "#1f78b4",
    "V": "#33a02c",
}


def _load_term_indices_for_terms(gmt_path, terms, node2index, lower, upper):
    gmt = ld.load_gmt(gmt_path)
    raw = {}
    missing_terms = []
    filtered_terms = []
    for term in terms:
        genes = gmt.get(term)
        if genes is None:
            missing_terms.append(term)
            continue
        idx = sorted({node2index[g] for g in genes if node2index[g] != -1})
        if len(idx) < lower or len(idx) > upper:
            filtered_terms.append(term)
            continue
        raw[term] = set(idx)
    return func_new.preconvert_indices_to_arrays(raw), missing_terms, filtered_terms


def _population_from_gmt(gmt_path, node2index):
    gmt = ld.load_gmt(gmt_path)
    genes = sorted({node2index[g] for values in gmt.values() for g in values if node2index[g] != -1})
    return np.asarray(genes, dtype=np.int32)


def _build_or_load_bma_cache(args, E_unit, pop1, pop2, size_pairs, out_dir):
    cache_path = (
        Path(args.cache)
        if args.cache
        else out_dir / "drug_disease_bma_cache.null"
    )
    seed = func_new.BmaNullBuilder.resolve_seed(args.seed)
    expected = func_new.BmaNullBuilder.build_metadata(E_unit, pop1, pop2, args.ite, seed)

    cache = func_new.BmaNullBuilder()
    if cache_path.exists() and not args.rebuild_cache:
        cache.load_artifact(cache_path)
        ok, reason = cache.metadata_matches(expected)
        missing = [pair for pair in size_pairs if pair not in cache.cache]
        if ok and not missing:
            print(f"Loaded {len(cache.cache)} BMA cache entries from {cache_path}")
            return cache
        print(f"Rebuilding BMA cache: {reason or f'{len(missing)} missing size pairs'}")

    if args.workers <= 1:
        cache.precompute(
            E_unit,
            pop1,
            size_pairs,
            ite=args.ite,
            seed=seed,
            population_idx2=pop2,
            verbose=True,
        )
    else:
        try:
            cache.precompute_parallel(
                E_unit,
                pop1,
                size_pairs,
                ite=args.ite,
                seed=seed,
                n_workers=args.workers,
                chunk_size=args.chunk_size,
                population_idx2=pop2,
                verbose=True,
            )
        except PermissionError as exc:
            print(f"Parallel cache build unavailable ({exc}); falling back to --workers 1.")
            cache.precompute(
                E_unit,
                pop1,
                size_pairs,
                ite=args.ite,
                seed=seed,
                population_idx2=pop2,
                verbose=True,
            )
    cache.save_artifact(cache_path, overwrite=cache_path.exists())
    print(f"Saved {len(cache.cache)} BMA cache entries to {cache_path}")
    return cache


def _comparison_metrics(reference, candidate):
    old = reference.to_numpy(dtype=np.float64).ravel()
    new = candidate.to_numpy(dtype=np.float64).ravel()
    mask = np.isfinite(old) & np.isfinite(new)
    old = old[mask]
    new = new[mask]
    diff = new - old
    metrics = {
        "n_values": int(mask.sum()),
        "pearson": float(np.corrcoef(old, new)[0, 1]) if old.size > 1 else np.nan,
        "spearman": float(pd.Series(old).corr(pd.Series(new), method="spearman")) if old.size > 1 else np.nan,
        "mae": float(np.mean(np.abs(diff))) if diff.size else np.nan,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else np.nan,
        "median_abs_diff": float(np.median(np.abs(diff))) if diff.size else np.nan,
        "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else np.nan,
        "old_mean": float(np.mean(old)) if old.size else np.nan,
        "new_mean": float(np.mean(new)) if new.size else np.nan,
        "old_std": float(np.std(old)) if old.size else np.nan,
        "new_std": float(np.std(new)) if new.size else np.nan,
    }
    return metrics, old, new


def _plot_zscore_scatter(reference, candidate, out_dir):
    metrics, old, new = _comparison_metrics(reference, candidate)
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=180)
    ax.scatter(old, new, s=4, alpha=0.18, linewidths=0, color="#26547c")
    lo = float(min(old.min(), new.min()))
    hi = float(max(old.max(), new.max()))
    ax.plot([lo, hi], [lo, hi], color="#b23a48", linewidth=1.0)
    ax.set_xlabel("old/reference z-score")
    ax.set_ylabel("new optimized z-score")
    ax.set_title("DrugBank x OMIM z-score agreement")
    ax.text(
        0.04,
        0.96,
        f"r={metrics['pearson']:.3f}\nrho={metrics['spearman']:.3f}\nMAE={metrics['mae']:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.9),
    )
    fig.tight_layout()
    fig.savefig(out_dir / "old_vs_new_zscores.png", bbox_inches="tight")
    fig.savefig(out_dir / "old_vs_new_zscores.pdf", bbox_inches="tight")
    plt.close(fig)
    return metrics


def _primary_atc_code(drug, drug_to_atc):
    codes = sorted(drug_to_atc.get(drug, []))
    return codes[0][0] if codes else ""


def plot_drug_disease_pca(zscores, args, out_dir):
    if not args.drug_atc or not args.atc_names:
        return None
    with open(args.drug_atc, "rb") as fh:
        drug_to_atc = pickle.load(fh)
    with open(args.atc_names, "rb") as fh:
        atc_names = pickle.load(fh)

    values = zscores.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    if values.shape[0] < 3 or values.shape[1] < 3:
        return None

    coords = PCA(n_components=3).fit_transform(values)
    labels = pd.Series(
        [_primary_atc_code(drug, drug_to_atc) for drug in zscores.index],
        index=zscores.index,
    )
    counts = labels.value_counts()
    selected = [code for code in ATC_COLORS if counts.get(code, 0) >= args.min_atc_count]
    if not selected:
        selected = [code for code in counts.index if code][:14]

    fig = plt.figure(figsize=(8, 8), dpi=200)
    ax = fig.add_subplot(projection="3d")
    for code in selected:
        idx = np.flatnonzero(labels.to_numpy() == code)
        name = atc_names.get(code, code).lower()
        ax.scatter(
            coords[idx, 1],
            coords[idx, 2],
            coords[idx, 0],
            s=18,
            alpha=0.9,
            color=ATC_COLORS.get(code, None),
            label=name,
        )
    ax.set_xlabel("PC2")
    ax.set_ylabel("PC3")
    ax.set_zlabel("PC1")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "drug_disease_pca_atc.png", bbox_inches="tight")
    fig.savefig(out_dir / "drug_disease_pca_atc.pdf", bbox_inches="tight")
    plt.close(fig)
    return selected


def run_drug_disease_zscore_validation(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print("Loading reference z-score matrix...")
    reference = pd.read_csv(args.reference_zscores, index_col=0)
    if args.limit_drugs:
        reference = reference.iloc[: args.limit_drugs, :]
    if args.limit_diseases:
        reference = reference.iloc[:, : args.limit_diseases]
    print(f"  reference: {reference.shape[0]} drugs x {reference.shape[1]} diseases")

    print("Loading embedding...")
    E_unit, gene_list = load_embedding(args.emb, args.genelist)
    g_node2index = defaultdict(lambda: -1, {g: i for i, g in enumerate(gene_list)})

    drug_terms_all = list(reference.index.astype(str))
    disease_terms_all = list(reference.columns.astype(str))
    drug_indices, missing_drugs, filtered_drugs = _load_term_indices_for_terms(
        args.drug_gmt, drug_terms_all, g_node2index, args.min_size, args.max_size
    )
    disease_indices, missing_diseases, filtered_diseases = _load_term_indices_for_terms(
        args.disease_gmt, disease_terms_all, g_node2index, args.min_size, args.max_size
    )

    drug_terms = [term for term in drug_terms_all if term in drug_indices]
    disease_terms = [term for term in disease_terms_all if term in disease_indices]
    if not drug_terms or not disease_terms:
        raise SystemExit("No valid drug or disease terms remain after embedding and size filters.")

    reference = reference.loc[drug_terms, disease_terms]
    print(f"Matched terms: {len(drug_terms)} drugs x {len(disease_terms)} diseases")
    if missing_drugs or filtered_drugs or missing_diseases or filtered_diseases:
        print(
            "Dropped terms: "
            f"{len(missing_drugs)} missing drugs, {len(filtered_drugs)} filtered drugs, "
            f"{len(missing_diseases)} missing diseases, {len(filtered_diseases)} filtered diseases"
        )

    pop1 = _population_from_gmt(args.drug_gmt, g_node2index)
    pop2 = _population_from_gmt(args.disease_gmt, g_node2index)
    size_pairs = sorted({
        (len(drug_indices[d]), len(disease_indices[s]))
        for d in drug_terms
        for s in disease_terms
    })
    print(f"Unique size pairs: {len(size_pairs)}")

    func_new.warmup_numba()
    cache = _build_or_load_bma_cache(args, E_unit, pop1, pop2, size_pairs, out_dir)

    print("Precomputing term embedding blocks...")
    blocks1 = func_new.precompute_term_embedding_blocks(
        E_unit, {term: drug_indices[term] for term in drug_terms}
    )
    blocks2 = func_new.precompute_term_embedding_blocks(
        E_unit, {term: disease_indices[term] for term in disease_terms}
    )

    print("Scoring optimized ANDES z-score matrix...")
    zscores, effective_workers, workspace_mb = func_new.score_bma_zscore_matrix_batched(
        drug_terms,
        disease_terms,
        cache,
        blocks1,
        blocks2,
        symmetric=False,
        n_workers=args.query_workers,
        max_workspace_mb=args.query_memory_mb,
        show_progress=True,
    )
    candidate = pd.DataFrame(zscores, index=drug_terms, columns=disease_terms)
    candidate.to_csv(out_dir / "drug_disease_new_zscores.csv")

    metrics = _plot_zscore_scatter(reference, candidate, out_dir)
    with open(out_dir / "drug_disease_zscore_metrics.json", "w") as fh:
        json.dump(
            {
                **metrics,
                "n_drugs": len(drug_terms),
                "n_diseases": len(disease_terms),
                "ite": args.ite,
                "seed": args.seed,
                "cache_entries": len(cache.cache),
                "query_workers_requested": args.query_workers,
                "query_workers_effective": effective_workers,
                "row_workspace_mb": workspace_mb,
            },
            fh,
            indent=2,
        )

    diff = (candidate - reference).stack().rename("new_minus_old").reset_index()
    diff.columns = ["drug", "disease", "new_minus_old"]
    diff["old_zscore"] = reference.stack().to_numpy()
    diff["new_zscore"] = candidate.stack().to_numpy()
    diff.reindex(columns=["drug", "disease", "old_zscore", "new_zscore", "new_minus_old"]).to_csv(
        out_dir / "drug_disease_zscore_differences.csv", index=False
    )

    selected_atc = [] if args.skip_pca else plot_drug_disease_pca(candidate, args, out_dir)
    elapsed = time.perf_counter() - t0
    print("\nDrug-disease z-score validation")
    print(f"  Pearson r      : {metrics['pearson']:.4f}")
    print(f"  Spearman rho   : {metrics['spearman']:.4f}")
    print(f"  MAE            : {metrics['mae']:.4f}")
    print(f"  RMSE           : {metrics['rmse']:.4f}")
    print(f"  Effective query workers: {effective_workers}")
    if selected_atc:
        print(f"  PCA ATC classes: {', '.join(selected_atc)}")
    print(f"  Wrote outputs to {out_dir}")
    print(f"  Total time: {elapsed:.1f}s")


# ════════════════════════════════════════════════════════════════════════════
# Loading
# ════════════════════════════════════════════════════════════════════════════

def load_embedding(emb_path, genelist_path):
    raw = np.loadtxt(emb_path, delimiter=",", dtype=np.float32)
    with open(genelist_path) as fh:
        gene_list = [line.strip() for line in fh]
    if len(gene_list) != raw.shape[0]:
        raise ValueError("embedding rows mismatch gene list length")
    E_unit = np.ascontiguousarray(
        func_new.l2_normalize_rows(raw), dtype=np.float32
    )
    return E_unit, gene_list


def load_ranked(path, g_node2index, gene_set):
    df = pd.read_csv(path, sep="\t", index_col=0, header=None)
    return np.array(
        [g_node2index[str(g)] for g in df.index if str(g) in gene_set],
        dtype=np.int32,
    )


def load_gse_to_disease(csv_path):
    """CSV with columns: gse_id, disease_label.
    disease_label must match a column in the ground-truth CSV."""
    df = pd.read_csv(csv_path)
    expected = {"gse_id", "disease_label"}
    if not expected.issubset(df.columns):
        raise ValueError(f"{csv_path} must have columns {expected}")
    return list(zip(df["gse_id"], df["disease_label"]))


# ════════════════════════════════════════════════════════════════════════════
# Scoring
# ════════════════════════════════════════════════════════════════════════════

def score_andes_new(E_unit, drug_indices_np, ranked_idx, cache):
    """Return dict drug_id -> z-score using new ANDES (cached)."""
    ranked_emb = compute_ranked_emb(E_unit, ranked_idx)
    out = {}
    for drug, idx in drug_indices_np.items():
        m = len(idx)
        if m == 0 or m not in cache:
            out[drug] = np.nan
            continue
        s = compute_es_score(E_unit, idx, ranked_emb)
        out[drug] = cache.get_zscore(s, m)
    return out


def score_andes_old(S, drug_indices_set, ranked_idx_list, ite=1000, seed=12345):
    """Return dict drug_id -> z-score using OLD ANDES via gsea_andes
    on the full similarity matrix."""
    out = {}
    drug_terms = list(drug_indices_set.keys())
    annotated = sorted(set().union(*[set(v) for v in drug_indices_set.values()]))
    f = lambda term: func_old.gsea_andes(
        term, ranked_list=ranked_idx_list, matrix=S,
        term2indices=drug_indices_set,
        annotated_indices=annotated,
        ite=ite,
    )
    for drug in tqdm(drug_terms, desc="old ANDES"):
        try:
            _, z = f(drug)
        except Exception:
            z = np.nan
        out[drug] = z
    return out


def score_hypergeometric(drug_indices_set, top_de_set, n_genes_total):
    """Hypergeometric one-sided p-value for drug targets enriched in top DE
    genes. Score is -log10(p), so larger = more enriched."""
    out = {}
    K = len(top_de_set)
    for drug, targets in drug_indices_set.items():
        targets = set(targets)
        n = len(targets)
        if n == 0:
            out[drug] = 0.0
            continue
        k = len(targets & top_de_set)
        # P(X >= k) where X ~ Hypergeom(N=n_genes_total, K=top_de, n=n_targets)
        pval = hypergeom.sf(k - 1, n_genes_total, K, n)
        # Avoid log(0)
        out[drug] = -np.log10(max(pval, 1e-300))
    return out


def load_existing_nes(path, kind):
    """Load existing GSEA or GSPA result CSVs.
    Returns: dict term -> NES."""
    if not Path(path).exists():
        return None
    df = pd.read_csv(path, index_col=0)
    nes_col = next((c for c in df.columns if c.upper() == "NES"), None)
    if nes_col is None:
        return None
    return df[nes_col].to_dict()


# ════════════════════════════════════════════════════════════════════════════
# Pipeline
# ════════════════════════════════════════════════════════════════════════════

def run_pipeline(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load embedding ─────────────────────────────────────────────────────
    print("Loading embedding...")
    E_unit, gene_list = load_embedding(args.emb, args.genelist)
    gene_set = set(gene_list)
    g_node2index = defaultdict(lambda: -1, {g: i for i, g in enumerate(gene_list)})
    print(f"  {len(gene_list)} genes  d={E_unit.shape[1]}  ({E_unit.nbytes/1e6:.1f} MB)")

    # ── Load drug GMT and background ───────────────────────────────────────
    print("Loading drug GMT and background...")
    drug_gmt = ld.load_gmt(args.drug_gmt)
    drug_indices_set = ld.term2indexes(
        drug_gmt, g_node2index, upper=args.max_size, lower=args.min_size
    )
    drug_indices_np = func_new.preconvert_indices_to_arrays(drug_indices_set)
    drug_terms = sorted(drug_indices_np.keys())
    print(f"  drugs (after size filter): {len(drug_terms)}")

    bg_gmt = ld.load_gmt(args.bg_gmt)
    all_bg_genes = set().union(*bg_gmt.values()) & gene_set
    pop = np.array(sorted(g_node2index[g] for g in all_bg_genes), dtype=np.int32)
    print(f"  background population: {len(pop)} genes")

    # ── Load GSE list ──────────────────────────────────────────────────────
    gse_list = load_gse_to_disease(args.gse_list)
    print(f"  GSE datasets: {len(gse_list)}")

    # ── Load ground truth ──────────────────────────────────────────────────
    truth = pd.read_csv(args.truth, index_col=0)  # rows=drugs, cols=diseases
    print(f"  ground truth: {truth.shape[0]} drugs x {truth.shape[1]} diseases")

    # ── Optional: build full S matrix for old ANDES ────────────────────────
    S = None
    if args.run_old_andes:
        print("Building full similarity matrix for old ANDES (this is the slow part)...")
        from sklearn.metrics.pairwise import cosine_similarity
        node_vectors_64 = np.loadtxt(args.emb, delimiter=",", dtype=np.float32)
        S = cosine_similarity(node_vectors_64, node_vectors_64)
        print(f"  S: {S.nbytes/1e9:.2f} GB")

    # ── Score each GSE ─────────────────────────────────────────────────────
    print("\nScoring drugs against each GSE...")
    func_new.warmup_numba()

    rows_new, rows_old, rows_gsea, rows_gspa, rows_hyper = [], [], [], [], []

    sizes_needed = sorted({len(arr) for arr in drug_indices_np.values()})

    for gse_id, disease in tqdm(gse_list, desc="GSE"):
        if disease not in truth.columns:
            print(f"  skip {gse_id}: disease '{disease}' not in ground truth")
            continue

        rank_path = Path(args.rank_dir) / f"{gse_id}_rank.txt"
        if not rank_path.exists():
            print(f"  skip {gse_id}: {rank_path} not found")
            continue
        ranked_idx = load_ranked(rank_path, g_node2index, gene_set)
        if len(ranked_idx) < 100:
            print(f"  skip {gse_id}: ranked list too short ({len(ranked_idx)})")
            continue

        # Build per-GSE ES cache (depends on this ranked list)
        ranked_emb = compute_ranked_emb(E_unit, ranked_idx)
        cache = RankedNullBuilder()
        cache.precompute_parallel(
            E_unit, pop, sizes_needed, ranked_emb,
            ite=args.ite, seed=args.seed, verbose=False,
            n_workers=args.workers,
        )

        # New ANDES
        new_z = score_andes_new(E_unit, drug_indices_np, ranked_idx, cache)
        for drug, z in new_z.items():
            rows_new.append((gse_id, drug, z))

        # Old ANDES (optional)
        if args.run_old_andes:
            old_z = score_andes_old(
                S, drug_indices_set, list(ranked_idx),
                ite=args.ite, seed=args.seed,
            )
            for drug, z in old_z.items():
                rows_old.append((gse_id, drug, z))

        # Hypergeometric: drug targets vs top 500 DE genes
        top_de = set(int(g) for g in ranked_idx[: args.top_de])
        hyper = score_hypergeometric(
            {d: list(idx) for d, idx in drug_indices_set.items()},
            top_de,
            n_genes_total=len(gene_list),
        )
        for drug, score in hyper.items():
            rows_hyper.append((gse_id, drug, score))

        # GSEA / GSPA: load existing
        gsea_nes = load_existing_nes(
            Path(args.gsea_dir) / f"{gse_id}_result.csv", "GSEA"
        ) if args.gsea_dir else None
        gspa_nes = load_existing_nes(
            Path(args.gspa_dir) / f"{gse_id}_GSPA_results_our_ppi.csv", "GSPA"
        ) if args.gspa_dir else None
        if gsea_nes is not None:
            for drug in drug_terms:
                rows_gsea.append((gse_id, drug, gsea_nes.get(drug, 0.0)))
        if gspa_nes is not None:
            for drug in drug_terms:
                rows_gspa.append((gse_id, drug, gspa_nes.get(drug, 0.0)))

    # ── Convert to score matrices (drugs x GSEs) ───────────────────────────
    def _to_matrix(rows):
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["gse", "drug", "score"])
        return df.pivot(index="drug", columns="gse", values="score")

    M_new = _to_matrix(rows_new)
    M_old = _to_matrix(rows_old) if args.run_old_andes else None
    M_gsea = _to_matrix(rows_gsea)
    M_gspa = _to_matrix(rows_gspa)
    M_hyper = _to_matrix(rows_hyper)

    M_new.to_csv(out_dir / "andes_new_zscores.csv")
    if M_old is not None:
        M_old.to_csv(out_dir / "andes_old_zscores.csv")

    # ── AUPRC per (method, GSE) ────────────────────────────────────────────
    print("\nComputing AUPRC...")

    def auprc_per_gse(M, method_name):
        """For each GSE column, compute AUPRC of M[:, gse] vs truth[disease]."""
        if M is None:
            return {}
        out = {}
        for gse_id, disease in gse_list:
            if gse_id not in M.columns or disease not in truth.columns:
                continue
            scores = M[gse_id]
            y_true = truth[disease].reindex(scores.index).fillna(0).astype(int)
            if y_true.sum() == 0:
                continue
            mask = ~scores.isna()
            if mask.sum() < 2:
                continue
            label = f"{gse_id} ({_disease_short(disease)})"
            out[label] = average_precision_score(
                y_true[mask].values, scores[mask].values
            )
        return out

    auprc_new   = auprc_per_gse(M_new,   "ANDES")
    auprc_old   = auprc_per_gse(M_old,   "old ANDES") if M_old is not None else {}
    auprc_gsea  = auprc_per_gse(M_gsea,  "GSEA")
    auprc_gspa  = auprc_per_gse(M_gspa,  "GSPA")
    auprc_hyper = auprc_per_gse(M_hyper, "hypergeometric")

    # Old vs new validation table
    if auprc_old:
        keys = sorted(set(auprc_new) & set(auprc_old))
        validation = pd.DataFrame({
            "ANDES_new": [auprc_new[k]  for k in keys],
            "ANDES_old": [auprc_old[k]  for k in keys],
            "abs_diff":  [abs(auprc_new[k] - auprc_old[k]) for k in keys],
        }, index=keys).sort_values("abs_diff", ascending=False)
        validation.to_csv(out_dir / "auprc_new_vs_old.csv")
        print(f"\n  Old-vs-new ANDES AUPRC agreement (top 5 disagreements):")
        print(validation.head().to_string())
        print(f"\n  Median |ΔAUPRC| = {validation['abs_diff'].median():.4f}")
        print(f"  Max    |ΔAUPRC| = {validation['abs_diff'].max():.4f}")

    # ── Build panel CSVs ───────────────────────────────────────────────────
    def to_panel(method_to_dict):
        common = set.intersection(*[set(d) for d in method_to_dict.values()])
        if not common:
            return None
        return pd.DataFrame(
            {m: [method_to_dict[m][k] for k in sorted(common)] for m in method_to_dict},
            index=sorted(common),
        )

    panel_a = to_panel({"hypergeometric": auprc_hyper, "ANDES": auprc_new})
    panel_b = to_panel({"GSEA": auprc_gsea, "GSPA": auprc_gspa, "ANDES": auprc_new})
    # Panel C: "corrected" = z-score / NES (the ones we already have)
    panel_c = to_panel({
        "corrected GSEA":  auprc_gsea,
        "corrected GSPA":  auprc_gspa,
        "corrected ANDES": auprc_new,
    })

    if panel_a is not None: panel_a.to_csv(out_dir / "panel_A.csv")
    if panel_b is not None: panel_b.to_csv(out_dir / "panel_B.csv")
    if panel_c is not None: panel_c.to_csv(out_dir / "panel_C.csv")

    # ── Plot ───────────────────────────────────────────────────────────────
    if all(p is not None for p in (panel_a, panel_b, panel_c)):
        make_figure(panel_a, panel_b, panel_c, out_dir / "figure_drug_repurposing.pdf")
    else:
        print("Not all three panels could be built; skipping figure.")
        print(f"  panel A available: {panel_a is not None}")
        print(f"  panel B available: {panel_b is not None}")
        print(f"  panel C available: {panel_c is not None}")


def _disease_short(label):
    """Return the disease abbreviation in parens that appears in the figure
    labels, e.g. 'dilated cardiomyopathy' -> 'DCM'. If the label already looks
    like an abbreviation, return as-is."""
    if len(label) <= 6 and label.isupper():
        return label
    # Heuristic: take initials of words longer than 3 chars
    parts = [w for w in label.split() if len(w) > 3]
    if not parts:
        return label
    return "".join(w[0].upper() for w in parts)


def plot_only(args):
    out_dir = Path(args.out_dir)
    panel_a = pd.read_csv(out_dir / "panel_A.csv", index_col=0)
    panel_b = pd.read_csv(out_dir / "panel_B.csv", index_col=0)
    panel_c = pd.read_csv(out_dir / "panel_C.csv", index_col=0)
    make_figure(panel_a, panel_b, panel_c,
                out_dir / "figure_drug_repurposing.pdf")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    if len(sys.argv) > 1 and sys.argv[1] == "drug-disease":
        p = argparse.ArgumentParser()
        sub = p.add_subparsers(dest="command")
        d = sub.add_parser(
            "drug-disease",
            help="Validate optimized DrugBank x OMIM BMA z-scores against the saved reference matrix.",
        )
        d.add_argument("--emb", default="data/embedding/node2vec_consensus.csv")
        d.add_argument("--genelist", default="data/embedding/consensus_node.txt")
        d.add_argument("--drug-gmt", default="paper/data/gmt/drugbank.2301.gmt")
        d.add_argument("--disease-gmt", default="paper/data/gmt/omim.20231030.gmt")
        d.add_argument("--reference-zscores", default="paper/results/drug_disease/drug_disease_fixed_seed.csv")
        d.add_argument("--out-dir", default="reports/drug_disease_validation")
        d.add_argument("--cache", default=None)
        d.add_argument("--min-size", type=int, default=10)
        d.add_argument("--max-size", type=int, default=300)
        d.add_argument("--ite", type=int, default=1000)
        d.add_argument(
            "--seed",
            type=int,
            default=12345,
            help="Null-cache seed. Use -1 for a fresh random seed.",
        )
        d.add_argument("--workers", type=int, default=8, help="Cache-build workers")
        d.add_argument("--query-workers", type=int, default=8, help="Batched query workers")
        d.add_argument("--query-memory-mb", type=float, default=2048)
        d.add_argument("--chunk-size", type=int, default=0, help="0 enables cost-balanced cache chunks")
        d.add_argument("--rebuild-cache", action="store_true")
        d.add_argument("--limit-drugs", type=int, default=0, help="Smoke-test only the first N reference drugs")
        d.add_argument("--limit-diseases", type=int, default=0, help="Smoke-test only the first N reference diseases")
        d.add_argument("--drug-atc", default="paper/data/drug_info/drug_bank_atc.pickle")
        d.add_argument("--atc-names", default="paper/data/drug_info/atc_name.pickle")
        d.add_argument("--min-atc-count", type=int, default=12)
        d.add_argument("--skip-pca", action="store_true")
        return p.parse_args()

    p = argparse.ArgumentParser()
    p.add_argument("--emb",       required=True)
    p.add_argument("--genelist",  required=True)
    p.add_argument("--drug-gmt",  required=True, help="DrugBank GMT")
    p.add_argument("--bg-gmt",    required=True, help="Background GMT")
    p.add_argument("--gse-list",  required=True,
                   help="CSV with columns gse_id, disease_label")
    p.add_argument("--rank-dir",  required=True,
                   help="Dir with <GSE>_rank.txt files")
    p.add_argument("--truth",     required=True,
                   help="CSV: rows=drug ids, cols=disease labels, 0/1")
    p.add_argument("--gsea-dir",  default=None)
    p.add_argument("--gspa-dir",  default=None)
    p.add_argument("--out-dir",   default="results/validation")

    p.add_argument("--min-size",  type=int, default=10)
    p.add_argument("--max-size",  type=int, default=300)
    p.add_argument("--ite",       type=int, default=1000)
    p.add_argument("--seed",      type=int, default=12345)
    p.add_argument("--workers",   type=int, default=8)
    p.add_argument("--top-de",    type=int, default=500,
                   help="Top-N DE genes to use for hypergeometric test")

    p.add_argument("--run-old-andes", action="store_true",
                   help="Also run old ANDES on full S matrix for direct "
                        "validation. Adds ~30 minutes for ~30 GSEs.")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip scoring; reload panel CSVs and just plot")
    return p.parse_args()


def main():
    args = parse_args()
    if getattr(args, "command", None) == "drug-disease":
        run_drug_disease_zscore_validation(args)
    elif args.plot_only:
        plot_only(args)
    else:
        t0 = time.perf_counter()
        run_pipeline(args)
        print(f"\nTotal pipeline time: {(time.perf_counter() - t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
