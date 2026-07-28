"""
download_geo_expression.py — download GEO datasets and generate ranked lists

Reads geo2kegg.txt, downloads each dataset from NCBI GEO via GEOparse,
maps probes to Entrez gene IDs via the GPL annotation table, infers
case/control from sample characteristics, and writes a two-column TSV:
    <entrez_gene_id>  <t_statistic>
sorted descending by t_statistic (case vs control, Welch's t-test).

Requirements (install with `uv sync --extra geo`):
    GEOparse >= 2.0

Usage:
    uv run python tools/download_geo_expression.py \\
        --geo2kegg paper/data/geo2kegg.txt \\
        --out-dir  data/expression \\
        --cache    /tmp/geo_cache
"""

import sys
import re
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

try:
    import GEOparse
except ImportError:
    sys.exit("GEOparse not installed. Run: uv sync --extra geo")

ROOT = Path(__file__).parent.parent
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Subset rules for datasets with sample-type suffixes ──────────────────────
# Maps geo_id_with_suffix → list of keywords, ANY of which must appear in the
# combined sample metadata text (case-insensitive).  Simpler than key-value
# matching because GEO characteristic key names vary across datasets.
SUBSET_RULES = {
    "GSE14924_CD4":       ["cd4"],
    "GSE14924_CD8":       ["cd8"],
    "GSE5281_EC":         ["entorhinal"],
    "GSE5281_HIP":        ["hippocampus"],
    "GSE5281_VCX":        ["visual cortex", "vcx"],
    "GSE24739_G0":        ["g0"],
    "GSE24739_G1":        ["g1"],
    "GSE6956AA":          ["african american", "african-american", " aa"],
    "GSE6956C":           ["caucasian"],
    "GSE38666_epithelia": ["epithelial"],
    "GSE38666_stroma":    ["stromal", "stroma"],
}

CASE_WORDS = {
    # Generic disease labels
    "cancer", "tumor", "tumour", "carcinoma", "adenocarcinoma",
    "lymphoma", "neoplasm", "malignant", "patient", "affected",
    # Leukemia / blood cancers
    "leukemia", "leukaemia", "aml", "cml", "mds",
    # Neurological
    "alzheimer", "parkinson", "huntington", "huntington's",
    "mci",        # mild cognitive impairment
    "glioma", "glioblastoma", "gbm", "lgg",
    # Other specific diseases
    "lupus", "les",
    "diabetes", "dmnd",
    "cardiomyopathy", "dcm",
    "copd", "pulmonary",
}
CTRL_WORDS = {
    "normal", "healthy", "control", "benign", "adjacent",
    "uninvolved", "non-tumor", "non-cancer", "non-disease",
    "donor",   # "normal donor", "healthy donor"
}

# ── Explicit title-pattern rules for datasets with no characteristics ─────────
# Maps geo_id → (case_substring, ctrl_substring), both matched case-insensitively
# against the sample title.  Used only when all other strategies fail.
TITLE_PATTERN_RULES = {
    # DCM study: PGA_PA-D_xxx = DCM patient, PGA_PA-N_xxx = normal
    "GSE1145": ("pa-d", "pa-n"),
    # Renal cancer: normal source mentions "adjacent to Renal Cell Carcinoma"
    # which cancels keyword scoring; title is unambiguous
    "GSE781":  ("renal clear cell carcinoma", "normal human kidney"),
}

# Candidate column names for Entrez gene ID in GPL annotation tables
ENTREZ_COL_PATTERNS = [
    "ENTREZ_GENE_ID", "Entrez_Gene_ID", "Entrez Gene ID", "EntrezGeneID",
    "GENE", "Gene ID", "GeneID", "entrez_id",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_geo2kegg(path):
    """Return list of geo_ids (with suffixes) from geo2kegg.txt."""
    ids = []
    with open(path) as f:
        for line in f:
            tok = line.strip().split()
            if tok:
                ids.append(tok[0])
    return ids


def base_geo_id(geo_id):
    """GSE14924_CD4 → GSE14924   GSE6956AA → GSE6956  (keep only GSE+digits)."""
    m = re.match(r"(GSE\d+)", geo_id)
    return m.group(1) if m else geo_id


def get_entrez_col(gpl_table):
    """Return the Entrez gene ID column name from a GPL table, or None."""
    cols = list(gpl_table.columns)
    for pattern in ENTREZ_COL_PATTERNS:
        for col in cols:
            if col.strip().lower() == pattern.lower():
                return col
    # Fallback: any column whose name contains "entrez" (catches "Entrez Gene",
    # "entrez_gene", "EntrezID", etc. used by non-standard platforms)
    for col in cols:
        if "entrez" in col.lower():
            return col
    return None


def build_probe_to_entrez(gpl_table):
    """Return {probe_id: entrez_id} mapping (first valid Entrez for each probe)."""
    col = get_entrez_col(gpl_table)
    if col is None:
        log.warning("No Entrez gene ID column found in platform annotation.")
        return {}

    # GEOparse reads platform tables with index_col=None, so the probe ID is
    # in an 'ID' column, not the DataFrame index.  Fall back to index if absent.
    id_col = next((c for c in gpl_table.columns
                   if c.strip().upper() in ("ID", "PROBE_ID", "PROBEID")), None)

    probe2entrez = {}
    for _, row in gpl_table.iterrows():
        probe_id = str(row[id_col]).strip() if id_col else str(row.name)
        raw = str(row[col]).strip()
        if raw in ("", "nan", "---", "NULL"):
            continue
        # Some platforms store multiple IDs space- or ///-separated; take first
        first = raw.split()[0].split("///")[0].strip()
        if first.isdigit():
            probe2entrez[probe_id] = first
    return probe2entrez


def sample_characteristics_flat(gsm):
    """Return a single lowercase string of all sample characteristics."""
    chars = gsm.metadata.get("characteristics_ch1", [])
    return " ".join(chars).lower()


def sample_source_flat(gsm):
    """Return a single lowercase string of sample source/title info."""
    parts = []
    for key in ("source_name_ch1", "title", "description"):
        parts.extend(gsm.metadata.get(key, []))
    return " ".join(parts).lower()


def _kw_score(text):
    """Signed keyword score: positive → case evidence, negative → control."""
    score = 0
    for w in CASE_WORDS:
        if w in text:
            score += 1
    for w in CTRL_WORDS:
        if w in text:
            score -= 1
    return score


def classify_samples_by_study(gsms, geo_id=None):
    """
    Robust within-study case/control classification.  Three strategies tried in order.

    Strategy 1 — characteristic key polarization (3 sub-modes):
      A. Key has explicit case values (score>0) AND control values (score<0).
      B. Key has explicit control values; non-control/non-absent values → case.
         Handles "Diagnosis: AML" present only in disease samples, absent in normals.
      C. Key has explicit case values; non-case values → control (case vs rest).

    Strategy 2 — source/title scoring + case-vs-rest promotion:
      Scores source_name and title only (not characteristics, to avoid study-level
      disease names that appear in ALL samples' metadata).

    Strategy 3 — explicit title-pattern rules (TITLE_PATTERN_RULES dict):
      For datasets with opaque sample codes and no usable metadata.

    Returns {gsm_id: 0/1/None} or None if no split can be inferred.
    """
    # Parse characteristics_ch1 as {key: [value, ...]} per sample
    sample_kv = {}
    for gsm_id, gsm in gsms.items():
        kv = defaultdict(list)
        for char in gsm.metadata.get("characteristics_ch1", []):
            if ":" in char:
                k, v = char.split(":", 1)
                kv[k.strip().lower()].append(v.strip().lower())
            else:
                kv["_misc"].append(char.strip().lower())
        sample_kv[gsm_id] = dict(kv)

    all_keys = set().union(*(set(d) for d in sample_kv.values()))

    best_assignment = None
    best_polarity   = -1

    for key in all_keys:
        # Include samples where this key is absent as a special "_absent_" group
        val_to_samples = defaultdict(list)
        for gsm_id, kv in sample_kv.items():
            vals = kv.get(key, [])
            for v in (vals if vals else ["_absent_"]):
                val_to_samples[v].append(gsm_id)

        if len(val_to_samples) < 2:
            continue

        val_score = {v: (_kw_score(v) if v != "_absent_" else 0)
                     for v in val_to_samples}
        pos_vals = [v for v, s in val_score.items() if s > 0]
        neg_vals = [v for v, s in val_score.items() if s < 0]
        neu_vals = [v for v, s in val_score.items() if s == 0 and v != "_absent_"]
        abs_vals = ["_absent_"] if "_absent_" in val_to_samples else []

        assignment = None
        polarity   = 0

        if pos_vals and neg_vals:
            # A: explicit case AND control labels
            polarity   = (max(val_score[v] for v in pos_vals)
                          - min(val_score[v] for v in neg_vals))
            assignment = {}
            for v, ids in val_to_samples.items():
                lbl = 1 if val_score[v] > 0 else (0 if val_score[v] < 0 else None)
                for sid in ids:
                    assignment[sid] = lbl

        elif neg_vals and (pos_vals or neu_vals):
            # B: control identified; non-control non-absent → case by elimination
            case_candidates = [v for v in val_to_samples if v not in neg_vals
                               and v != "_absent_"]
            if case_candidates:
                polarity   = abs(min(val_score[v] for v in neg_vals))
                assignment = {}
                for v, ids in val_to_samples.items():
                    if v in neg_vals:
                        lbl = 0
                    elif v == "_absent_":
                        lbl = None
                    else:
                        lbl = 1  # case by elimination
                    for sid in ids:
                        assignment[sid] = lbl

        elif pos_vals and (neg_vals or neu_vals or abs_vals):
            # C: case identified; non-case → control (case vs rest)
            polarity   = max(val_score[v] for v in pos_vals)
            assignment = {}
            for v, ids in val_to_samples.items():
                lbl = 1 if val_score[v] > 0 else 0
                for sid in ids:
                    assignment[sid] = lbl

        if assignment is not None and polarity > best_polarity:
            has_case = any(l == 1 for l in assignment.values())
            has_ctrl = any(l == 0 for l in assignment.values())
            if has_case and has_ctrl:
                best_polarity   = polarity
                best_assignment = assignment

    if best_assignment:
        return best_assignment

    # Strategy 2: score source_name / title (not characteristics, to avoid
    # study-level disease names appearing in every sample's metadata)
    assignment = {}
    for gsm_id, gsm in gsms.items():
        text = sample_source_flat(gsm)
        s = _kw_score(text)
        assignment[gsm_id] = 1 if s > 0 else (0 if s < 0 else None)

    has_case = any(v == 1 for v in assignment.values())
    has_ctrl = any(v == 0 for v in assignment.values())

    # Case-vs-rest promotion: if case identified but no clear controls,
    # promote samples with zero case evidence to control
    if has_case and not has_ctrl:
        for gsm_id in list(assignment.keys()):
            if assignment[gsm_id] is None:
                text = sample_source_flat(gsms[gsm_id])
                if not any(w in text for w in CASE_WORDS):
                    assignment[gsm_id] = 0
        has_ctrl = any(v == 0 for v in assignment.values())

    if has_case and has_ctrl:
        return assignment

    # Strategy 3: explicit title-pattern rules for opaque sample codes
    if geo_id:
        patterns = TITLE_PATTERN_RULES.get(geo_id)
        if patterns:
            case_pat, ctrl_pat = patterns
            assignment = {}
            for gsm_id, gsm in gsms.items():
                title = " ".join(gsm.metadata.get("title", [])).lower()
                if case_pat in title:
                    assignment[gsm_id] = 1
                elif ctrl_pat in title:
                    assignment[gsm_id] = 0
                else:
                    assignment[gsm_id] = None
            has_case = any(v == 1 for v in assignment.values())
            has_ctrl = any(v == 0 for v in assignment.values())
            if has_case and has_ctrl:
                return assignment

    return None


def apply_subset_rule(gsms, geo_id):
    """
    Filter sample dict to only those whose combined metadata contains at least
    one of the keywords listed in SUBSET_RULES[geo_id].
    Returns the full dict if no rule is defined for geo_id.
    """
    keywords = SUBSET_RULES.get(geo_id)
    if not keywords:
        return gsms

    def matches(gsm):
        text = sample_characteristics_flat(gsm) + " " + sample_source_flat(gsm)
        return any(kw in text for kw in keywords)

    filtered = {k: v for k, v in gsms.items() if matches(v)}
    if not filtered:
        log.warning(f"  Subset rule for {geo_id} matched 0/{len(gsms)} samples; using all")
        return gsms
    log.info(f"  Subset rule for {geo_id}: {len(filtered)}/{len(gsms)} samples kept")
    return filtered


def build_expression_matrix(gse, sample_ids):
    """
    Return a (genes × samples) DataFrame of expression values for sample_ids.
    Tries common value column names used by GEOparse.
    """
    VALUE_COLS = ["VALUE", "VALUE ", "NORMALIZED VALUE", "log2 ratio"]
    for col in VALUE_COLS:
        try:
            df = gse.pivot_samples(col)[sample_ids]
            df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
            if df.shape[0] > 0:
                return df
        except (KeyError, Exception):
            continue
    raise RuntimeError(f"Could not extract expression matrix from {gse.name}")


def ranked_list_from_matrix(expr, condition):
    """
    expr:      (genes × samples) DataFrame
    condition: array-like of 0/1 aligned with expr.columns
    Returns:   Series {gene_id: t_stat} sorted descending.
    Requires n >= 2 in each group (Welch t-test needs at least 2 per group).
    """
    import warnings
    condition = np.array(condition)
    case_mask = condition == 1
    ctrl_mask = condition == 0
    n_case, n_ctrl = case_mask.sum(), ctrl_mask.sum()
    if n_case < 2 or n_ctrl < 2:
        raise ValueError(
            f"Need ≥ 2 samples per group for Welch t-test; "
            f"got case={n_case}, ctrl={n_ctrl}."
        )

    mat = expr.values.astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        t_stats, _ = stats.ttest_ind(mat[:, case_mask], mat[:, ctrl_mask],
                                     axis=1, equal_var=False, nan_policy="omit")
    s = pd.Series(t_stats, index=expr.index.astype(str))
    s = s.dropna().sort_values(ascending=False)
    return s


def aggregate_probes_to_genes(probe_series, probe2entrez):
    """
    Map probe IDs → Entrez IDs.  When multiple probes map to the same gene,
    keep the one with the highest absolute t-statistic.
    Returns a Series indexed by Entrez gene ID.
    """
    entrez2best = {}
    for probe_id, t in probe_series.items():
        entrez = probe2entrez.get(str(probe_id))
        if entrez is None:
            continue
        if entrez not in entrez2best or abs(t) > abs(entrez2best[entrez]):
            entrez2best[entrez] = t

    s = pd.Series(entrez2best)
    return s.sort_values(ascending=False)


# ── Main processing ───────────────────────────────────────────────────────────

def process_geo(geo_id, cache_dir, out_dir, overwrite=False):
    out_path = Path(out_dir) / f"{geo_id}_rank.txt"
    if out_path.exists() and not overwrite:
        log.info(f"  {geo_id}: already exists, skipping ({out_path})")
        return True

    gse_id = base_geo_id(geo_id)
    log.info(f"\n{'='*60}")
    log.info(f"Processing {geo_id}  (base: {gse_id})")

    try:
        gse = GEOparse.get_GEO(geo=gse_id, destdir=str(cache_dir), silent=True)
    except Exception as e:
        log.error(f"  Download failed: {e}")
        return False

    # Platform annotation
    if not gse.gpls:
        log.error(f"  No platform annotation found for {gse_id}")
        return False
    gpl = list(gse.gpls.values())[0]
    probe2entrez = build_probe_to_entrez(gpl.table)
    if not probe2entrez:
        log.error(f"  Could not build probe→Entrez map for {gse_id}")
        return False
    log.info(f"  Platform {gpl.name}: {len(probe2entrez)} probe→Entrez mappings")

    # Sample subset
    gsms = apply_subset_rule(dict(gse.gsms), geo_id)

    # Case/control classification — use within-study variation first
    assignment = classify_samples_by_study(gsms, geo_id=geo_id)
    if assignment is None:
        log.error(
            f"  Could not classify case/control for {geo_id}.  "
            "No characteristic key polarises cleanly into disease vs normal.  "
            "Run with --geos {geo_id} --debug-chars to inspect metadata."
        )
        return False

    case_ids = [sid for sid, lbl in assignment.items() if lbl == 1]
    ctrl_ids = [sid for sid, lbl in assignment.items() if lbl == 0]
    ambiguous = [sid for sid, lbl in assignment.items() if lbl is None]
    if ambiguous:
        log.warning(f"  {len(ambiguous)} unlabelled samples excluded")
    log.info(f"  case={len(case_ids)}  control={len(ctrl_ids)}")

    if not case_ids or not ctrl_ids:
        log.error(
            f"  Classification produced case={len(case_ids)}, ctrl={len(ctrl_ids)}.  "
            "Check sample characteristics."
        )
        return False

    # Expression matrix
    sample_ids = case_ids + ctrl_ids
    try:
        expr = build_expression_matrix(gse, sample_ids)
    except RuntimeError as e:
        log.error(f"  {e}")
        return False

    condition = [1] * len(case_ids) + [0] * len(ctrl_ids)
    log.info(f"  Expression matrix: {expr.shape[0]} probes × {expr.shape[1]} samples")

    # T-statistics per probe
    probe_t = ranked_list_from_matrix(expr, condition)

    # Aggregate probes → Entrez genes
    gene_t = aggregate_probes_to_genes(probe_t, probe2entrez)
    if gene_t.empty:
        log.error(f"  No Entrez-mapped genes after aggregation for {geo_id}")
        return False

    log.info(f"  {len(gene_t)} genes in ranked list")

    # Save
    gene_t.to_csv(out_path, sep="\t", header=False)
    log.info(f"  Saved: {out_path}")
    return True


def main():
    p = argparse.ArgumentParser(
        description="Download GEO datasets and generate t-statistic ranked lists"
    )
    p.add_argument("--geo2kegg", default="paper/data/geo2kegg.txt",
                   help="GEO→disease mapping (default: paper/data/geo2kegg.txt)")
    p.add_argument("--out-dir",  default="data/expression",
                   help="Output directory for _rank.txt files (default: data/expression)")
    p.add_argument("--cache",    default="/tmp/geo_cache",
                   help="Directory for GEOparse SOFT file cache (default: /tmp/geo_cache)")
    p.add_argument("--geos",     nargs="+", default=None,
                   help="Only process these GEO IDs (default: all in geo2kegg)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-download and re-process even if output already exists")
    p.add_argument("--debug-chars", metavar="GEO_ID", default=None, dest="debug_chars",
                   help="Print all sample characteristics for one GEO ID then exit")
    args = p.parse_args()

    geo2kegg_path = ROOT / args.geo2kegg if not Path(args.geo2kegg).is_absolute() else Path(args.geo2kegg)
    out_dir = ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    cache_dir = Path(args.cache)

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Debug mode: print sample characteristics and exit
    if args.debug_chars:
        gse_id = base_geo_id(args.debug_chars)
        gse = GEOparse.get_GEO(geo=gse_id, destdir=str(cache_dir), silent=True)
        gsms = apply_subset_rule(dict(gse.gsms), args.debug_chars)
        print(f"\n{args.debug_chars}  ({len(gsms)} samples after subset filter)")
        for sid, gsm in list(gsms.items())[:20]:
            chars = gsm.metadata.get("characteristics_ch1", [])
            src   = gsm.metadata.get("source_name_ch1", [])
            title = gsm.metadata.get("title", [])
            print(f"  {sid}  title={title}  src={src}  chars={chars}")
        sys.exit(0)

    geo_ids = parse_geo2kegg(geo2kegg_path)
    if args.geos:
        geo_ids = [g for g in geo_ids if g in args.geos]

    log.info(f"Processing {len(geo_ids)} GEO datasets → {out_dir}")

    succeeded, failed = [], []
    for geo_id in geo_ids:
        ok = process_geo(geo_id, cache_dir, out_dir, overwrite=args.overwrite)
        (succeeded if ok else failed).append(geo_id)

    print(f"\n{'='*60}")
    print(f"Done: {len(succeeded)} succeeded, {len(failed)} failed")
    if failed:
        print(f"Failed: {failed}")
        print("\nInspect characteristics for a failed dataset:")
        print("  uv run python tools/download_geo_expression.py --debug-chars <ID>")
        print("Then re-run with:")
        print(
            "  uv run python tools/download_geo_expression.py "
            "--geos <ID> --overwrite"
        )


if __name__ == "__main__":
    main()
