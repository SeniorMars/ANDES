"""
compare_scores.py — Check that new and old pipeline z-scores agree.

Called by run_benchmarks.sh after both new and old runs complete.
Exits 0 if Spearman ρ ≥ 0.90 for all comparisons, 1 otherwise.

Usage
-----
  python benchmarks/compare_scores.py \\
      --andes-new reports/andes_new_scores.csv \\
      --andes-old reports/andes_old_scores.csv \\
      --gsea-new  reports/gsea_new_scores.csv  \\
      --gsea-old  reports/gsea_old_scores.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


RHO_WARN = 0.90


def _metrics(v_new, v_old, label):
    mask = np.isfinite(v_new) & np.isfinite(v_old)
    v_new, v_old = v_new[mask], v_old[mask]
    n = int(mask.sum())
    if n < 3:
        print(f"[{label}]  only {n} finite values — skipping")
        return True

    rho, _  = spearmanr(v_new, v_old)
    r,   _  = pearsonr(v_new, v_old)
    mad     = float(np.abs(v_new - v_old).mean())
    sign_ag = float((np.sign(v_new) == np.sign(v_old)).mean())

    status = "PASS" if rho >= RHO_WARN else f"WARN (ρ < {RHO_WARN})"
    print(f"[{label}]  n={n}  ρ={rho:.4f}  r={r:.4f}  mean|Δz|={mad:.4f}"
          f"  sign-agree={sign_ag:.1%}  → {status}")
    return rho >= RHO_WARN


def compare_andes(new_path, old_path):
    new_df = pd.read_csv(new_path, index_col=0)
    old_df = pd.read_csv(old_path, index_col=0)
    rows = new_df.index.intersection(old_df.index)
    cols = new_df.columns.intersection(old_df.columns)
    v_new = new_df.loc[rows, cols].values.ravel().astype(np.float64)
    v_old = old_df.loc[rows, cols].values.ravel().astype(np.float64)
    return _metrics(v_new, v_old, "ANDES z-scores")


def compare_gsea(new_path, old_path):
    new_df = pd.read_csv(new_path, index_col=0)
    old_df = pd.read_csv(old_path, index_col=0)
    common = new_df.index.intersection(old_df.index)
    v_new = new_df.loc[common, "z_score"].values.astype(np.float64)
    v_old = old_df.loc[common, "z_score"].values.astype(np.float64)
    return _metrics(v_new, v_old, "GSEA  z-scores")


def parse_args():
    p = argparse.ArgumentParser(description="Compare new vs old pipeline z-scores")
    p.add_argument("--andes-new", default="")
    p.add_argument("--andes-old", default="")
    p.add_argument("--gsea-new",  default="")
    p.add_argument("--gsea-old",  default="")
    return p.parse_args()


def main():
    args = parse_args()
    all_pass = True

    if args.andes_new and args.andes_old:
        all_pass &= compare_andes(args.andes_new, args.andes_old)

    if args.gsea_new and args.gsea_old:
        all_pass &= compare_gsea(args.gsea_new, args.gsea_old)

    if not (args.andes_new or args.gsea_new):
        print("No score files specified. Pass --andes-new/--andes-old and/or "
              "--gsea-new/--gsea-old")
        sys.exit(1)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
