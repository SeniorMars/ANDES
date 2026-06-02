"""
load_data.py — GMT file parsing and gene-set index filtering

GMT format: one gene set per line, tab-separated.
  col 0 : term ID
  col 1 : term description (skipped)
  col 2+: gene IDs

Functions here handle only I/O and filtering; scoring and null builds live in
func_optimized.py and func_gsea.py.
"""

from collections import defaultdict


def load_gmt(file):
    """Parse a GMT file and return a dict mapping term ID → list of gene IDs.

    The description field (column 1) is discarded.  Gene IDs are returned as
    raw strings; callers apply node2index mapping via term2indexes.
    """
    ret = defaultdict(list)
    with open(file, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            term = tokens[0]
            for x in tokens[2:]:
                ret[term].append(x)
                
    return ret

def term2name(file_name):
    """Parse a GMT file and return a dict mapping term ID → description string (column 1)."""
    term2name = {}
    with open(file_name, 'r') as f:
        for line in f:
            tokens = line.split('\t')
            tokens = [x.strip() for x in tokens]
            term2name[tokens[0]]=tokens[1]
            
    return term2name


def term2indexes(go_dict, node2index, upper=300, lower=5):
    """Map gene-set gene IDs to embedding indices and filter by size.

    Genes not present in node2index (i.e., absent from the embedding) are
    dropped.  Terms with fewer than `lower` or more than `upper` surviving
    genes are excluded entirely.

    Parameters
    ----------
    go_dict : dict {str: list of str}
        Raw gene sets from load_gmt.
    node2index : dict {str: int}
        Gene ID → embedding row index mapping.  Must return -1 (or raise
        KeyError caught by .get) for unknown genes.
    upper, lower : int
        Inclusive size bounds after filtering.

    Returns
    -------
    defaultdict {str: set of int}
        Filtered gene sets as index sets.
    """
    ret = defaultdict(set)
    for key in go_dict:
        genes = go_dict[key]
        genes = [node2index.get(x, -1) for x in genes]
        genes = [x for x in genes if x != -1]
        if len(genes)>=lower and len(genes)<=upper:
            ret[key] = set(genes)
    return ret
