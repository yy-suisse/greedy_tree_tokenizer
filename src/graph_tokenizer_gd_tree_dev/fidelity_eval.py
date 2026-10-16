"""
Evaluation used by "(final)ehr_application_analysis.ipynb".

Same context trees as eval.py (tokenizer.tokenize_all_rel), but with the metric set of the
redrafted paper:

  core metrics (one value per token set T)
    semantic_fidelity    mean over concepts of f_m = completeness_m x proximity_m   (higher = better)
    unique_rate          share of concepts that have >= 1 token AND a context tree no other
                         concept has                                                (higher = better)
    unknown_rate         share of concepts without any token                         (lower  = better)
    tokens_per_concept   mean number of token leaves per context tree                (lower  = better)

  secondary / diagnostic
    uncovered_branch_rate  share of concepts with >= 1 branch that ends without a token
    completeness           mean over concepts of n_m / (n_m + u_m)
    proximity              mean over REPRESENTED concepts (n_m > 0) of 1 - mean token depth / (D + 1)
    fidelity_median, fidelity_p10, fidelity_p25   distribution of f_m

Per concept m (n_m token leaves at depths l_1..l_n, u_m uncovered branches):
    completeness_m = n_m / (n_m + u_m)
    proximity_m    = 1 - mean(l_i) / (D + 1)          (0 if n_m = 0)
    fidelity_m     = completeness_m * proximity_m
                   = 1 - mean over the n_m + u_m entries of depth / (D + 1),
                     uncovered entries counted at depth D + 1
fidelity_m is exactly the per-concept "distance score" of eval.py (uncovered branches at D + 1);
the product form only makes its two components visible.

Semantic coverage (Eq. 1 at lambda = 1) is no longer computed here.

The worker functions are module-level so that multiprocessing works on Windows (spawn), where
functions defined inside a notebook cannot be sent to worker processes.
"""

from collections import Counter

import polars as pl

from src.graph_tokenizer_gd_tree_dev.eval import _iter_contexts, build_context_trees
from src.graph_tokenizer_gd_tree_dev.tokenizer import signature

CORE_METRICS = ["semantic_fidelity", "unique_rate", "unknown_rate", "tokens_per_concept"]
HIGHER_IS_BETTER = {
    "semantic_fidelity": True,
    "unique_rate": True,
    "unknown_rate": False,
    "tokens_per_concept": False,
    "uncovered_branch_rate": False,
    "completeness": True,
    "proximity": True,
    "fidelity_median": True,
    "fidelity_p10": True,
    "fidelity_p25": True,
}


# ---------------------------------------------------------------------------------------------
# per concept
# ---------------------------------------------------------------------------------------------
def concept_record(tree, D):
    """Per-concept quantities of one context tree (plain Python values)."""
    ctxs = list(_iter_contexts(tree))
    depths = [d for ctx in ctxs for _, d in ctx.tokens]
    n = len(depths)
    u = sum(1 for ctx in ctxs if ctx.uncovered)
    completeness = n / (n + u) if (n + u) else 0.0
    proximity = 1.0 - (sum(depths) / n) / (D + 1) if n else 0.0
    return {
        "n_tokens": n,
        "n_uncovered": u,
        "n_contexts": len(ctxs),
        "mean_token_depth": (sum(depths) / n) if n else None,
        "completeness": completeness,
        "proximity": proximity,
        "fidelity": completeness * proximity,
    }


def per_concept_table(trees, D):
    """One row per concept: concept_record + signature group size and id."""
    sigs = {c: signature(t) for c, t in trees.items()}
    counts = Counter(sigs.values())
    sig_id = {}
    rows = []
    for c, tree in trees.items():
        r = concept_record(tree, D)
        r["concept"] = c
        r["group_size"] = counts[sigs[c]]
        r["signature_id"] = sig_id.setdefault(sigs[c], len(sig_id))
        rows.append(r)
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        (pl.col("n_tokens") == 0).alias("unknown"),
        (pl.col("n_uncovered") > 0).alias("has_uncovered_branch"),
        ((pl.col("group_size") == 1) & (pl.col("n_tokens") > 0)).alias("unique"),
    )


def summarize(pc):
    """Aggregate metrics of one token set from its per-concept table."""
    f = pc["fidelity"]
    represented = pc.filter(~pl.col("unknown"))
    return {
        "semantic_fidelity": float(f.mean()),
        "unique_rate": float(pc["unique"].mean()),
        "unknown_rate": float(pc["unknown"].mean()),
        "tokens_per_concept": float(pc["n_tokens"].mean()),
        "uncovered_branch_rate": float(pc["has_uncovered_branch"].mean()),
        "completeness": float(pc["completeness"].mean()),
        "proximity": float(represented["proximity"].mean()) if represented.height else 0.0,
        "fidelity_median": float(f.median()),
        "fidelity_p10": float(f.quantile(0.10, "nearest")),
        "fidelity_p25": float(f.quantile(0.25, "nearest")),
    }


def evaluate_token_set(M, adj, T, D, id_to_label=None, return_per_concept=False):
    """All metrics of one token set T for the concepts M (serial)."""
    trees = build_context_trees(M, adj, T, D, id_to_label)
    pc = per_concept_table(trees, D)
    out = summarize(pc)
    return (out, pc) if return_per_concept else out


# ---------------------------------------------------------------------------------------------
# multiprocessing workers (Pool(initializer=init_worker, initargs=(adj, M, D)))
# ---------------------------------------------------------------------------------------------
_STATE = {}


def init_worker(adj, M, D):
    _STATE.update(adj=adj, M=M, D=D)


def worker(task):
    """task = (label, T) -> (label, metrics dict). label is carried through unchanged."""
    label, T = task
    return label, evaluate_token_set(_STATE["M"], _STATE["adj"], T, _STATE["D"])
