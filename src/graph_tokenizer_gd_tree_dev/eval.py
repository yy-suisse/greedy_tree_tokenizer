"""
Evaluation metrics for a selected token set T, complementary to semantic coverage
(F_D(T), see new_tokenizer.compute_semantic_coverage / LazyGreedyTokenSelector).

All metrics here are computed from the context trees (new_tokenizer.tokenize_all_rel)
that T actually produces for the mapped concepts M -- i.e. they evaluate the *realized*
representation, not the coverage score used to select T in the first place.

Two exceptions, added later:
  br_weighted_PC       branch-weighted proximity-coverage. Computed on the same expansion as the
                       context trees (tokenizer.expand), but every concept splits its weight equally
                       between its outgoing relationships, so many tokens in one branch can no longer
                       hide a branch that ends without a token. See the block "branch-weighted PC" below.
  num_effective_token  number of distinct tokens of T that appear in at least one context tree,
                       i.e. the part of the vocabulary that the concepts actually use.
"""

import math
from collections import Counter
from functools import lru_cache

import numpy as np
import polars as pl
import scipy.sparse as sp

from src.graph_tokenizer_gd_tree_dev.tokenizer import signature, tokenize_all_rel


def _iter_contexts(ctx):
    """Yield ctx and every nested subcontext (pre-order) -- the whole tree, flattened."""
    yield ctx
    for sub in ctx.subcontexts:
        yield from _iter_contexts(sub)


def build_context_trees(M, adj, T, D, id_to_label=None):
    """
    tokenize_all_rel(c, adj, T, D, id_to_label) for every mapped concept c in M, once.

    adj is the precomputed out-adjacency from tokenizer.build_out_adjacency(G) -- built once
    per graph and shared across calls, rather than each call re-deriving it from G.
    """
    labels = id_to_label or {}
    T = set(T)  # expand() does `u in T` on every recursive call; on a list that's an O(|T|) scan each time
    return {c: tokenize_all_rel(c, adj, T, D, labels) for c in M}


def conciseness(trees):
    """Mean number of token leaves per concept's context tree (lower = more concise)."""
    counts = [sum(len(ctx.tokens) for ctx in _iter_contexts(tree)) for tree in trees.values()]
    return sum(counts) / len(counts) if counts else 0.0


def distance_score(trees, D):
    """
    Higher = better. Mean hop-distance at which tokens are found, averaged per concept first
    (so concepts with many tokens don't dominate) then across concepts, converted to a (0, 1]
    similarity via 1 - mean_distance / (D + 1) -- 1.0 means every concept's tokens were found
    immediately (distance 0), 0.0 means everything bottomed out at the worst case.

    Contexts marked uncovered (a branch that hit the depth budget or a dead end without
    finding a token) count as a data point at D + 1 -- one hop past the deepest horizon
    actually searched -- instead of being dropped from the average. Mirrors
    graph_tokenizer_dev's __get_distance_n_score: "UNK concepts... treat them as worst-case
    (max_dist_candidate + 1) instead of letting the mean silently skip them."
    """
    per_concept_means = []
    for tree in trees.values():
        distances = [d for ctx in _iter_contexts(tree) for _, d in ctx.tokens]
        distances += [D + 1 for ctx in _iter_contexts(tree) if ctx.uncovered]
        if distances:
            per_concept_means.append(sum(distances) / len(distances))
    mean_distance = sum(per_concept_means) / len(per_concept_means) if per_concept_means else D + 1
    return 1 - mean_distance / (D + 1)


def uniqueness_entropy(trees):
    """
    Shannon entropy (bits, normalized to [0,1]) of the distribution of context-tree
    signatures across concepts. 1.0 = every concept gets a distinct signature (maximally
    discriminative vocabulary); near 0 = most concepts collapse onto a handful of shared
    signatures (heavy collisions, low uniqueness).
    """
    sigs = [signature(tree) for tree in trees.values()]
    n = len(sigs)
    if n <= 1:
        return 1.0
    counts = Counter(sigs)
    probs = [c / n for c in counts.values()]
    h = -sum(p * math.log2(p) for p in probs)
    return h / math.log2(n)


def unk_rate(trees):
    """
    Fraction of mapped concepts whose context tree has at least one context marked
    `uncovered` -- i.e. at least one semantic branch never reached a token within the
    depth budget D. Analogous to an out-of-vocabulary rate.
    """
    n = len(trees)
    if n == 0:
        return 0.0
    n_with_unk = sum(1 for tree in trees.values() if any(ctx.uncovered for ctx in _iter_contexts(tree)))
    return n_with_unk / n


def uncovered_rate(trees):
    """
    Fraction of mapped concepts with *zero* tokens anywhere in their context tree -- not
    partially represented, not represented at all (same condition as drilldown.is_fully_unk).
    Stricter than unk_rate: a concept can have unk_rate's uncovered=True on one branch while
    still finding tokens via another (e.g. one IS_A parent dead-ends but a sibling relation
    reaches T), so it counts toward unk_rate but not uncovered_rate. uncovered_rate <=
    unk_rate always.
    """
    n = len(trees)
    if n == 0:
        return 0.0
    n_uncovered = sum(1 for tree in trees.values() if not any(ctx.tokens for ctx in _iter_contexts(tree)))
    return n_uncovered / n


def unique_rate(trees):
    """
    Fraction of mapped concepts whose flattened context-tree signature (tokenizer.signature)
    isn't shared by any other concept -- i.e. redundancy_group_size == 1 (drilldown.py). The
    signature is already graph-invariant: it's built from frozensets over a context's tokens
    and its subcontexts' signatures (tokenizer.py's `signature`), so two trees that reach the
    same tokens via subcontexts discovered/ordered differently -- multiple IS_A parents visited
    in a different order, non-IS_A relations opened in a different sequence -- still collapse to
    the same signature and are correctly treated as duplicates, not distinct shapes.
    Per-concept decomposition of uniqueness_entropy (see per_concept_metrics).
    """
    n = len(trees)
    if n == 0:
        return 0.0
    sigs = [signature(tree) for tree in trees.values()]
    counts = Counter(sigs)
    n_unique = sum(1 for s in sigs if counts[s] == 1)
    return n_unique / n


def tree_complexity(trees):
    """Mean number of contexts (root + subcontexts) per concept's context tree."""
    counts = [sum(1 for _ in _iter_contexts(tree)) for tree in trees.values()]
    return sum(counts) / len(counts) if counts else 0.0


def exact_rate(M, T):
    """Fraction of mapped concepts that are themselves selected as a token (0-hop representation)."""
    M = list(M)
    T = set(T)
    return sum(1 for c in M if c in T) / len(M) if M else 0.0


# --- number of effectively used tokens --------------------------------------------------------


def effective_tokens(trees):
    """Distinct tokens that appear in at least one context tree (the used part of T)."""
    return {tok for tree in trees.values() for ctx in _iter_contexts(tree) for tok, _ in ctx.tokens}


def num_effective_token(trees):
    """
    Number of distinct tokens of T that at least one concept uses once it is tokenized.
    A selected token is unused when no concept reaches it before reaching another token
    (e.g. a general hub whose concepts all stop at more specific tokens). Higher = less of
    the vocabulary is wasted; at most |T|.
    """
    return len(effective_tokens(trees))


# --- branch-weighted PC (br_weighted_PC) --------------------------------------------------------
#
# Same expansion as tokenizer.expand: the token test comes before the depth test, the expansion
# stops at the first token, and a branch is uncovered at a dead end or at depth D. The difference
# with distance_score (the current PC) is only the weighting of the branch ends:
#   * current PC: every branch end of the context tree weighs the same (flat average);
#   * br_weighted_PC: every concept splits its weight equally between its outgoing relationships
#     (nested average), so many tokens under one relationship cannot hide another relationship
#     that ends without a token.
#
#   v(u, d) = 1 - d / (D + 1)                                     if u is a token
#           = 0                                                   if u is not a token and (d = D or u has no out-edge)
#           = mean over the out-edges (r, w) of u of v(w, d + 1)  otherwise
#   br_weighted_PC(m) = v(m, 0)
#
# Read as a random walk from m that picks an outgoing relationship uniformly at every step and
# stops at the first token J hops away: br_weighted_PC(m) = E[1 - J / (D + 1)], with 0 when no
# token is found within D hops. It splits like the current PC:
#   br_completeness(m) = probability that the walk ends on a token (= semantic coverage at lambda = 1)
#   br_proximity(m)    = br_weighted_PC(m) / br_completeness(m)   (mean 1 - J/(D+1) given a token is found)
#   br_weighted_PC(m)  = br_completeness(m) * br_proximity(m)
#
# Out-edges are counted once per relationship occurrence (one entry per typed edge in adj), like
# the out-degree of the coverage objective. The context tree is not needed: the vectorised version
# below runs D sparse matrix products over the whole graph, i.e. every concept at once.

_WALK_CACHE = {"adj": None, "P": None, "node_to_idx": None}


def build_walk_transition(adj):
    """
    Row-normalised out-adjacency of the expansion graph (adj from tokenizer.build_out_adjacency).
    P[u, w] = (number of relationships u -> w) / out-degree(u); rows of dead ends are 0.
    Returns (P, node_to_idx).
    """
    node_to_idx = {u: i for i, u in enumerate(adj)}
    for edges in adj.values():
        for _, w in edges:
            if w not in node_to_idx:
                node_to_idx[w] = len(node_to_idx)
    n = len(node_to_idx)
    rows, cols = [], []
    for u, edges in adj.items():
        i = node_to_idx[u]
        for _, w in edges:
            rows.append(i)
            cols.append(node_to_idx[w])
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    out_degree = np.asarray(A.sum(axis=1)).ravel()
    inv = np.divide(1.0, out_degree, out=np.zeros(n), where=out_degree > 0)
    return (sp.diags(inv) @ A).tocsr(), node_to_idx


def _walk_transition(adj):
    """build_walk_transition, built once per adjacency object (per process) and then reused."""
    if _WALK_CACHE["adj"] is not adj:
        P, node_to_idx = build_walk_transition(adj)
        _WALK_CACHE.update(adj=adj, P=P, node_to_idx=node_to_idx)
    return _WALK_CACHE["P"], _WALK_CACHE["node_to_idx"]


def br_weighted_pc_per_concept(M, adj, T, D):
    """
    Per-concept branch-weighted PC for the concepts M (in the order of M).
    Returns a dict of numpy arrays: br_weighted_PC, br_completeness, br_proximity
    (br_proximity is 0 when no token is reached, like proximity in fidelity_eval).
    """
    P, idx = _walk_transition(adj)
    n = P.shape[0]
    T = set(T)
    t = np.zeros(n)
    t[[idx[x] for x in T if x in idx]] = 1.0
    not_t = 1.0 - t
    W = t / (D + 1)                     # remaining budget h = 0 (depth D): a token there scores 1/(D+1)
    R = t.copy()                        # probability of ending on a token
    for h in range(1, D + 1):           # remaining budget h <-> depth D - h
        W = t * ((h + 1) / (D + 1)) + not_t * (P @ W)
        R = t + not_t * (P @ R)
    M = list(M)
    pos = np.array([idx.get(m, -1) for m in M], dtype=np.int64)
    in_graph = pos >= 0
    own = np.array([1.0 if m in T else 0.0 for m in M])     # concepts outside the graph: token or not
    w = np.where(in_graph, W[np.where(in_graph, pos, 0)], own)
    r = np.where(in_graph, R[np.where(in_graph, pos, 0)], own)
    p = np.divide(w, r, out=np.zeros_like(w), where=r > 0)
    return {"br_weighted_PC": w, "br_completeness": r, "br_proximity": p}


def br_weighted_pc(M, adj, T, D):
    """Mean branch-weighted PC over the concepts M (higher = better)."""
    M = list(dict.fromkeys(M))
    return float(br_weighted_pc_per_concept(M, adj, T, D)["br_weighted_PC"].mean()) if M else 0.0


def br_weighted_pc_concept(c, adj, T, D):
    """
    Reference implementation for one concept, written like tokenizer.expand (slow; for checks
    and worked examples). Must equal br_weighted_pc_per_concept([c], adj, T, D).
    """
    T = set(T)

    @lru_cache(maxsize=None)
    def v(u, d):
        if u in T:
            return 1 - d / (D + 1)
        edges = adj.get(u, ())
        if d == D or not edges:
            return 0.0
        return sum(v(w, d + 1) for _, w in edges) / len(edges)

    return v(c, 0)


def evaluate(M, adj, T, D, id_to_label=None):
    """
    Compute all metrics at once for a given token set T, building the context trees only once
    and reusing them across metrics. Returns a dict of metric_name -> value.
    distance_score is the current PC (kept for versioning); br_weighted_PC is the branch-weighted PC.
    """
    M = list(dict.fromkeys(M))
    trees = build_context_trees(M, adj, T, D, id_to_label)
    return {
        "conciseness": conciseness(trees),
        "distance_score": distance_score(trees, D),
        "uniqueness_entropy": uniqueness_entropy(trees),
        "unk_rate": unk_rate(trees),
        "uncovered_rate": uncovered_rate(trees),
        "tree_complexity": tree_complexity(trees),
        "exact_rate": exact_rate(M, T),
        "unique_rate": unique_rate(trees),
        "br_weighted_PC": br_weighted_pc(M, adj, T, D),
        "num_effective_token": num_effective_token(trees),
    }


def per_concept_metrics(trees, T, D, adj=None):
    """
    Per-concept values of conciseness/distance_score/tree_complexity/unk_rate/uncovered_rate/
    exact_rate/unique_rate -- i.e. every per-tree quantity evaluate() averages/counts over
    before returning a single scalar -- as one row per concept instead. Lets callers stratify
    (e.g. group_by a concept's own concept_type, from mapped_concepts.parquet) instead of only
    ever seeing the population mean evaluate() returns. With adj (tokenizer.build_out_adjacency),
    br_weighted_PC is added as well. num_effective_token is a property of the whole vocabulary,
    not of one concept, so it has no per-concept column.

    uniqueness_entropy isn't included: it's a property of the *distribution* of signatures
    across a population of concepts, not of any single concept's tree, so it doesn't decompose
    into a per-concept column the way the others do -- recompute it per stratum instead, e.g.
    eval.uniqueness_entropy({c: t for c, t in trees.items() if c in stratum_ids}). unique_rate
    is different: "is this concept's own signature unique" is a well-defined per-concept bool
    (it's just uniqueness_entropy's redundancy_group_size == 1), so it is included below.
    """
    T = set(T)
    sigs = {c: signature(tree) for c, tree in trees.items()}
    sig_counts = Counter(sigs.values())
    rows = []
    for c, tree in trees.items():
        contexts = list(_iter_contexts(tree))
        distances = [d for ctx in contexts for _, d in ctx.tokens]
        distances += [D + 1 for ctx in contexts if ctx.uncovered]
        mean_distance = sum(distances) / len(distances) if distances else D + 1
        n_tokens = sum(len(ctx.tokens) for ctx in contexts)
        rows.append({
            "concept": c,
            "conciseness": n_tokens,
            "distance_score": 1 - mean_distance / (D + 1),
            "tree_complexity": len(contexts),
            "unk_rate": any(ctx.uncovered for ctx in contexts),
            "uncovered_rate": n_tokens == 0,
            "exact_rate": c in T,
            "unique_rate": sig_counts[sigs[c]] == 1,
        })
    df = pl.DataFrame(rows)
    if adj is not None:
        df = df.with_columns(pl.Series("br_weighted_PC", br_weighted_pc_per_concept(list(trees), adj, T, D)["br_weighted_PC"]))
    return df
