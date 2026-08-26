"""
Evaluation metrics for a selected token set T, complementary to semantic coverage
(F_D(T), see new_tokenizer.compute_semantic_coverage / LazyGreedyTokenSelector).

All metrics here are computed from the context trees (new_tokenizer.tokenize_all_rel)
that T actually produces for the mapped concepts M -- i.e. they evaluate the *realized*
representation, not the coverage score used to select T in the first place.
"""

import math
from collections import Counter

import polars as pl

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


def evaluate(M, adj, T, D, id_to_label=None):
    """
    Compute all eight metrics at once for a given token set T, building the context trees
    only once and reusing them across metrics. Returns a dict of metric_name -> value.
    """
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
    }


def per_concept_metrics(trees, T, D):
    """
    Per-concept values of conciseness/distance_score/tree_complexity/unk_rate/uncovered_rate/
    exact_rate/unique_rate -- i.e. every per-tree quantity evaluate() averages/counts over
    before returning a single scalar -- as one row per concept instead. Lets callers stratify
    (e.g. group_by a concept's own concept_type, from mapped_concepts.parquet) instead of only
    ever seeing the population mean evaluate() returns.

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
    return pl.DataFrame(rows)
