"""
Lazy greedy token selection (tokenizer.LazyGreedyTokenSelector) with a fast, exact initial scan.

The original select() seeds the heap with one CANDIDATE_DELTA call per candidate (|V| backward
propagations), which dominates the run time. With T empty, every initial gain has a closed form:
the graph is a DAG, so a walk reaches t at most once, and

    gain(t) = (1/|M|) * sum_{c in M} sum_{j=0..D} lam^j (A^j)[c, t]

with A the row-normalized out-adjacency (multi-edges counted, as in Eq. 1). For all candidates at
once this is D sparse products  r_j = A^T r_{j-1},  r_0 = 1_M / |M|,  gains = sum_j lam^j r_j.

The lazy loop afterwards is unchanged: the top of the heap is recomputed exactly with
selector.candidate_delta and committed with selector.commit, as in LazyGreedyTokenSelector.select.
"""
import heapq

import numpy as np

from src.graph_tokenizer_gd_tree_dev.tokenizer import LazyGreedyTokenSelector, build_coverage_transition


def initial_gains(G, M, D, lam):
    """Exact Delta_D(t | {}) for every node t of G. Returns {node: gain}."""
    A, node_to_idx = build_coverage_transition(G, lam=1.0)        # row-normalized, no lam folded in
    AT = A.T.tocsr()
    r = np.zeros(A.shape[0])
    r[[node_to_idx[c] for c in M]] = 1.0 / len(M)
    acc = r.copy()
    for j in range(1, D + 1):
        r = AT @ r
        acc += (lam ** j) * r
    return {u: float(acc[i]) for u, i in node_to_idx.items()}


def select_tokens(G, M, D, lam, k, check_sample=0, seed=0):
    """
    Same result as LazyGreedyTokenSelector(G, M, D, lam).select(k) (up to the order of exact ties).
    check_sample > 0: verify that many closed-form gains against candidate_delta before selecting.
    Returns the history [(token, marginal_gain, cumulative_score), ...].
    """
    sel = LazyGreedyTokenSelector(G, M, D, lam=lam)
    gains = initial_gains(G, sel.M, D, lam)
    if check_sample:
        rng = np.random.default_rng(seed)
        for t in rng.choice(sel.V, size=min(check_sample, len(sel.V)), replace=False):
            exact, _ = sel.candidate_delta(t)
            assert np.isclose(exact, gains[t], rtol=1e-9, atol=1e-12), (t, exact, gains[t])

    heap = [(-gains[t], t) for t in sel.V]
    heapq.heapify(heap)
    history = []
    while len(sel.T) < k and heap:
        while True:
            _, t = heapq.heappop(heap)
            gain, delta = sel.candidate_delta(t)                      # exact, against the current T
            best_other_bound = -heap[0][0] if heap else float("-inf")
            if gain >= best_other_bound:                              # true maximizer
                sel.commit(t, delta)
                sel.current_score += gain
                history.append((t, gain, sel.current_score))
                break
            heapq.heappush(heap, (-gain, t))                          # stale bound: re-queue
    return history
