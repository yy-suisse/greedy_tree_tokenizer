"""
Status-dependent variant of the greedy tree tokenizer.

The original code (src/graph_tokenizer_gd_tree_dev/tokenizer.py) multiplies every hop by the
distance decay factor lambda. Here a hop u -> v is multiplied by lambda only when its target v is
a *primitive* concept; a hop into a *fully defined* concept is free:

    S_h(u, T) = 1                                                     if u in T
              = 0                                                     if u not in T and d+(u) = 0
              = 1/d+(u) * sum_{(u,R,v) in Out(u)} w(v) * S_{h-1}(v, T)  otherwise

    w(v) = lambda  if v is primitive
         = 1       if v is fully defined

So along a path m -> v1 -> v2 -> t, every hop target (v1, v2 and the token t) costs lambda if it is
primitive; the mapped concept m itself is never counted. lambda = 1 gives back the original Eq. (1)
exactly, and the horizon D is still counted in hops.

Distance score (evaluation only): a token's depth is the number of primitive hop targets on the
path from the mapped concept to that token (same counting rule as above, i.e. the token is counted,
the mapped concept is not). Uncovered contexts still count as D + 1, so

    distance score = 1 - mean_m( mean_{p in P_m} l_p ) / (D + 1)

is unchanged in form; only l_p changes. All other metrics (and every context-tree helper in
src/graph_tokenizer_gd_tree_dev/eval.py) read the trees built here unchanged: the tree structure,
its tokens and its signature are identical to the original ones, only the stored depths differ.

Monotonicity and submodularity still hold: S_D(u, T) is the probability that a walk on a
sub-stochastic chain (transition A[u, v] = w(v) / d+(u), the missing mass ends the walk) visits T
within D steps, i.e. the probability of a union of events. The (1 - 1/e) greedy guarantee stays.
"""

import heapq
import multiprocessing as mp
import os

import numpy as np
import polars as pl
import scipy.sparse as sp

from src.graph_tokenizer_gd_tree_dev import eval as base_eval
from src.graph_tokenizer_gd_tree_dev.tokenizer import (
    GREEDY_RATIO,
    Context,
    LazyGreedyTokenSelector,
    build_out_adjacency,
    build_out_edges,
    collapse_redundant,
    compute_semantic_coverage,
    semantic_coverage_score,
)

PRIMITIVE = "primitive"
DEFINED = "defined"


# --- concept status ---------------------------------------------------------


def load_is_primitive(path, nodes=None, missing_as_primitive=True, verbose=True):
    """
    {concept id: True if primitive, False if fully defined} from concept_and_status.parquet
    (columns id, status with values "primitive" / "defined").

    nodes: if given (e.g. G.nodes()), the result covers exactly these concepts; concepts without
    a status get `missing_as_primitive` and their number is printed. The selector and the
    context-tree builders below need a status for every node of the graph.
    """
    df = pl.read_parquet(path, columns=["id", "status"]).filter(pl.col("status").is_not_null())
    unexpected = set(df["status"].unique().to_list()) - {PRIMITIVE, DEFINED}
    if unexpected:
        raise ValueError(f"unexpected status values {unexpected}; expected '{PRIMITIVE}' or '{DEFINED}'")
    if df["id"].is_duplicated().any():
        raise ValueError("concept_and_status has more than one status for some ids")
    status = dict(zip(df["id"].to_list(), (df["status"] == PRIMITIVE).to_list()))
    if nodes is None:
        return status

    nodes = list(nodes)
    n_missing = sum(1 for u in nodes if u not in status)
    if verbose:
        n_prim = sum(status.get(u, missing_as_primitive) for u in nodes)
        print(f"status: {n_prim:,} primitive / {len(nodes) - n_prim:,} fully defined "
              f"({n_missing:,} without status, treated as {'primitive' if missing_as_primitive else 'fully defined'})")
    return {u: status.get(u, missing_as_primitive) for u in nodes}


def hop_weights(is_primitive, lam):
    """w(v) = lam if v is primitive, 1 if fully defined -- the factor paid by a hop *into* v."""
    return {u: (lam if prim else 1.0) for u, prim in is_primitive.items()}


def _check_complete(nodes, is_primitive):
    missing = [u for u in nodes if u not in is_primitive]
    if missing:
        raise KeyError(f"{len(missing):,} graph nodes have no status (e.g. {missing[:3]}); "
                       "build is_primitive with load_is_primitive(path, nodes=G.nodes())")


# --- semantic coverage ------------------------------------------------------


def build_coverage_transition_status(G, is_primitive, lam=1.0):
    """
    Same as tokenizer.build_coverage_transition, but each column v is scaled by w(v) instead of
    the whole matrix by lam:  A = diag(1 / d+(u)) @ Adj @ diag(w).  Then
    (A @ S[h-1])[u] = 1/d+(u) * sum_{(u,R,v)} w(v) S_{h-1}(v), the third case of the equation.

    Use it with the unchanged tokenizer.compute_semantic_coverage / semantic_coverage_score.
    lam = 1 makes w = 1 everywhere, so A is exactly the original (lambda-free) transition matrix:
    that is the matrix to use for the evaluation.
    """
    V = list(G.nodes())
    _check_complete(V, is_primitive)
    n = len(V)
    node_to_idx = {u: i for i, u in enumerate(V)}

    out_edges = build_out_edges(G)  # one entry per relationship occurrence (multi-edges kept)
    rows, cols = [], []
    for u in V:
        i = node_to_idx[u]
        for v in out_edges[u]:
            rows.append(i)
            cols.append(node_to_idx[v])

    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    out_degree = np.asarray(A.sum(axis=1)).flatten()
    inv_out_degree = np.divide(1.0, out_degree, out=np.zeros(n), where=out_degree > 0)
    w = np.array([lam if is_primitive[u] else 1.0 for u in V])
    A = (sp.diags(inv_out_degree) @ A @ sp.diags(w)).tocsr()
    return A, node_to_idx


# --- lazy greedy token selection --------------------------------------------


def _candidate_delta_core_status(t, node_to_idx, D, S, in_nodes, out_degree, T, M, w):
    """
    tokenizer._candidate_delta_core with the per-hop factor lam replaced by w(v): a change dv at
    v reaches each predecessor u as w(v) * dv / d+(u). Returns (gain, delta) like the original.
    """
    idx = node_to_idx
    delta = [None] * (D + 1)
    delta[0] = {t: 1.0 - S[0, idx[t]]}
    for h in range(1, D + 1):
        layer = {}
        for v, dv in delta[h - 1].items():
            push = w[v] * dv  # what v hands to every predecessor, before the 1/d+(u) average
            if push == 0.0:
                continue  # lam = 0 and v primitive: nothing propagates through this hop
            for u in in_nodes.get(v, ()):
                if u in T or u == t:
                    continue  # tokens (and t) are absorbing: set by the override, not propagated
                layer[u] = layer.get(u, 0.0) + push / out_degree[u]
        override = 1.0 - S[h, idx[t]]
        if override > 0:
            layer[t] = override
        delta[h] = layer
    gain = sum(delta[D].get(c, 0.0) for c in M) / len(M)
    return gain, delta


_WORKER_STATE = {}  # per-process globals for the parallel initial scan


def _init_worker_status(node_to_idx, D, S, in_nodes, out_degree, T, M, w):
    _WORKER_STATE.update(node_to_idx=node_to_idx, D=D, S=S, in_nodes=in_nodes,
                         out_degree=out_degree, T=T, M=M, w=w)


def _worker_seed_gain_status(t):
    st = _WORKER_STATE
    gain, _ = _candidate_delta_core_status(t, st["node_to_idx"], st["D"], st["S"], st["in_nodes"],
                                           st["out_degree"], st["T"], st["M"], st["w"])
    return t, gain


class StatusLazyGreedyTokenSelector(LazyGreedyTokenSelector):
    """
    LazyGreedyTokenSelector with the status-dependent decay: a hop into a primitive concept costs
    lam, a hop into a fully defined concept costs nothing. Same interface, same output
    (select() returns [(token, marginal_gain, cumulative_score), ...]).

    is_primitive: {node: bool} for every node of G (see load_is_primitive(..., nodes=G.nodes())).
    """

    def __init__(self, G, M, D, is_primitive, lam=1.0):
        super().__init__(G, M, D, lam=lam)
        _check_complete(self.V, is_primitive)
        self.w = {u: (lam if is_primitive[u] else 1.0) for u in self.V}

    def candidate_delta(self, t):
        return _candidate_delta_core_status(t, self.node_to_idx, self.D, self.S, self.in_nodes,
                                            self.out_degree, self.T, self.M, self.w)

    def select(self, k, candidates=None, verbose=True, n_jobs=1, progress_every=5000):
        """Same lazy-greedy loop as LazyGreedyTokenSelector.select; only the parallel seeding
        workers differ (they need w instead of lam)."""
        candidates = self.V if candidates is None else candidates
        candidates = [t for t in candidates if t not in self.T]
        n_candidates = len(candidates)
        heap = []

        if n_jobs == 1:
            for i, t in enumerate(candidates, 1):
                gain, _ = self.candidate_delta(t)
                heapq.heappush(heap, (-gain, t))
                if verbose and (i % progress_every == 0 or i == n_candidates):
                    print(f"  seeding candidates: {i}/{n_candidates}")
        else:
            n_workers = os.cpu_count() if n_jobs == -1 else n_jobs
            init_args = (self.node_to_idx, self.D, self.S, self.in_nodes, self.out_degree, self.T, self.M, self.w)
            with mp.Pool(n_workers, initializer=_init_worker_status, initargs=init_args) as pool:
                if verbose:
                    print(f"  seeding candidates: 0/{n_candidates}  ({n_workers} workers)")
                for i, (t, gain) in enumerate(pool.imap_unordered(_worker_seed_gain_status, candidates, chunksize=64), 1):
                    heapq.heappush(heap, (-gain, t))
                    if verbose and (i % progress_every == 0 or i == n_candidates):
                        print(f"  seeding candidates: {i}/{n_candidates}  ({n_workers} workers)")

        history = []
        while len(self.T) < k and heap:
            n_evals = 0
            while True:
                _, t = heapq.heappop(heap)
                gain, delta = self.candidate_delta(t)
                n_evals += 1
                best_other_bound = -heap[0][0] if heap else float("-inf")
                if gain >= best_other_bound:
                    self.commit(t, delta)
                    self.current_score += gain
                    history.append((t, gain, self.current_score))
                    if verbose:
                        i = len(self.T)
                        opt_upper_bound = min(1.0, self.current_score / GREEDY_RATIO)
                        print(f"[{i:>4}/{k}] token={t!r:<15} gain={gain:.5f}  score={self.current_score:.5f}  "
                              f"opt<={opt_upper_bound:.5f}  (re-evals={n_evals}, queue={len(heap)})")
                    break
                heapq.heappush(heap, (-gain, t))
        return history


# --- context trees with primitive-hop depth ---------------------------------


def build_out_adjacency_status(G, is_primitive):
    """tokenizer.build_out_adjacency (non-IS_A first, IS_A last), each edge extended with
    1 if its target is primitive else 0: {u: [(relation, v, prim_v), ...]}."""
    _check_complete(G.nodes(), is_primitive)
    return {u: [(r, v, int(is_primitive[v])) for r, v in edges] for u, edges in build_out_adjacency(G).items()}


def expand_status(u, q, d, p, adj, T, D):
    """
    tokenizer.expand with two counters: d = hops taken (checked against the horizon D, as before)
    and p = primitive hop targets so far (stored as the token's depth and as the subcontext's
    distance, which collapse_redundant uses to keep the most direct duplicate).
    """
    if u in T:
        q.add_token(u, p)
    else:
        edges = adj.get(u, ())
        if d == D or not edges:
            q.mark_uncovered()
        else:
            for r, v, prim_v in edges:
                pv = p + prim_v
                if r == "IS_A":
                    expand_status(v, q, d + 1, pv, adj, T, D)
                else:
                    child, is_new = q.open_subcontext(r, v, pv)
                    if is_new:
                        expand_status(v, child, d + 1, pv, adj, T, D)


def tokenize_all_rel_status(c, adj, T, D, id_to_label):
    root = Context(id_to_label, "ROOT", distance=0)
    expand_status(c, root, 0, 0, adj, T, D)
    return collapse_redundant(root)


def build_context_trees_status(M, adj, T, D, id_to_label=None):
    """eval.build_context_trees with primitive-hop depths. adj from build_out_adjacency_status.
    The returned trees work with every function of src/graph_tokenizer_gd_tree_dev/eval.py
    (distance_score, per_concept_metrics, ...)."""
    labels = id_to_label or {}
    T = set(T)
    return {c: tokenize_all_rel_status(c, adj, T, D, labels) for c in M}


def evaluate_status(M, adj, T, D, id_to_label=None):
    """eval.evaluate on the primitive-hop trees: same eight metrics, only distance_score changes."""
    trees = build_context_trees_status(M, adj, T, D, id_to_label)
    return {
        "conciseness": base_eval.conciseness(trees),
        "distance_score": base_eval.distance_score(trees, D),
        "uniqueness_entropy": base_eval.uniqueness_entropy(trees),
        "unk_rate": base_eval.unk_rate(trees),
        "uncovered_rate": base_eval.uncovered_rate(trees),
        "tree_complexity": base_eval.tree_complexity(trees),
        "exact_rate": base_eval.exact_rate(M, T),
        "unique_rate": base_eval.unique_rate(trees),
    }


# --- parallel evaluation (drop-in for tokenizer._init_coverage_worker / _worker_coverage_score) ---

_EVAL_STATE = {}


def _init_eval_worker(A, node_to_idx, adj, M, D, id_to_label=None):
    """A/node_to_idx: build_coverage_transition_status(G, is_primitive, lam=1.0);
    adj: build_out_adjacency_status(G, is_primitive)."""
    _EVAL_STATE.update(A=A, node_to_idx=node_to_idx, adj=adj, M=M, D=D, id_to_label=id_to_label)


def _worker_eval(task):
    label, T = task
    st = _EVAL_STATE
    S = compute_semantic_coverage(st["A"], st["node_to_idx"], T, st["D"])
    metrics = evaluate_status(st["M"], st["adj"], T, st["D"], st["id_to_label"])
    metrics["semantic_coverage"] = semantic_coverage_score(S, st["node_to_idx"], st["M"])
    return label, metrics
