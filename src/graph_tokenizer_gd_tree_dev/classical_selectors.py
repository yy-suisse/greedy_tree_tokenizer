import heapq

import polars as pl


def load_within_d_distances(distance_df: pl.DataFrame, D: int) -> pl.DataFrame:
    """
    `all_rel_distance_subgraph.parquet` (src_id, dst_id, distance) filtered to the same
    `distance <= D` horizon `highest_degree`/`most_children` already use, self-loops dropped
    (a node "covering" itself isn't a meaningful signal for `HardGreedySetCoverSelector`).
    """
    return distance_df.filter((pl.col("distance") <= D) & (pl.col("src_id") != pl.col("dst_id")))


class HardGreedySetCoverSelector:
    """
    Discrete (hard) greedy maximum-coverage selection (TODO.md item 5, second bullet): the
    same lazy-greedy *mechanism* as `tokenizer.LazyGreedyTokenSelector`, but Eq. 1's soft,
    lam-decayed, multi-hop-averaged `S_h(u, T)` is replaced by a **binary** "is concept `c`
    within `D` hops of *some* token in `T`" indicator. That indicator is exactly what the
    precomputed distance table already encodes -- no BFS/graph-walk machinery needs
    reimplementing here: concept `c` is covered by `T` iff some `t` in `T` appears as a
    `(src_id=c, dst_id=t, distance<=D)` row.

    This is the textbook NP-hard maximum-coverage problem's lazy/CELF greedy, with the same
    `(1 - 1/e)` approximation guarantee (Nemhauser, Wolsey & Fisher 1978) that
    `LazyGreedyTokenSelector` inherits for the *soft* objective -- so comparing the two
    rankings doubles as a literature baseline (real "hard set-cover" algorithm, not a
    strawman) and an ablation of what the soft Eq. 1 formulation buys over classic hard
    greedy, per TODO.md's framing.
    """

    def __init__(self, distance_df: pl.DataFrame, M, D, candidate_pool=None):
        """
        distance_df: the raw (src_id, dst_id, distance) table from
            `config.ProcessedGraph().all_rel_distance_subgraph` (notebook 3) -- shortest-path
            hop distance from src_id to dst_id following outgoing edges, over every node in
            the candidate pool (not just M).
        M: mapped concepts to cover -- the universe the greedy loop maximizes coverage of.
        D: hop budget (`max_dist_candidate`).
        candidate_pool: every node eligible to be picked as a token. Defaults to every
            `dst_id` within `D` hops of *any* node (the same universe `highest_degree` ranks
            over). On this project's graph every such candidate happens to also be within
            `D` hops of *some* mapped concept (it's built as the union of `M`'s own D-hop ego
            graphs), so in practice every candidate covers >=1 concept -- but the padding
            below stays as a defensive fallback in case that invariant ever changes (e.g. a
            future graph-construction change, or a smaller custom `M`).
        """
        self.M = set(M)
        self.D = D
        within_d = load_within_d_distances(distance_df, D)

        covered_df = within_d.filter(pl.col("src_id").is_in(list(self.M)))
        self.covers: dict[str, set] = {row["dst_id"]: set(row["covered"]) for row in covered_df.group_by("dst_id").agg(pl.col("src_id").alias("covered")).iter_rows(named=True)}

        if candidate_pool is None:
            candidate_pool = within_d["dst_id"].unique().to_list()
        # zero-coverage tokens (see docstring above): never enter the greedy loop, only pad
        # the tail of the ranking so `.head(k)` never comes up short of `k` for any `k` up to
        # `len(candidate_pool)`.
        self._leftover = sorted(set(candidate_pool) - self.covers.keys())

    def select(self, k: int, verbose: bool = True, progress_every: int = 5000) -> list:
        """
        Lazy/CELF greedy max-coverage, mirroring `LazyGreedyTokenSelector.select`'s structure:
        pop the heap's current top, recompute its *true* marginal gain (newly-covered concept
        count) against what's covered so far, and only commit it once that true gain is >=
        every other candidate's stale (submodularity-guaranteed-optimistic) bound -- so most
        iterations need just one recompute instead of rescanning every remaining candidate.

        Returns history: a list of `(token, gain, cumulative_score)`, in pick order.
          - gain: # of *previously uncovered* mapped concepts this token newly covers (an
            integer, unlike the soft selector's fractional Eq. 8 gain).
          - cumulative_score: `|covered so far| / |M|` -- directly comparable in scale to
            `semantic_coverage`/`F_D(T)`, both in `(0, 1]`.
        Once every concept in `M` is covered, remaining picks (if `k` is larger than the
        number of useful candidates) have gain 0 and are appended in a fixed, deterministic
        order (heap tie-break, then leftover padding) rather than an arbitrary one.
        """
        heap = [(-len(covered), tok) for tok, covered in self.covers.items()]
        heapq.heapify(heap)
        covered: set = set()
        history: list = []
        n_targets = len(self.M)

        while heap and len(history) < k:
            n_evals = 0
            while True:
                _, t = heapq.heappop(heap)
                gain  =len(self.covers[t] - covered)
                n_evals += 1
                best_other_bound = -heap[0][0] if heap else -1
                if gain >= best_other_bound:  # true maximizer: nothing else in the queue can beat it
                    covered |= self.covers[t]
                    score = len(covered) / n_targets if n_targets else 0.0
                    history.append((t, gain, score))
                    if verbose and (len(history) % progress_every == 0 or len(history) == k):
                        print(f"[{len(history):>6}/{k}] token={t!r:<15} gain={gain:>4}  " f"covered={len(covered)}/{n_targets} ({score:.3%})  (re-evals={n_evals})")
                    break
                heapq.heappush(heap, (-gain, t))  # stale bound: re-queue with the freshly computed gain

        score = len(covered) / n_targets if n_targets else 0.0
        for t in self._leftover:
            if len(history) >= k:
                break
            history.append((t, 0, score))
        return history
