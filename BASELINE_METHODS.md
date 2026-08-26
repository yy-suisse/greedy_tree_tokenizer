# Baseline candidate-selection methods

This project compares its own submodular-optimization method (`greedy_tree_margin`, see
`IMPLEMENTATION.md` §2) against a set of **baseline** candidate-selection heuristics — fixed,
non-learned rankings over the candidate pool `V` (all nodes of `combined_subgraphs`), each
computed **once** and truncated with `.head(k)` at eval time (`4.comparison_candidate_set.ipynb`),
the same way `greedy_tree_margin`'s rankings are.

**The graph these methods run on is a true DAG.** `combined_subgraphs` (75,634 nodes, 290,745
edges) has **zero** directed cycles — every one of its 75,634 strongly connected components is
trivial (size 1). The `IS_A`-only edge-subgraph (75,630 nodes, 117,433 edges, used by
`most_children`) is a DAG too. This is stated up front because it isn't just trivia — it directly
weakens the mathematical guarantees of one method below (eigenvector centrality) while leaving
the others unaffected; each method's section notes which case it is.

| file_type | one-line rule | notebook | output file (`config.CandidateLists()`) |
|---|---|---|---|
| `k_random_all_samples` | uniform random sample, redrawn per `(k, iter)` | `3.baseline_candidate_selection.ipynb` | `.k_random_all_samples` |
| `highest_degree` | most distinct predecessors within `D` hops, all relations | `3.baseline_candidate_selection.ipynb` | `.highest_degree` |
| `highest_degree_dist_1` | most distinct predecessors within 1 hop, all relations | `3.baseline_candidate_selection.ipynb` | `.highest_degree_dist_1` |
| `most_children` | most distinct predecessors within `D` hops, `IS_A` only | `3.baseline_candidate_selection.ipynb` | `.most_children` |
| `pagerank` | PageRank on `combined_subgraphs` | `5.3.centrality_baselines.ipynb` | `.pagerank` |
| `personalized_pagerank` | PageRank restarting on `M` | `5.3.centrality_baselines.ipynb` | `.personalized_pagerank` |
| `closeness_centrality` | closeness centrality (incoming distance) | `5.3.centrality_baselines.ipynb` | `.closeness_centrality` |
| `eigenvector_centrality` | eigenvector centrality | `5.3.centrality_baselines.ipynb` | `.eigenvector_centrality` |

All output files live under `config.CandidateLists().baseline_path`
(`/mnt/z/graph_tokenizer_greedy_tree/baseline_candidates/`) and are auto-discovered by
`4.comparison_candidate_set.ipynb` / `drilldown.load_all_candidates()` / `app_new_tokenizer.py`
via a glob over that directory — no registry to update when a new file_type is added.

---

## `k_random_all_samples`

**Math.** Not a ranking at all: for every `k ∈ Ks` (23 values) and every `iter ∈ rnd_iters` (20
repeats), an independent uniform sample of `k` nodes without replacement from `V`:

```
T_{k,iter} ~ Uniform( {S ⊆ V : |S| = k} )
```

460 independent draws total, all concatenated into one file (columns `k`/`iter` distinguish
them). Structurally different from every other baseline here: there's no single ranking to
`.head(k)` off of, so comparing it fairly means scoring each `(k, iter)` draw separately and
averaging metrics **over `iter`** per `k` (`4.comparison_candidate_set.ipynb` does this in its
own dedicated cell, not the shared `df.head(k)` task loop the other seven use).

**DAG?** N/A — doesn't look at edges at all, only node membership in `V`.

**Called from.** `3.baseline_candidate_selection.ipynb`, `# k random` cell — `nodes_df.sample(n=k)`
(polars, not `networkx`) once per `(k, iter)`.

**Output.** `baseline_candidates/k_random_all_samples.parquet` — columns `index` (row index within
this file, not a rank), `token`, `label`, `k`, `iter`.

---

## `highest_degree` / `highest_degree_dist_1`

**Math.** Rank each candidate `v ∈ V` by its number of *distinct* predecessors within `≤ Δ` hops,
over every relation type:

```
score(v) = |{u ∈ V : dist(u, v) ≤ Δ, u ≠ v}|
```

`highest_degree` uses `Δ = D = 3` (`config.TokenizerParam.max_dist_candidate`); `highest_degree_dist_1`
uses `Δ = 1` (literal direct in-edges, collapsing multi-edges to a single distinct predecessor).
Despite the name, this isn't graph in-degree in the strict sense once `Δ > 1` — it's a
bounded-radius reachability count, computed once via `nx.all_pairs_shortest_path_length` over the
whole graph and cached to `all_rel_distance_subgraph.parquet` (`config.ProcessedGraph().all_rel_distance_subgraph`,
columns `src_id`/`dst_id`/`distance`), then both baselines just filter that table at a different
`Δ` and `group_by("dst_id").agg(n_unique("src_id"))`.

**DAG?** Irrelevant either way — shortest-path-length via BFS is well-defined on any directed
graph, cyclic or not. (This graph happens to be acyclic, so "distance" here is unambiguously a
simple-path length, but the method doesn't depend on that.)

**Called from.** `3.baseline_candidate_selection.ipynb`, `# highest degree candidate selection`
and `# highest degree candidate selection dist <= 1` cells (the shared distance table is built in
`# highest degree and is_a child dataframe construction`).

**Output.**
- `baseline_candidates/highest_degree.parquet`
- `baseline_candidates/highest_degree_dist_1.parquet`

Both: columns `index`, `token`, `num_in_edges` (the `score(v)` above). No `label` column (unlike
the four newer baselines below).

---

## `most_children`

**Math.** Same formula as `highest_degree`, but restricted to the `IS_A`-only edge-subgraph:

```
score(v) = |{u ∈ V_IS_A : dist_IS_A(u, v) ≤ D, u ≠ v}|
```

i.e. "how many descendant concepts does `v` have in the hierarchy within `D` hops" — the
distances come from a separate `nx.all_pairs_shortest_path_length` pass over the `IS_A`-only
edge-subgraph, cached to `is_a_distance_subgraph.parquet`
(`config.ProcessedGraph().is_a_distance_subgraph`).

**DAG?** Same as above — BFS distance doesn't care about acyclicity — but here it's worth noting
the `IS_A`-only subgraph is *itself* confirmed a DAG (it's the concept hierarchy; a cycle there
would mean a concept is its own ancestor, which would be a genuine ontology defect, not present in
this data).

**Called from.** `3.baseline_candidate_selection.ipynb`, `# most children via is_a` cell.

**Output.** `baseline_candidates/most_children.parquet` — columns `index`, `token`, `num_in_edges`.
No `label` column.

**Known result (see `IMPLEMENTATION.md`, "Analysis of tendency" §5):** dominated by
`highest_degree` at every `k` — non-hierarchy relations carry real coverage signal a pure-`IS_A`
heuristic misses.

---

## `pagerank`

**Math.** Standard PageRank — the stationary distribution of a random walk that, at each step,
follows a random outgoing edge with probability `α = 0.85` (networkx default) or teleports to a
uniformly random node with probability `1 - α`:

```
PR(v) = (1 - α)/N + α · Σ_{u → v} PR(u) / outdeg(u)
```

solved by power iteration. "Dangling" nodes (no outgoing edges — this DAG's sinks) have their walk
weight redistributed by the personalization vector (uniform here) rather than lost, which is what
keeps the stationary distribution well-defined at all.

**DAG?** **Yes, by design.** The `α`-teleportation term is exactly what PageRank needs to stay
well-defined on a graph that isn't strongly connected: it makes the underlying Markov chain
irreducible and aperiodic regardless of the original graph's structure, so a unique stationary
distribution exists whether or not `combined_subgraphs` has cycles. This is the standard "why not
just use PageRank" answer this baseline exists to preempt.

**Called from.** `5.3.centrality_baselines.ipynb`, `## PageRank` cell — `nx.pagerank(combined_subgraphs)`,
run directly on the `MultiDiGraph` (a `(u, v)` pair connected by two relation types gets double
random-walk weight, treated here as a reasonable reading of "more strongly connected").

**Output.** `baseline_candidates/pagerank.parquet` — columns `index`, `token`, `pagerank`, `label`.

---

## `personalized_pagerank`

**Math.** Identical recursion, but the restart distribution is biased toward `M` instead of
uniform:

```
PR(v) = (1 - α) · p(v) + α · Σ_{u → v} PR(u) / outdeg(u)      where  p(v) = 1/|M| if v ∈ M else 0
```

(`personalization={m: 1 for m in mapped_ids}`; networkx normalizes internally.) This is the one
centrality baseline that's `M`-aware, like the paper's own `greedy_tree_margin` objective, rather
than purely structural — the fairest of the four to compare against.

**DAG?** Same as plain PageRank — yes, unaffected by acyclicity for the same teleportation reason.

**Called from.** `5.3.centrality_baselines.ipynb`, `## Personalized PageRank seeded on M` cell.

**Output.** `baseline_candidates/personalized_pagerank.parquet` — columns `index`, `token`,
`personalized_pagerank`, `label`.

---

## `closeness_centrality`

**Math.** Reciprocal of the average shortest-path distance *to* `v` from every node that can
reach it (`wf_improved=True`, networkx default, scales by the fraction of the graph actually
reachable so disconnected components don't get an unfairly inflated score):

```
C(v) = (n_v - 1)/(N - 1) · (n_v - 1) / Σ_{u : dist(u,v) < ∞} dist(u, v)
```

where `n_v` is the number of nodes that can reach `v`. For directed graphs, networkx's default
uses **incoming** distance (distance *to* `v`, not from it) — exactly the same "how reachable is
this candidate from the rest of the graph" notion `highest_degree` already ranks by via a bounded
`Δ`-hop count, just as an unbounded, continuous shortest-path measure instead. Conceptually the
closest existing baseline to the `distance_score` metric.

**DAG?** Works correctly, no special caveat — closeness doesn't need strong connectivity for a
unique answer, only reachability, which BFS gives regardless of cycles. On this DAG specifically:
nodes with no incoming edges (`n_v = 1`, nothing reaches them) get `C(v) = 0` by convention —
an honest reflection of being a "source" concept, not a degenerate case needing special handling.

**Called from.** `5.3.centrality_baselines.ipynb`, shared cell with eigenvector centrality below —
`nx.closeness_centrality(G_simple)`. Runs fine directly on the `MultiDiGraph` too (unlike
eigenvector centrality, it doesn't require simplification), but the notebook converts once and
reuses `G_simple` for both, for consistency.

**Output.** `baseline_candidates/closeness_centrality.parquet` — columns `index`, `token`,
`closeness_centrality`, `label`.

---

## `eigenvector_centrality`

**Math.** `v`'s centrality is proportional to the sum of its predecessors' centrality:

```
λ x_v = Σ_{u → v} x_u
```

for the eigenvalue `λ` of largest modulus, solved by power iteration on the adjacency matrix.
Requires a simple graph — `NetworkXNotImplemented` is raised directly on a `MultiDiGraph`.

**DAG? Runs, but its core uniqueness guarantee doesn't hold here — and the result shows it.**
Per networkx's own docs, the Perron–Frobenius theorem guarantees a *unique*, all-positive
eigenvector **only if `G` is strongly connected**. `combined_subgraphs` is about as far from
strongly connected as a graph this size can be (every SCC is trivial), so that guarantee simply
doesn't apply: a DAG's adjacency matrix, in topological order, is strictly triangular and hence
nilpotent — every eigenvalue is 0, and "the" eigenvector for the largest-modulus eigenvalue isn't
uniquely defined. In practice, power iteration on this graph still converges to *a* left
eigenvector (concentrated on nodes reached by many/long predecessor chains), and empirically it's
stable — a random vs. the default all-ones starting vector produce the exact same top-20 ranking
— but **73,604 of 75,634 nodes (97.3%) land at essentially zero** (`< 1e-9`). Only ~2,030 nodes
carry any real signal; past that point in a `k`-sweep, this baseline's ranking is an arbitrary
tie-break among near-zero-score nodes, not a meaningful ordering. Worth flagging explicitly if
this baseline's results are reported at the larger end of `Ks` (`up to 11,500`).

**Called from.** `5.3.centrality_baselines.ipynb`, shared cell with closeness centrality —
`nx.eigenvector_centrality(G_simple, max_iter=500)` (raised past the `max_iter=100` default as a
margin of safety; empirically it converges within 100 anyway).

**Output.** `baseline_candidates/eigenvector_centrality.parquet` — columns `index`, `token`,
`eigenvector_centrality`, `label`.
