# graph_tokenizer_gd_tree_dev — implementation overview

This project builds and evaluates several strategies for picking a small set of
SNOMED concept-graph nodes ("**tokens**", `T`) that can represent a target set of
concepts ("**mapped concepts**", `M`) via a short graph walk. Each mapped concept
`c ∈ M` is expanded outward from itself into a small **context tree** (bounded
depth `D = 3`); the tree's leaves are whichever tokens from `T` were reached along
the way. A good `T` is one where every concept in `M` reaches nearby tokens
reliably and unambiguously.

Four notebooks form a linear pipeline, each one reading the previous step's
saved output (all paths are centralized in `config.py`):

```
1.graph_prepare.ipynb              -> combined_subgraphs.gpickle, id_to_label.pkl
2.greedy_tree_candidate_seledction -> greedy_tree_candidates/{lam}.parquet   (6 files, lam = 0.0..1.0)
3.baseline_candidate_selection     -> baseline_candidates/{highest_degree,highest_degree_dist_1,most_children,k_random_all_samples}.parquet
5.3.centrality_baselines           -> baseline_candidates/{pagerank,personalized_pagerank,closeness_centrality,eigenvector_centrality}.parquet
4.comparison_candidate_set         -> results/{greedy_tree,baseline}_performance.parquet + plots
```

All eight `baseline_candidates/*.parquet` files (from both notebook 3 and `5.3.centrality_baselines.ipynb`)
are documented in full — math, DAG-compatibility, calling notebook, output schema — in
**`BASELINE_METHODS.md`**.

Source lives in `src/graph_tokenizer_gd_tree_dev/`: `graph_fct.py` (graph
construction), `tokenizer.py` (context trees, semantic coverage, greedy
selection), `eval.py` (the 8 realized-representation metrics), `config.py` (all
paths and parameters).

---

## 1. `1.graph_prepare.ipynb` — build the working graph

- Reads `connectivity.parquet` (all SNOMED relations, `src.id -> dst.id` labeled
  by `relation`, e.g. `IS_A` for the hierarchy plus other attribute
  relationships) and `mapped_concepts.parquet` (`M`, the target concept set).
- Drops the SNOMED root concept (`138875005`, `config.TokenizerParam.exclude_cpt`).
- `graph_fct.build_relations_graph` builds a full `nx.MultiDiGraph` (one edge per
  relation occurrence — a `MultiDiGraph` because a `(src, dst)` pair can carry
  more than one relation type).
- `graph_fct.get_combined_subgraphs_from_nodes` then takes the **union of the
  3-hop ego-graph** (`nx.ego_graph(G, node, radius=3)`, following outgoing edges)
  around every node in `M`. This — not the full SNOMED graph — is what gets
  pickled as `combined_subgraphs.gpickle` and used everywhere downstream. So the
  candidate pool for tokens (`V`, all nodes in this combined subgraph) is
  exactly "every concept reachable within `D=3` hops of some mapped concept" —
  candidates outside anyone's 3-hop neighborhood are never even considered.
- `id_to_label.pkl`: a plain `{concept_id: label}` dict for display purposes.

## 2. `2.greedy_tree_candidate_seledction.ipynb` — the `greedy_tree_margin` lists

This produces the "smart" candidate lists via submodular optimization, not a
fixed heuristic like the baselines in step 3.

**The objective (`tokenizer.py`, "semantic coverage" section).** Define, for a
token set `T` and horizon `h = 0..D`:

```
S_0(u, T) = 1[u ∈ T]
S_h(u, T) = 1                                    if u ∈ T
          = lam · mean(S_{h-1}(v) for v ∈ Out(u)) otherwise      (Eq. 1)
```

i.e. a node's coverage score propagates backward from tokens, one hop per
horizon step, averaged over out-neighbors, decayed by `lam` per hop (`lam = 1`
= no decay; a token `j` hops away contributes `lam**j`). The overall objective is

```
F_D(T) = mean over c in M of S_D(c, T)                            (Eq. 2)
```

— "how well, on average, does a 3-hop walk from each mapped concept end up
reaching a token." `F_D` is normalized, monotone, and **submodular** in `T`
(Proposition 1 in the code's comments), which is what makes greedy selection
have a provable quality guarantee.

**`LazyGreedyTokenSelector`** (`tokenizer.py:329`) does lazy-greedy maximization
of `F_D`: it keeps one (possibly stale) marginal-gain bound per candidate in a
max-heap; submodularity guarantees a stale bound is always ≥ the candidate's
true current gain, so at each step it only has to re-verify the heap's current
top against the next-best stale bound, instead of recomputing every remaining
candidate from scratch (`CANDIDATE_DELTA`/`COMMIT`, §6.3 in the code's
docstrings). This gives the classic Nemhauser-Wolsey-Fisher guarantee:
`F_D(T_greedy) ≥ (1 - 1/e) · OPT` (`GREEDY_RATIO`).

**What the notebook actually does** (`cell 5`):
```python
lambdas_list = np.round(np.arange(0, 1.2, 0.2), 1)   # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
for lam in tqdm.tqdm(lambdas_list):
    selector = tokenizer.LazyGreedyTokenSelector(combined_subgraphs, mapped_ids, D=3, lam=lam)
    history = selector.select(k=len(mapped_ids), n_jobs=-1)
    ...
    df_history.write_parquet(f"{config.CandidateLists().path_greedy_tree}{lam}.parquet")
```
For each `lam` in `{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`, it runs greedy selection all
the way out to `k = len(mapped_ids)` (i.e. ranks **every** candidate, not just a
top-k), and saves the full ranking (`token, gain, cumulative_score`, in
selection order) to `{lam}.parquet`. **This is why the `greedy_tree_margin`
file_types are literally named `"0.0"` .. `"1.0"`** — those are the `lam` values
each ranking was selected under, not a separate "margin" concept. Notebook 4
later takes `df.head(k)` off these rankings to get "the top-k greedy tokens for
this `lam`."

Important: `lam` here only shapes *which order* candidates get picked in
during **selection**. It has no effect on how they're later **scored** — see
the "lam: selection vs. evaluation" note below.

## 3. `3.baseline_candidate_selection.ipynb` / `5.3.centrality_baselines.ipynb` — the `baseline` lists

Eight simple, non-learned ranking heuristics (four degree-style heuristics from notebook 3, four
`networkx` centrality measures added later in `5.3.centrality_baselines.ipynb`), each saved as a
**full ranking over all candidate nodes** (so notebook 4 can `.head(k)` them the same way as the
greedy lists) under `baseline_candidates/`. Full detail on each — the exact formula, whether it's
valid on a DAG (`combined_subgraphs` turns out to be one), which cell computes it, and its output
file's schema — is in **`BASELINE_METHODS.md`**, not duplicated here.

## 4. `4.comparison_candidate_set.ipynb` — scoring and comparison

**Task construction.** `all_candidates = {"greedy_tree_margin": {...6 lam files...}, "baseline": {...4 files...}}`.
For every `(category, file_type)` except `k_random_all_samples`, and every
`k ∈ Ks` (`np.arange(500, 12000, 500)`, 23 values), `T = df.head(k)["token"]` is
one task — 9 file_types × 23 k's = **207 tasks**. `k_random_all_samples` is
handled in its own cell: `Ks × rnd_iters` (23 × 20 = 460 tasks currently),
scored individually then meant to be grouped/averaged by `k` before comparing.

**Scoring a single `T`** (`tokenizer._worker_coverage_score`) computes, for
every mapped concept `c ∈ M`, a **context tree** (`tokenizer.tokenize_all_rel` /
`expand`):

- Start at `c`, depth `0`. If the current node is in `T`, record it as a found
  token at the current depth and stop that branch.
- Else, if at depth `D` or the node has no outgoing edges, mark this branch
  **uncovered** (a dead end — nothing in `T` was reached).
- Else, walk outgoing edges: `IS_A` edges stay **in the same context** (the
  hierarchy is "transparent" — a direct statement always wins over a parent's
  restatement of it); every other relation type **opens a new subcontext**
  keyed by `(relation, destination)`, deduplicated so revisiting the same
  `(relation, destination)` pair keeps only the shortest distance.

The result is a small tree per concept: `ctx.tokens` = tokens found in that
context, `ctx.uncovered` = whether some branch dead-ended, `ctx.subcontexts` =
nested contexts opened by non-`IS_A` relations.

From the 46,150 trees (one per `c ∈ M`), `eval.py` computes **9 metrics**:

| metric | meaning | direction |
|---|---|---|
| `conciseness` | mean # of token leaves per concept's tree | lower = more concise |
| `distance_score` | mean hop-distance tokens are found at (per concept, then averaged across concepts); dead-end branches count as `D+1` instead of being skipped; converted to `1 - mean_distance/(D+1)` | **higher = better**, bounded `(0, 1]` |
| `uniqueness_entropy` | normalized Shannon entropy of context-tree *signatures* across concepts — how often two different concepts end up with an identical tree shape | higher = more discriminative vocabulary |
| `unk_rate` | fraction of concepts with ≥1 uncovered branch anywhere in their tree | lower = better (out-of-vocabulary rate) |
| `uncovered_rate` | fraction of concepts with **zero** tokens anywhere in their tree — not partially represented, not represented at all (a stricter subset of `unk_rate`: a concept can have one uncovered branch while another branch still finds a token, so it counts toward `unk_rate` but not `uncovered_rate`; `uncovered_rate ≤ unk_rate` always) | lower = better |
| `tree_complexity` | mean # of contexts (root + subcontexts) per tree | lower = simpler |
| `exact_rate` | fraction of concepts that are themselves in `T` (0-hop representation) | higher = better |
| `unique_rate` | fraction of concepts whose context-tree signature (order/distance-invariant, same `signature()` used by `uniqueness_entropy`) isn't shared by any other concept — the per-concept decomposition of `uniqueness_entropy` (a concept with `unique_rate = True` has `redundancy_group_size == 1`, see `drilldown.py`) | higher = more discriminative vocabulary |
| `semantic_coverage` | `F_D(T)`, Eq. 2 above — same objective the greedy selector optimizes, but **always evaluated at `lam = 1.0`** here (see below) | higher = better, `(0, 1]` |

`distance_score`'s "count dead-ends at `D+1`, higher-is-better" convention was
adopted to mirror the sibling `graph_tokenizer_dev` project's
`__get_distance_n_score`, which does the same thing.

**lam: selection vs. evaluation.** `lam` is swept during *selection* (notebook
2, `{0.0 .. 1.0}`) to shape which tokens the greedy algorithm picks. During
*evaluation* (notebook 4), `tokenizer.build_coverage_transition(combined_subgraphs)`
is called with **no `lam` argument**, i.e. the default `lam = 1.0` (no distance
penalty). So every candidate list — regardless of what `lam` it was *selected*
under — is scored on the same, unpenalized `semantic_coverage`. The comparison
answers "how good is this token set at any distance," not "how good is it under
its own selection-time penalty."

**Performance.** The naive approach recomputes the graph's adjacency structure
from scratch on every one of the 207+460 tasks. Since none of that structure
actually depends on `T`, two pieces are now precomputed **once** per notebook
run and shared across all tasks via the `multiprocessing.Pool` initializer:
`tokenizer.build_coverage_transition(G)` (the semantic-coverage transition
matrix) and `tokenizer.build_out_adjacency(G)` (context-tree out-edges,
pre-sorted). Scoring itself is still the dominant cost — building 46,150
context trees per task — and is parallelized across `os.cpu_count()` worker
processes via `imap_unordered`.

**Output.** Results are written to `config.Results().perf_greedy_tree_path` /
`perf_baseline_path`, then plotted: one subplot per metric, x-axis = `k`, one
line per `file_type` (color graduated by `lam` for the greedy lists).

---

## Config reference (`config.py`)

- `BasicConfig`: source SNOMED data — `relation_path` (all relations),
  `concept_path` (concept table, for labels), `mapped_path` (`M`, the target
  concepts to be represented).
- `TokenizerParam`: `max_dist_candidate = D = 3` (the depth budget used
  *everywhere* — graph radius in step 1, context-tree depth in step 4, greedy
  selection horizon in step 2); `Ks = arange(500, 12000, 500)` (23 values);
  `rnd_iters = arange(0, 20, 1)` (random-baseline repeats per `k`);
  `exclude_cpt = ["138875005"]` (SNOMED root, excluded everywhere).
- `ProcessedGraph` / `CandidateLists` / `Results`: output paths for each
  pipeline stage, all rooted at `/mnt/z/graph_tokenizer_greedy_tree/`.

## Things to watch for when reading results

- **`distance_score`'s definition changed recently** (now higher-is-better,
  bounded, dead-ends counted; previously lower-is-better and dead-ends were
  silently skipped). Any already-persisted `*_performance.parquet` predating
  that change still holds the *old* values under the same column name — don't
  compare old and new runs directly; re-run the sweep to regenerate.
- `k_random_all_samples` needs a `group_by("k").mean()` over `iter` before
  it's comparable to the other (single-ranking) file_types.
- `greedy_tree_margin` file_types (`"0.0"` .. `"1.0"`) are selection-time `lam`
  values, not a margin threshold — and evaluation always uses `lam=1.0`
  regardless of which one a list was selected with.
- **`uncovered_rate`/`unique_rate` are new columns**, added after `unk_rate` and
  `exact_rate`/`uniqueness_entropy` respectively. Any `*_performance.parquet`
  persisted before that change simply doesn't have these columns at all (not
  stale values — missing entirely) — re-run notebook 4's sweep cells to
  regenerate before the comparison plots or the Streamlit app's leaderboard
  can show them.

## Analysis of tendency

Pulled from the current persisted results — `greedy_tree_performance.parquet`
(207 rows: 9 file_types × 23 `k`) and `baseline_performance.parquet` (460 rows:
23 `k` × 20 `iter` for `k_random_all_samples`, averaged over `iter` the same
way notebook 4 does before comparing it to the rest. Re-run this section's
numbers if the sweeps are re-run, since results will shift.

**1. Structured selection is dramatically more sample-efficient than random,
and stays that way across the whole tested range.** `highest_degree` reaches
`semantic_coverage = 0.70` with only `k = 500` tokens — a level
`k_random_all_samples` still hasn't reached with **23× the budget**
(`k = 11500 → 0.477`). This gap doesn't close within the tested range: any
degree/greedy-based method beats random at every `k` on every metric except
`conciseness`/`tree_complexity` (where random's lack of structure keeps its
trees artificially "busy" rather than genuinely better — see point 4).

**2. Within `greedy_tree_margin`, `lam` has two different effects depending on
the metric — monotonic on some, peaked on others.** At every `k` tested:
  - `semantic_coverage` and `unk_rate` improve **strictly monotonically** as
    `lam` goes `0.0 → 1.0` (e.g. at `k=11500`: `semantic_coverage` `0.27 →
    0.95 → 0.98 → 0.99 → 0.996 → 1.0`; `unk_rate` `0.75 → 0.32 → 0.22 → 0.17 →
    0.12 → 0.0`). Makes sense: evaluation always scores at `lam=1` (no
    distance penalty), so the closer the *selection*-time `lam` already is to
    1, the better selection and evaluation are aligned.
  - `distance_score` and `conciseness` instead **peak at an intermediate `lam`
    (≈ 0.6–0.8)** and are *worse* at `lam=1.0` than at `0.6`/`0.8` (at
    `k=11500`: `distance_score` peaks at `lam=0.8` (`0.729`) and *drops* at
    `lam=1.0` (`0.627`); `conciseness` similarly jumps from `~4.2–4.8` at
    `lam=0.2–0.8` up to `5.7` at `lam=1.0`). At `lam=1.0` the selector is
    purely maximizing raw reachability with no preference for *close* tokens,
    so it happily reaches more concepts, but at greater average hop-distance
    and with more redundant token matches per concept. `lam≈0.6–0.8` is the
    better trade-off if concise, nearby representations matter as much as raw
    coverage.

**3. `lam=0.0` is a pathological outlier, not just "the weak end of the
range" — at `lam=0`, selection stops optimizing coverage at all and
degenerates into maximizing `exact_rate`.** It scores at or below
`k_random_all_samples` on nearly every metric (e.g. `semantic_coverage` at
`k=500` is `0.012` — worse than random's `0.030`). This falls directly out of
the math: `build_coverage_transition` folds `lam` into the transition matrix
as `A = lam · A`, so at `lam=0` the matrix is all zero, and in
`compute_semantic_coverage`:

```python
S[0, T_idx] = 1.0
for h in 1..D:
    S[h] = A @ S[h-1]      # = 0 everywhere, since A is all-zero
    S[h, T_idx] = 1.0      # override re-applies regardless
```

`A` contributes nothing, so *every* horizon collapses to the same thing as
`h=0`: `S_h(u,T) = 1[u ∈ T]` for all `h`, not just `h=0` — no propagation ever
happens. That makes `F_D(T) = mean over c ∈ M of S_D(c,T)` reduce to `mean
over c ∈ M of 1[c ∈ T]`, which is **exactly `eval.py`'s `exact_rate(M, T)`**.
Selecting under `lam=0` therefore isn't "coverage with a bad setting," it's
optimizing a completely different objective: "how many mapped concepts did I
select as tokens myself." Any candidate that isn't itself in `M` contributes
*zero* marginal gain to that objective — the selector has no signal to rank
non-`M` candidates at all.

This also explains why `greedy_tree_margin/0.0`'s evaluated
`semantic_coverage` still creeps up with `k` (`0.012 → 0.273`) instead of
staying flat: once the selector exhausts `M` itself (all the positive-gain
picks), everything after that is a zero-gain tie among the remaining
candidates — essentially arbitrary filler, not coverage-optimized. But
*evaluation* is always done at `lam=1` (real distance-aware scoring), so even
that arbitrary filler still picks up some incidental multi-hop coverage —
just far less efficiently than a list actually selected to maximize it.

**4. `conciseness` peaks early (`k ≈ 1500–3500`) then *declines* for every
strategy, greedy and baseline alike** (`most_children` peaks later, `k=7500`,
but still turns over). This is counterintuitive — more candidate tokens
should mean more matches, not fewer — but it's a direct structural
consequence of `expand()`: a branch **stops the moment it hits a token**
(`if u in T: q.add_token(u, d)`, no further recursion from `u`). As `T` grows,
branches close to the root increasingly terminate in 1–2 hops on the first
token they meet, instead of continuing on to `D=3` and opening more
subcontexts that might have accumulated additional distinct token leaves.
Larger `T` therefore doesn't monotonically increase the token leaves counted
per concept — past a point it does the opposite.

**5. `most_children` (IS_A-hierarchy-only in-degree) is dominated by
`highest_degree` (any-relation in-degree) at every `k`** — `semantic_coverage`
`0.43` vs `0.70` at `k=500`, `0.95` vs `0.99` at `k=11500`; `distance_score`
`0.18` vs `0.32` at `k=500`, `0.55` vs `0.67` at `k=11500`. Non-hierarchy
relations carry real, exploitable coverage signal that a pure-`IS_A` heuristic
misses entirely.

**6. Differences among the *strong* strategies compress at large `k`, but
weak strategies never catch up.** Among greedy `lam ≥ 0.6` and both
`highest_degree` variants, `semantic_coverage` gaps shrink from roughly `0.11`
at `k=500` to `~0.01` at `k=11500` — everything reasonable eventually
approaches the `1.0` ceiling. `k_random_all_samples` and `lam=0.0`, by
contrast, are still far below that ceiling at `k=11500` — the gap doesn't
close within the tested range, it just becomes the dominant story once the
"reasonable" strategies have already saturated.
