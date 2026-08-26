# Summary: `src/graph_tokenizer_gd_tree_dev/`

*Written as prep for the paper draft in `paper/`. Based only on the current
source files (`tokenizer.py`, `eval.py`, `classical_selectors.py`,
`config.py`, `graph_fct.py`, `drilldown.py`, `graph_viz.py`,
`phenotype_clustering.py`, `utils.py`) and `Tokenizer_semantic_coverage
(3).pdf` — `IMPLEMENTATION.md` and `BASELINE_METHODS.md` were excluded per
the author's note that they describe an obsolete, pre-refactor version
(computed at `max_dist_candidate = 3`; current code uses `9`).*

## What it does
Selects a small set of SNOMED CT concept-graph nodes ("tokens") that can
represent a much larger target concept set via a short, semantics-aware walk
of the graph, then represents each target concept as a small labeled tree
(rather than a flat token bag) that preserves which relationships were used
to reach each token.

## Core method
1. **Candidate pool.** `graph_fct.get_combined_subgraphs_from_nodes` takes the
   union of the `D`-hop ego-graph (following outgoing edges) around every
   mapped concept `c ∈ M`. This is the only pool candidates are drawn from.
2. **Coverage score.** `tokenizer.build_coverage_transition` builds a
   row-normalized adjacency matrix `A` (with a `λ` distance-penalty folded
   in); `compute_semantic_coverage` propagates `S_h(u,T)` forward via
   `S[h] = A @ S[h-1]`, overriding token rows to `1` at every horizon
   (absorbing). `F_D(T)` is the mean of `S_D(c,T)` over `c ∈ M`.
3. **Selection.** `LazyGreedyTokenSelector` (lazy/CELF greedy) maximizes
   `F_D` up to `k` tokens. `_candidate_delta_core` computes one candidate's
   exact marginal gain by propagating a sparse delta backward through
   `in_nodes`, instead of recomputing `F_D` from scratch — this is what makes
   greedy selection tractable at candidate-pool scale.
4. **Tokenization.** `tokenize_all_rel` / `expand` recursively walks each
   mapped concept: `IS_A` edges stay in the current `Context`; every other
   relation opens a new subcontext keyed by `(relation, destination)`. A
   token found in `T` stops that branch; exhausting depth `D` or having no
   outgoing edges marks the branch `uncovered`.
5. **Evaluation.** `eval.evaluate` builds every mapped concept's context tree
   against a fixed `T` and computes 8 metrics from it: `conciseness`,
   `distance_score`, `uniqueness_entropy`, `unk_rate`, `uncovered_rate`,
   `tree_complexity`, `exact_rate`, `unique_rate` — plus `semantic_coverage`
   (always scored at `λ=1`, added separately by
   `tokenizer._worker_coverage_score`).
6. **Baselines.** `classical_selectors.HardGreedySetCoverSelector` is a hard
   (binary-coverage) greedy ablation of the same mechanism. Degree- and
   centrality-based baselines (`highest_degree`, `highest_degree_dist_1`,
   `most_children`, `pagerank`, `personalized_pagerank`,
   `closeness_centrality`, `eigenvector_centrality`, `k_random_all_samples`)
   are referenced by `config.CandidateLists` / `drilldown.load_all_candidates`
   but their construction lives in notebooks, not in `src/`.

## Novel vs. infrastructure
- **Novel / core contribution:** the `λ`-decayed semantic coverage recursion
  and its submodularity-backed lazy-greedy maximization
  (`tokenizer.py`, "semantic coverage" + "lazy greedy" sections); the
  context-tree tokenization with `IS_A`-transparent / non-`IS_A`-opens-subcontext
  semantics (`Context`, `expand`, `tokenize_all_rel`); the 8-metric evaluation
  suite decomposing "how good is this token set" into efficiency / fidelity /
  discriminative-power / semantic-content axes (`eval.py`); the hard-vs-soft
  coverage ablation (`classical_selectors.HardGreedySetCoverSelector`).
- **Supporting infrastructure (not paper-worthy on its own):** graph
  construction/pickling (`graph_fct.py`), path/parameter config
  (`config.py`), Streamlit drill-down helpers and pyvis rendering
  (`drilldown.py`, `graph_viz.py`), the multiprocessing worker-state
  plumbing in `tokenizer.py` / `phenotype_clustering.py` (real engineering,
  but not what a reviewer evaluates).

## Inputs & outputs
- In: `connectivity.parquet` (SNOMED relations), `mapped_concepts.parquet`
  (`M`), `concept_snomed_hug.parquet` (labels).
- Intermediate: `combined_subgraphs.gpickle` (the candidate-pool DAG),
  `id_to_label.pkl`.
- Out: per-`λ` full rankings (`greedy_tree_candidates/{λ}.parquet`),
  per-baseline full rankings (`baseline_candidates/*.parquet`), performance
  tables (`*_performance.parquet`) — truncated with `.head(k)` at evaluation
  time for every `(method, k)` combination.

## Key design decisions
- **One `D` for three roles** (candidate-pool ego-radius, context-tree depth,
  coverage horizon) via `TokenizerParam.max_dist_candidate` — simplifies the
  hyperparameter surface at the cost of coupling three conceptually distinct
  choices. *(Flagged to the author as worth an explicit callout in the
  paper — see `method.tex`.)*
- **`λ` folded into the transition matrix once, shared across all
  candidates** in a sweep, rather than recomputed per task — a performance
  decision, not a modeling one.
- **Evaluation always at `λ=1`** regardless of the `λ` a list was *selected*
  under, so every method is compared on the same undiscounted coverage
  notion. This is a modeling choice with real interpretive consequences
  (Section `results.tex`, "Effect of the distance penalty").
- **Non-`IS_A` relations are deduplicated by `(relation, destination)`, kept
  at shortest distance** (`Context.open_subcontext`) — a direct restatement
  by a parent concept collapses into the same subcontext as the child's own
  statement, rather than creating a duplicate.

## Assumptions & limitations
- Assumes the candidate/working subgraph is a DAG (well-defined finite hop
  distances); this is asserted, not checked, anywhere in the current
  `src/` code — verified in the old, now-obsolete analysis at `D=3`, not
  re-verified at the current `D=9`.
- `LazyGreedyTokenSelector`'s lazy-greedy correctness depends on
  submodularity of `F_D`, which depends on `λ ∈ (0, 1]`; `λ=0` is
  mathematically a different (degenerate) objective, not just a bad setting
  (see `results.tex`, derived directly from the recursion).
- `eigenvector_centrality`'s uniqueness guarantee (Perron–Frobenius) requires
  strong connectivity, which a DAG cannot have — the baseline still runs and
  converges, but its late-ranking is not theoretically meaningful.

## Dependencies
`tokenizer.py` and `eval.py` are mutually referenced (local import inside
`_worker_coverage_score` to avoid a circular import at module load).
`classical_selectors.py` depends only on the precomputed distance table, not
on `tokenizer.py` directly. `drilldown.py` / `graph_viz.py` depend on both,
plus `config.py`, and are consumed by `app_new_tokenizer.py` (not read in
this pass). `phenotype_clustering.py` is a separate downstream consumer
(clustering on tokenized features for an aneurysm/"IA" patient cohort) that
does not feed back into the tokenizer itself.

## Open questions for the author
1. Is the phenotype-discovery / IA-cohort case study (`phenotype_clustering.py`,
   `config.IACohort`/`IAFeatures`/`IAResults`) in scope for this paper, given
   it's live in `src/` but its driving notebooks are in the now-obsolete
   `old_notebooks/`? (Also flagged in `limitations.tex` / `discussion.tex`.)
2. Is the shared single-`D` design (candidate radius = tree depth = coverage
   horizon) intentional/load-bearing, or an implementation convenience worth
   relaxing in a future revision?
3. Current candidate-pool size and relation-type count under `D=9` — needed
   for `background.tex` / `results.tex`, not derivable from source alone
   without a fresh run of `1.graph_prepare.ipynb`.
4. Definition of "token reuse rate" (mentioned in `main.tex`'s Results
   outline as "need to be implemented") — not yet in `eval.py`.
