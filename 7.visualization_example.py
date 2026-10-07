"""
7. Visualization example: tokenize one mapped concept and inspect the result.

Choose a distance decay lambda (one greedy candidate list in
config.CandidateLists().path_greedy_tree), a token budget k (the top-k tokens of
that list) and a mapped concept (fuzzy search on label or id), then click
"Tokenize". The app shows:

  1. the subgraph that Expand (tokenizer.expand) explored for the concept,
     with relation types on the edges;
  2. the resulting context tree;
  3. the per-concept scores: semantic coverage, distance score (with its
     breakdown), tokens per concept, and how they compare with all mapped concepts;
  4. the other mapped concepts that share the same signature.

The tracing, drawing and search helpers live in src/graph_tokenizer_gd_tree_dev/concept_viz.py
(shared with 8.compare_visualization_our_baseline.py).

Run from the repository root:

    streamlit run 7.visualization_example.py

Optional, for better fuzzy search:  pip install rapidfuzz
"""

import glob
import os

import polars as pl
import streamlit as st
import streamlit.components.v1 as components

import src.graph_tokenizer_gd_tree_dev.config as config
from src.graph_tokenizer_gd_tree_dev import drilldown
from src.graph_tokenizer_gd_tree_dev.concept_viz import (
    build_graph_html,
    build_search_index,
    context_entries,
    count_dead_ends,
    dataframe_width_kwargs,
    fuzzy_rank,
    legend_html,
    nested_expression,
    percentile,
    tokenize_traced,
)
from src.graph_tokenizer_gd_tree_dev.eval import _iter_contexts, build_context_trees
from src.graph_tokenizer_gd_tree_dev.tokenizer import (
    build_coverage_transition,
    build_out_adjacency,
    compute_semantic_coverage,
    signature,
    tokenize_all_rel,
)

st.set_page_config(page_title="Context tree explorer", layout="wide")

D = config.TokenizerParam().max_dist_candidate
GREEDY_DIR = config.CandidateLists().path_greedy_tree
SEARCH_LIMIT = 50
DF_KW = dataframe_width_kwargs(st)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading the candidate subgraph and the mapped concepts...")
def load_static():
    G = drilldown.load_graph()
    id_to_label = drilldown.load_id_to_label()
    df_mapped = (
        drilldown.load_mapped_concepts()
        .unique("id", keep="first", maintain_order=True)
        .with_columns(pl.col("label").fill_null(pl.col("id")))
    )
    mapped_ids = df_mapped["id"].to_list()
    adj = build_out_adjacency(G)
    # Semantic coverage is always evaluated at lambda = 1 (paper, Sec. 3.6).
    A1, node_to_idx = build_coverage_transition(G, lam=1.0)
    return G, id_to_label, df_mapped, mapped_ids, adj, A1, node_to_idx


def label_of(cid):
    id_to_label = load_static()[1]
    return id_to_label.get(cid) or mapped_label_lookup().get(cid) or cid


@st.cache_resource
def mapped_label_lookup():
    df_mapped = load_static()[2]
    return dict(zip(df_mapped["id"].to_list(), df_mapped["label"].to_list()))


@st.cache_data
def list_lambda_files():
    """{lambda name: parquet path}, sorted by lambda; file names are the lambda values."""
    out = {}
    for path in glob.glob(os.path.join(GREEDY_DIR, "*.parquet")):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            float(name)
        except ValueError:
            continue
        out[name] = path
    return dict(sorted(out.items(), key=lambda kv: float(kv[0])))


@st.cache_data
def load_candidate_list(path):
    return pl.read_parquet(path).select("token", "gain", "cumulative_score")


@st.cache_data
def max_list_length():
    return max(load_candidate_list(path).height for path in list_lambda_files().values())


@st.cache_data
def top_k_tokens(lam_name, k):
    path = list_lambda_files()[lam_name]
    return load_candidate_list(path).head(k)["token"].to_list()


@st.cache_data
def selection_rank(lam_name):
    """token -> 1-based position in the greedy order of this lambda's list."""
    path = list_lambda_files()[lam_name]
    tokens = load_candidate_list(path)["token"].to_list()
    return {t: i + 1 for i, t in enumerate(tokens)}


@st.cache_resource(max_entries=4, show_spinner=False)
def tokenize_population(lam_name, k):
    """Context tree of every mapped concept for (lambda, k), reduced to a per-concept table.

    Needed for the signature groups (N_sigma) and for the population means/percentiles.
    The trees themselves are dropped once the table is built, to keep memory low.
    """
    _G, id_to_label, _df, mapped_ids, adj, A1, node_to_idx = load_static()
    T = top_k_tokens(lam_name, k)
    trees = build_context_trees(mapped_ids, adj, T, D, id_to_label)
    S = compute_semantic_coverage(A1, node_to_idx, T, D)
    df = drilldown.get_all_concept_scores(trees, S, node_to_idx, mapped_ids, D)
    has_uncovered = [any(ctx.uncovered for ctx in _iter_contexts(trees[c])) for c in df["mapped_id"].to_list()]
    n_uncovered = [sum(1 for ctx in _iter_contexts(trees[c]) if ctx.uncovered) for c in df["mapped_id"].to_list()]
    return df.with_columns(
        pl.Series("has_uncovered_branch", has_uncovered),
        pl.Series("n_uncovered", n_uncovered),
    )


@st.cache_resource(max_entries=3, show_spinner=False)
def objective_row(lam_name, k):
    """S_D(u, T) under the selection lambda (the greedy objective), for every node."""
    G, *_rest, node_to_idx = load_static()
    A_lam, _ = build_coverage_transition(G, lam=float(lam_name))
    return compute_semantic_coverage(A_lam, node_to_idx, top_k_tokens(lam_name, k), D)[-1]


@st.cache_resource(max_entries=32, show_spinner=False)
def tokenize_one(lam_name, k, concept):
    _G, id_to_label, _df, _m, adj, _A1, _idx = load_static()
    T = set(top_k_tokens(lam_name, k))
    tree, trace = tokenize_traced(concept, adj, T, D, id_to_label)
    # Safety check: the traced copy must give exactly what the tokenizer gives.
    reference = tokenize_all_rel(concept, adj, T, D, id_to_label)
    consistent = signature(reference) == signature(tree)
    return tree, trace, consistent


@st.cache_resource
def search_index():
    df_mapped = load_static()[2]
    return build_search_index(df_mapped["id"].to_list(), df_mapped["label"].to_list())


@st.cache_data(max_entries=200, show_spinner=False)
def fuzzy_search(query, limit=SEARCH_LIMIT):
    return fuzzy_rank(query, search_index(), limit)


# ---------------------------------------------------------------------------
# Sidebar: parameters
# ---------------------------------------------------------------------------

lam_files = list_lambda_files()
if not lam_files:
    st.error(f"No candidate lists found in `{GREEDY_DIR}`.")
    st.stop()

_G, _id_to_label, _df_mapped, mapped_ids, *_ = load_static()
search_ids = search_index()[0]

with st.sidebar:
    st.header("Tokenizer")
    lam_names = list(lam_files)
    lam_name = st.selectbox(
        "Distance decay λ",
        lam_names,
        index=lam_names.index("0.9") if "0.9" in lam_names else len(lam_names) - 1,
        help=f"One greedy candidate list per λ, read from {GREEDY_DIR}",
        key="lam",
    )
    n_available = load_candidate_list(lam_files[lam_name]).height
    k = int(
        st.number_input(
            "Token budget k",
            min_value=1,
            max_value=max_list_length(),
            value=min(5000, n_available),
            step=500,
            key="k",
            help="The first k tokens of the greedy list of the chosen λ.",
        )
    )
    if k > n_available:
        st.caption(f"This list holds only {n_available:,} tokens; k is capped at {n_available:,}.")
        k = n_available

    st.header("Mapped concept")
    query = st.text_input("Search (label or SNOMED CT id)", placeholder="e.g. fracture of upper limb")
    hits = fuzzy_search(query) if query else []
    concept = None
    if query and not hits:
        st.warning("No mapped concept matches this search.")
    elif hits:
        options = [search_ids[i] for i in hits]
        concept = st.selectbox(
            "Best matches",
            options,
            format_func=lambda cid: f"{label_of(cid)} ({cid})",
        )

    if st.button("Tokenize", type="primary", disabled=concept is None):
        st.session_state["run"] = {"lam": lam_name, "k": k, "concept": concept}

    st.header("Graph view")
    view = st.radio("Show", ["Whole explored subgraph", "Only paths that reach a token"])
    layout_name = st.selectbox(
        "Layout",
        ["Hierarchical, general concepts on top", "Hierarchical, left to right", "Force-directed"],
    )
    show_isa_labels = st.checkbox("Label IS_A edges", value=True)
    show_ids = st.checkbox("Show concept ids in nodes", value=False)
    graph_height = st.slider("Graph height (px)", 500, 1600, 800, 100)

layout = {
    "Hierarchical, general concepts on top": "bottom_up",
    "Hierarchical, left to right": "left_right",
    "Force-directed": "force",
}[layout_name]

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

run = st.session_state.get("run")
if run is None:
    st.title("Context tree explorer")
    st.info("Choose λ, k and a mapped concept in the sidebar, then click **Tokenize**.")
    st.stop()

lam, kk, c = run["lam"], run["k"], run["concept"]
if (lam, kk) != (lam_name, k) or (concept is not None and concept != c):
    st.caption("The sidebar settings have changed. Click **Tokenize** to update the results.")

with st.spinner(
    f"Tokenizing all {len(mapped_ids):,} mapped concepts with λ = {lam}, k = {kk:,} "
    "(once per λ and k; needed for the signature groups and the comparison with all concepts)..."
):
    pop = tokenize_population(lam, kk)
tree, trace, consistent = tokenize_one(lam, kk, c)
T = set(top_k_tokens(lam, kk))
row = pop.filter(pl.col("mapped_id") == c)

entries = context_entries(tree, label_of, D)
n_m = sum(1 for e in entries if e["kind"] == "token")
u_m = sum(1 for e in entries if e["kind"] == "uncovered")
q_m = sum(1 for _ in _iter_contexts(tree))
mean_depth = sum(e["depth"] for e in entries) / len(entries) if entries else D + 1
dist_score = 1 - mean_depth / (D + 1)
coverage = row["frac_sem_cov"][0] if row.height else None
group_size = int(row["redundancy_group_size"][0]) if row.height else None

st.title(label_of(c))
st.caption(f"`{c}` · λ = {lam} · k = {kk:,} · D = {D} · {len(mapped_ids):,} mapped concepts")

if not consistent:
    st.warning("The traced expansion differs from tokenizer.tokenize_all_rel. Check that expand() was not changed.")
if row.height and int(row["num_tokens"][0]) != n_m:
    st.warning("The token count differs from the population table; the results may be inconsistent.")

if c in T:
    st.success("This concept is itself a token: an exact match at depth 0.")
elif n_m == 0:
    st.error(f"Unknown: no token is reached within D = {D} hops, so the concept cannot be represented.")
elif u_m:
    st.warning(f"Represented by {n_m} token(s), but {u_m} context(s) end without a token (uncovered branch).")
else:
    st.info(f"Represented by {n_m} token(s); every branch reaches a token.")

# 1. Graph ---------------------------------------------------------------------
st.header("1 · Explored subgraph")
st.caption(
    "Everything Expand visited from the mapped concept. A branch stops at a token, at depth D, or at a "
    "concept without outgoing relationships. Hover a node or an edge for details; nodes can be dragged."
)
st.markdown(legend_html(), unsafe_allow_html=True)

html, n_shown = build_graph_html(
    c, trace, T, label_of, D, pruned=view.startswith("Only"), layout=layout,
    show_isa_labels=show_isa_labels, show_ids=show_ids, height=graph_height,
)
n_dead = count_dead_ends(trace, c)
n_tokens_reached = sum(1 for n in trace.roles if "token" in trace.roles[n])
st.caption(
    f"{n_shown:,} concepts shown of {len(trace.min_depth):,} visited · {n_tokens_reached} distinct tokens · "
    f"{n_dead} dead ends · {len(trace.edges):,} relationships followed"
)
if n_shown > 600:
    st.warning("Large graph: rendering may be slow. Try **Only paths that reach a token**.")
if hasattr(st, "iframe"):  # newer Streamlit; components.html is deprecated there
    st.iframe(html, height=graph_height + 30)
else:
    components.html(html, height=graph_height + 30, scrolling=False)
st.download_button(
    "Download graph (HTML)",
    html,
    file_name=f"explored_subgraph_{c}_lam{lam}_k{kk}.html",
    mime="text/html",
)

# 2. Context tree ----------------------------------------------------------------
st.header("2 · Context tree")
st.caption(
    "IS_A hops stay in the same context; every other relationship opens a subcontext. "
    "`•` token with its depth d · `∅` uncovered context."
)
st.code(drilldown.render_context_tree(tree, load_static()[1]), language=None)
with st.expander("Nested notation (as in the paper)"):
    st.code("(" + nested_expression(tree, label_of) + ")", language=None)
    st.code("(" + nested_expression(tree, str) + ")", language=None)

ranks = selection_rank(lam)
token_rows = [
    {"token": e["entry"], "id": e["id"], "context": e["context"], "depth": e["depth"], "selection rank": ranks.get(e["id"])}
    for e in entries if e["kind"] == "token"
]
if token_rows:
    st.markdown("**Tokens**")
    st.dataframe(pl.DataFrame(token_rows), **DF_KW, hide_index=True)
    st.caption("Selection rank: position of the token in the greedy order of this λ (1 = selected first).")

# 3. Scores ---------------------------------------------------------------------
st.header("3 · Scores of this concept")


def compare(col, value, digits, lower_better=False):
    if value is None or not pop.height:
        return None, None
    mean = pop[col].drop_nulls().mean()
    delta = f"{value - mean:+.{digits}f} vs mean"
    pct = percentile(pop[col], value)
    return delta, (f"Mean over all mapped concepts: {mean:.{digits}f}. P{pct:.0f}: "
                   f"{pct:.0f}% of mapped concepts have a value ≤ this one.")


m1, m2, m3, m4 = st.columns(4)
delta, help_text = compare("frac_sem_cov", coverage, 3)
m1.metric(
    "Semantic coverage S_D(m, T)",
    f"{coverage:.3f}" if coverage is not None else "—",
    delta=delta,
    help=("Evaluated at λ = 1, whatever λ was used for selection. " + (help_text or ""))
    if coverage is not None else "This concept is not in the candidate subgraph.",
)
delta, help_text = compare("distance_score", dist_score, 3)
m2.metric("Distance score", f"{dist_score:.3f}", delta=delta, help=help_text)
delta, help_text = compare("num_tokens", n_m, 2)
m3.metric("Tokens n_m", n_m, delta=delta, delta_color="inverse", help=help_text)
m4.metric(
    "Signature group N_σ",
    group_size if group_size is not None else "—",
    help="Number of mapped concepts with this exact signature, this one included (1 = unique).",
)

st.markdown("**Distance score breakdown**")
st.caption(f"One row per token with its depth, plus one row per uncovered context, counted at depth D + 1 = {D + 1}.")
st.dataframe(
    pl.DataFrame([{k_: e[k_] for k_ in ("entry", "kind", "context", "depth")} for e in entries])
    if entries else pl.DataFrame(schema={"entry": pl.Utf8, "kind": pl.Utf8, "context": pl.Utf8, "depth": pl.Int64}),
    **DF_KW,
    hide_index=True,
)
sum_depth = sum(e["depth"] for e in entries)
st.latex(
    rf"\bar\ell_m = \frac{{1}}{{n_m + |U_m|}}\sum_{{p \in P_m}} \ell_p"
    rf" = \frac{{{sum_depth}}}{{{n_m} + {u_m}}} = {mean_depth:.3f}"
    rf"\qquad 1 - \frac{{\bar\ell_m}}{{D + 1}} = 1 - \frac{{{mean_depth:.3f}}}{{{D + 1}}} = {dist_score:.3f}"
)

st.markdown("**This concept against all mapped concepts**")
summary = [
    ("Semantic coverage", coverage, "frac_sem_cov", "↑"),
    ("Distance score", dist_score, "distance_score", "↑"),
    ("Tokens per concept (n_m)", n_m, "num_tokens", "↓"),
    ("Contexts |Q_m|", q_m, "tree_complexity", "↓"),
    ("Uncovered contexts |U_m|", u_m, "n_uncovered", "↓"),
    ("Signature group N_σ", group_size, "redundancy_group_size", "↓"),
]
rows_cmp = []
for name, value, col, better in summary:
    pct = percentile(pop[col], value)
    rows_cmp.append({
        "metric": f"{name} {better}",
        "this concept": None if value is None else float(value),
        "mean over mapped concepts": float(pop[col].drop_nulls().mean()),
        "percentile (≤)": None if pct is None else round(pct, 1),
    })
rates = [
    ("Unknown (n_m = 0)", n_m == 0, (pop["num_tokens"] == 0).mean(), "unknown rate"),
    ("Has an uncovered branch", u_m > 0, pop["has_uncovered_branch"].mean(), "uncovered branch rate"),
    ("Unique signature (N_σ = 1)", group_size == 1, (pop["redundancy_group_size"] == 1).mean(), "unique rate"),
]
st.dataframe(pl.DataFrame(rows_cmp), **DF_KW, hide_index=True)
st.dataframe(
    pl.DataFrame([{"property": p, "this concept": "yes" if v else "no", "share of mapped concepts": float(r), "metric": m}
                  for p, v, r, m in rates]),
    **DF_KW,
    hide_index=True,
)

with st.expander(f"Selection objective at λ = {lam}"):
    with st.spinner("Computing the objective..."):
        S_lam = objective_row(lam, kk)
    node_to_idx = load_static()[6]
    if c in node_to_idx:
        st.metric(f"S_D(m, T) with λ = {lam}", f"{S_lam[node_to_idx[c]]:.3f}")
        st.caption(
            "The quantity the greedy selection maximizes: the same random-walk coverage as above, but each hop "
            "multiplies by λ, so tokens found far from the concept count less. At λ = 1 it equals the semantic coverage."
        )
    else:
        st.caption("This concept is not in the candidate subgraph.")

# 4. Signature sharing -----------------------------------------------------------
st.header("4 · Concepts with the same signature")
if group_size is None:
    st.caption("This concept is missing from the population table.")
elif group_size == 1:
    st.success("Unique signature: no other mapped concept receives this representation.")
else:
    sig_id = row["signature_id"][0]
    others = pop.filter((pl.col("signature_id") == sig_id) & (pl.col("mapped_id") != c))["mapped_id"].to_list()
    st.write(
        f"**{len(others)}** other mapped concept(s) share this signature (N_σ = {group_size}): "
        "a downstream model receives the same representation for all of them."
    )
    st.dataframe(
        pl.DataFrame({"label": [label_of(o) for o in others], "id": others}).sort("label"),
        **DF_KW,
        hide_index=True,
    )
    other = st.selectbox("Inspect one of them", sorted(others, key=label_of), format_func=lambda o: f"{label_of(o)} ({o})")
    if st.button("Tokenize this concept"):
        st.session_state["run"] = {"lam": lam, "k": kk, "concept": other}
        st.rerun()
