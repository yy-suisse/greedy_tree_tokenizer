"""
8. Compare our method with a baseline on one mapped concept.

The window is split in two, with the same token budget k on both sides:
  - left: our method, one greedy candidate list per distance decay lambda
    (config.CandidateLists().path_greedy_tree);
  - right: one baseline token selection method (config.CandidateLists().baseline_path).
    For random selection, the draw iter = RANDOM_ITER of k_random_all_samples is used.

Pick k and a mapped concept (fuzzy search) in the sidebar, lambda and the baseline at
the top of each side, then click "Tokenize". For each side the app shows the explored
subgraph, the context tree, the per-concept scores and the signature group.

The tracing, drawing and search helpers live in src/graph_tokenizer_gd_tree_dev/concept_viz.py
(shared with 7.visualization_example.py).

Run from the repository root:

    streamlit run 8.compare_visualization_our_baseline.py

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

st.set_page_config(page_title="Ours vs. baseline", layout="wide")

D = config.TokenizerParam().max_dist_candidate
GREEDY_DIR = config.CandidateLists().path_greedy_tree
BASELINE_DIR = config.CandidateLists().baseline_path
SEARCH_LIMIT = 50
DF_KW = dataframe_width_kwargs(st)

RANDOM_FILE = "k_random_all_samples"
RANDOM_ITER = 19  # which of the random draws (iter column) to use
DEFAULT_BASELINE = "personalized_pagerank"
BASELINE_NAMES = {  # file name -> name used in the paper
    "most_children": "Most IS_A children",
    "highest_degree": "Highest degree",
    "highest_degree_dist_1": "Highest degree, d ≤ 1",
    "closeness_centrality": "Closeness centrality",
    "pagerank": "PageRank",
    "personalized_pagerank": "Personalized PageRank",
    "eigenvector_centrality": "Eigenvector centrality",
    "discrete_set_cover": "Set cover (ablation)",
    RANDOM_FILE: f"Random selection (draw {RANDOM_ITER})",
}


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


@st.cache_resource
def mapped_label_lookup():
    df_mapped = load_static()[2]
    return dict(zip(df_mapped["id"].to_list(), df_mapped["label"].to_list()))


def label_of(cid):
    return load_static()[1].get(cid) or mapped_label_lookup().get(cid) or cid


def _parquet_files(folder):
    return {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(folder, "*.parquet"))}


@st.cache_data
def list_lambda_files():
    """{lambda name: parquet path}, sorted by lambda; file names are the lambda values."""
    out = {}
    for name, path in _parquet_files(GREEDY_DIR).items():
        try:
            float(name)
        except ValueError:
            continue
        out[name] = path
    return dict(sorted(out.items(), key=lambda kv: float(kv[0])))


@st.cache_data
def list_baseline_files():
    """{file name: parquet path}, sorted by the display name."""
    files = _parquet_files(BASELINE_DIR)
    return dict(sorted(files.items(), key=lambda kv: baseline_name(kv[0]).lower()))


def baseline_name(file_name):
    return BASELINE_NAMES.get(file_name, file_name.replace("_", " ").capitalize())


def side_title(source, name):
    return f"Ours (λ = {name})" if source == "ours" else baseline_name(name)


@st.cache_data
def full_token_list(source, name):
    """The whole ranked candidate list of a method (not used for random selection)."""
    path = list_lambda_files()[name] if source == "ours" else list_baseline_files()[name]
    return pl.read_parquet(path, columns=["token"])["token"].to_list()


@st.cache_data
def random_ks():
    path = list_baseline_files()[RANDOM_FILE]
    return sorted(pl.scan_parquet(path).select("k").unique().collect()["k"].to_list())


@st.cache_data
def tokens_for(source, name, k):
    """(token list in rank order, note for the user or None)."""
    if source == "baseline" and name == RANDOM_FILE:
        path = list_baseline_files()[RANDOM_FILE]
        tokens = (
            pl.scan_parquet(path)
            .filter((pl.col("k") == k) & (pl.col("iter") == RANDOM_ITER))
            .select("token")
            .collect()["token"]
            .to_list()
        )
        if not tokens:
            ks = random_ks()
            return [], (f"No random draw at k = {k:,}. Draws exist for k = {ks[0]:,} to {ks[-1]:,} "
                        f"in steps of {ks[1] - ks[0]:,}." if len(ks) > 1 else "No random draw at this k.")
        return tokens, None
    tokens = full_token_list(source, name)
    if k > len(tokens):
        return tokens, f"This list holds only {len(tokens):,} tokens; all of them are used."
    return tokens[:k], None


@st.cache_resource(max_entries=6, show_spinner=False)
def tokenize_population(source, name, k):
    """Per-concept table over all mapped concepts (signature groups, means, percentiles)."""
    _G, id_to_label, _df, mapped_ids, adj, A1, node_to_idx = load_static()
    T, _ = tokens_for(source, name, k)
    trees = build_context_trees(mapped_ids, adj, T, D, id_to_label)
    S = compute_semantic_coverage(A1, node_to_idx, T, D)
    df = drilldown.get_all_concept_scores(trees, S, node_to_idx, mapped_ids, D)
    ids = df["mapped_id"].to_list()
    return df.with_columns(
        pl.Series("has_uncovered_branch", [any(q.uncovered for q in _iter_contexts(trees[c])) for c in ids]),
        pl.Series("n_uncovered", [sum(1 for q in _iter_contexts(trees[c]) if q.uncovered) for c in ids]),
    )


@st.cache_resource(max_entries=32, show_spinner=False)
def tokenize_one(source, name, k, concept):
    _G, id_to_label, _df, _m, adj, _A1, _idx = load_static()
    T = set(tokens_for(source, name, k)[0])
    tree, trace = tokenize_traced(concept, adj, T, D, id_to_label)
    consistent = signature(tokenize_all_rel(concept, adj, T, D, id_to_label)) == signature(tree)
    return tree, trace, consistent


@st.cache_resource
def search_index():
    df_mapped = load_static()[2]
    return build_search_index(df_mapped["id"].to_list(), df_mapped["label"].to_list())


@st.cache_data(max_entries=200, show_spinner=False)
def fuzzy_search(query, limit=SEARCH_LIMIT):
    return fuzzy_rank(query, search_index(), limit)


@st.cache_data
def rank_lookup(source, name, k):
    """token -> 1-based position in the method's ranked list (position in the draw for random)."""
    tokens = tokens_for(source, name, k)[0] if name == RANDOM_FILE else full_token_list(source, name)
    return {t: i + 1 for i, t in enumerate(tokens)}


# ---------------------------------------------------------------------------
# One side of the comparison
# ---------------------------------------------------------------------------


def compute_side(source, name, k, concept):
    T_list, note = tokens_for(source, name, k)
    side = {"source": source, "name": name, "title": side_title(source, name), "note": note, "ok": bool(T_list)}
    if not side["ok"]:
        return side
    with st.spinner(f"{side['title']}: tokenizing all {len(load_static()[3]):,} mapped concepts (once per method and k)..."):
        pop = tokenize_population(source, name, k)
    tree, trace, consistent = tokenize_one(source, name, k, concept)
    entries = context_entries(tree, label_of, D)
    row = pop.filter(pl.col("mapped_id") == concept)
    mean_depth = sum(e["depth"] for e in entries) / len(entries) if entries else D + 1
    side.update(
        T=set(T_list), k_used=len(T_list), pop=pop, row=row, tree=tree, trace=trace, consistent=consistent,
        entries=entries,
        n_m=sum(1 for e in entries if e["kind"] == "token"),
        u_m=sum(1 for e in entries if e["kind"] == "uncovered"),
        q_m=sum(1 for _ in _iter_contexts(tree)),
        mean_depth=mean_depth,
        dist=1 - mean_depth / (D + 1),
        cov=row["frac_sem_cov"][0] if row.height else None,
        group=int(row["redundancy_group_size"][0]) if row.height else None,
        tokens={e["id"] for e in entries if e["kind"] == "token"},
    )
    return side


def status_banner(side, concept):
    if side["note"]:
        (st.warning if not side["ok"] else st.caption)(side["note"])
    if not side["ok"]:
        return
    if not side["consistent"]:
        st.warning("The traced expansion differs from tokenizer.tokenize_all_rel.")
    if concept in side["T"]:
        st.success("Itself a token: an exact match at depth 0.")
    elif side["n_m"] == 0:
        st.error(f"Unknown: no token within D = {D} hops.")
    elif side["u_m"]:
        st.warning(f"{side['n_m']} token(s), {side['u_m']} uncovered context(s).")
    else:
        st.info(f"{side['n_m']} token(s); every branch reaches a token.")


def render_graph(side, concept, view, layout, show_isa_labels, show_ids, height):
    html, n_shown = build_graph_html(
        concept, side["trace"], side["T"], label_of, D, pruned=view.startswith("Only"), layout=layout,
        show_isa_labels=show_isa_labels, show_ids=show_ids, height=height,
    )
    trace = side["trace"]
    n_tok = sum(1 for n in trace.roles if "token" in trace.roles[n])
    st.caption(
        f"{n_shown:,} of {len(trace.min_depth):,} concepts shown · {n_tok} distinct tokens · "
        f"{count_dead_ends(trace, concept)} dead ends · {len(trace.edges):,} relationships"
    )
    if n_shown > 600:
        st.warning("Large graph: try **Only paths that reach a token**.")
    if hasattr(st, "iframe"):  # newer Streamlit; components.html is deprecated there
        st.iframe(html, height=height + 30)
    else:
        components.html(html, height=height + 30, scrolling=False)
    st.download_button(
        "Download graph (HTML)", html, mime="text/html", key=f"dl_{side['source']}",
        file_name=f"explored_subgraph_{concept}_{side['name']}_k{run['k']}.html",
    )


def render_tree(side, other):
    st.code(drilldown.render_context_tree(side["tree"], load_static()[1]), language=None)
    with st.expander("Nested notation (as in the paper)"):
        st.code("(" + nested_expression(side["tree"], label_of) + ")", language=None)
        st.code("(" + nested_expression(side["tree"], str) + ")", language=None)
    ranks = rank_lookup(side["source"], side["name"], run["k"])
    other_tokens = other.get("tokens", set()) if other.get("ok") else set()
    rows = [
        {"token": e["entry"], "id": e["id"], "context": e["context"], "depth": e["depth"],
         "rank in list": ranks.get(e["id"]), "also on other side": "✓" if e["id"] in other_tokens else ""}
        for e in side["entries"] if e["kind"] == "token"
    ]
    if rows:
        st.dataframe(pl.DataFrame(rows), **DF_KW, hide_index=True)


def render_scores(side):
    pop = side["pop"]

    def delta(col, value, digits):
        if value is None:
            return None
        return f"{value - pop[col].drop_nulls().mean():+.{digits}f} vs mean"

    m1, m2 = st.columns(2)
    m1.metric("Semantic coverage", f"{side['cov']:.3f}" if side["cov"] is not None else "—",
              delta=delta("frac_sem_cov", side["cov"], 3), help="S_D(m, T), evaluated at λ = 1.")
    m2.metric("Distance score", f"{side['dist']:.3f}", delta=delta("distance_score", side["dist"], 3))
    m3, m4 = st.columns(2)
    m3.metric("Tokens n_m", side["n_m"], delta=delta("num_tokens", side["n_m"], 2), delta_color="inverse")
    m4.metric("Signature group N_σ", side["group"] if side["group"] is not None else "—",
              help="Mapped concepts with this exact signature, this one included (1 = unique).")

    entries = side["entries"]
    st.markdown("**Distance score breakdown**")
    st.dataframe(
        pl.DataFrame([{c: e[c] for c in ("entry", "kind", "context", "depth")} for e in entries])
        if entries else pl.DataFrame(schema={"entry": pl.Utf8, "kind": pl.Utf8, "context": pl.Utf8, "depth": pl.Int64}),
        **DF_KW, hide_index=True,
    )
    total = sum(e["depth"] for e in entries)
    st.latex(
        rf"\bar\ell_m = \frac{{{total}}}{{{side['n_m']} + {side['u_m']}}} = {side['mean_depth']:.3f}"
        rf"\qquad 1 - \frac{{{side['mean_depth']:.3f}}}{{{D + 1}}} = {side['dist']:.3f}"
    )

    with st.expander("This concept against all mapped concepts"):
        summary = [
            ("Semantic coverage ↑", side["cov"], "frac_sem_cov"),
            ("Distance score ↑", side["dist"], "distance_score"),
            ("Tokens per concept (n_m) ↓", side["n_m"], "num_tokens"),
            ("Contexts |Q_m| ↓", side["q_m"], "tree_complexity"),
            ("Uncovered contexts |U_m| ↓", side["u_m"], "n_uncovered"),
            ("Signature group N_σ ↓", side["group"], "redundancy_group_size"),
        ]
        rows = []
        for name, value, col in summary:
            pct = percentile(pop[col], value)
            rows.append({"metric": name, "this concept": None if value is None else float(value),
                         "mean over mapped concepts": float(pop[col].drop_nulls().mean()),
                         "percentile (≤)": None if pct is None else round(pct, 1)})
        st.dataframe(pl.DataFrame(rows), **DF_KW, hide_index=True)
        st.caption(
            f"Unknown rate {float((pop['num_tokens'] == 0).mean()):.3f} · "
            f"uncovered branch rate {float(pop['has_uncovered_branch'].mean()):.3f} · "
            f"unique rate {float((pop['redundancy_group_size'] == 1).mean()):.3f}"
        )


def render_signature(side, concept):
    if side["group"] is None:
        st.caption("This concept is missing from the population table.")
        return
    if side["group"] == 1:
        st.success("Unique signature.")
        return
    pop = side["pop"]
    sig_id = side["row"]["signature_id"][0]
    others = pop.filter((pl.col("signature_id") == sig_id) & (pl.col("mapped_id") != concept))["mapped_id"].to_list()
    st.write(f"**{len(others)}** other mapped concept(s) share this signature (N_σ = {side['group']}).")
    st.dataframe(pl.DataFrame({"label": [label_of(o) for o in others], "id": others}).sort("label"),
                 **DF_KW, hide_index=True)
    other = st.selectbox("Inspect one of them", sorted(others, key=label_of),
                         format_func=lambda o: f"{label_of(o)} ({o})", key=f"sib_{side['source']}")
    if st.button("Tokenize this concept", key=f"open_{side['source']}"):
        st.session_state["run"] = {**run, "concept": other}
        st.rerun()


# ---------------------------------------------------------------------------
# Sidebar: shared parameters
# ---------------------------------------------------------------------------

lam_files = list_lambda_files()
baseline_files = list_baseline_files()
if not lam_files:
    st.error(f"No candidate lists of our method found in `{GREEDY_DIR}`.")
    st.stop()
if not baseline_files:
    st.error(f"No baseline candidate lists found in `{BASELINE_DIR}`.")
    st.stop()

mapped_ids = load_static()[3]
search_ids = search_index()[0]

with st.sidebar:
    st.header("Shared")
    k = int(st.number_input("Token budget k (both sides)", min_value=1, value=5000, step=500, key="k"))
    query = st.text_input("Search a mapped concept (label or SNOMED CT id)", placeholder="e.g. pain of breast")
    hits = fuzzy_search(query) if query else []
    concept = None
    if query and not hits:
        st.warning("No mapped concept matches this search.")
    elif hits:
        concept = st.selectbox("Best matches", [search_ids[i] for i in hits],
                               format_func=lambda cid: f"{label_of(cid)} ({cid})")
    clicked = st.button("Tokenize", type="primary", disabled=concept is None)

    st.header("Graph view")
    view = st.radio("Show", ["Whole explored subgraph", "Only paths that reach a token"])
    layout_name = st.selectbox(
        "Layout", ["Hierarchical, general concepts on top", "Hierarchical, left to right", "Force-directed"],
    )
    show_isa_labels = st.checkbox("Label IS_A edges", value=True)
    show_ids = st.checkbox("Show concept ids in nodes", value=False)
    graph_height = st.slider("Graph height (px)", 400, 1400, 650, 50)

layout = {
    "Hierarchical, general concepts on top": "bottom_up",
    "Hierarchical, left to right": "left_right",
    "Force-directed": "force",
}[layout_name]

# ---------------------------------------------------------------------------
# Main area: method choice on each side
# ---------------------------------------------------------------------------

st.title("Our method vs. a baseline")
left, right = st.columns(2, gap="large")
with left:
    st.subheader("Left · our method")
    lam_names = list(lam_files)
    lam_name = st.selectbox("Distance decay λ", lam_names, key="lam",
                            index=lam_names.index("0.9") if "0.9" in lam_names else len(lam_names) - 1)
with right:
    st.subheader("Right · baseline")
    base_names = list(baseline_files)
    base_name = st.selectbox("Baseline method", base_names, key="baseline", format_func=baseline_name,
                             index=base_names.index(DEFAULT_BASELINE) if DEFAULT_BASELINE in base_names else 0)

if clicked:
    st.session_state["run"] = {"k": k, "concept": concept, "lam": lam_name, "baseline": base_name}

run = st.session_state.get("run")
if run is None:
    st.info("Choose k and a mapped concept in the sidebar, λ and a baseline above, then click **Tokenize**.")
    st.stop()
if (run["k"], run["lam"], run["baseline"]) != (k, lam_name, base_name) or (concept is not None and concept != run["concept"]):
    st.caption("The settings have changed. Click **Tokenize** to update the results.")

c = run["concept"]
ours = compute_side("ours", run["lam"], run["k"], c)
base = compute_side("baseline", run["baseline"], run["k"], c)
sides = [ours, base]

st.header(label_of(c))
st.caption(f"`{c}` · k = {run['k']:,} on both sides · D = {D} · {len(mapped_ids):,} mapped concepts")

# Summary --------------------------------------------------------------------------
available = [s for s in sides if s["ok"]]
if available:
    def fmt(v, digits=3):
        return "—" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))

    summary_rows = [
        ("Tokens n_m ↓", lambda s: fmt(s["n_m"])),
        ("Uncovered contexts |U_m| ↓", lambda s: fmt(s["u_m"])),
        ("Distance score ↑", lambda s: fmt(s["dist"])),
        ("Semantic coverage ↑", lambda s: fmt(s["cov"])),
        ("Signature group N_σ ↓", lambda s: fmt(s["group"])),
        ("Exact match", lambda s: "yes" if c in s["T"] else "no"),
        ("Tokens also used by the other side", lambda s: fmt(len(s["tokens"] & (ours.get("tokens", set()) if s is base else base.get("tokens", set()))))),
    ]
    table = {"": [r[0] for r in summary_rows]}
    for s in sides:
        table[s["title"]] = [r[1](s) if s["ok"] else "—" for r in summary_rows]
    st.dataframe(pl.DataFrame(table), **DF_KW, hide_index=True)

# Sections, aligned left/right ------------------------------------------------------
cols = st.columns(2, gap="large")
for col, s in zip(cols, sides):
    with col:
        st.subheader(s["title"])
        status_banner(s, c)

st.subheader("1 · Explored subgraph")
st.markdown(legend_html(), unsafe_allow_html=True)
cols = st.columns(2, gap="large")
for col, s in zip(cols, sides):
    with col:
        if s["ok"]:
            render_graph(s, c, view, layout, show_isa_labels, show_ids, graph_height)

st.subheader("2 · Context tree")
cols = st.columns(2, gap="large")
for col, s, other in zip(cols, sides, [base, ours]):
    with col:
        if s["ok"]:
            render_tree(s, other)

st.subheader("3 · Scores of this concept")
cols = st.columns(2, gap="large")
for col, s in zip(cols, sides):
    with col:
        if s["ok"]:
            render_scores(s)

st.subheader("4 · Concepts with the same signature")
cols = st.columns(2, gap="large")
for col, s in zip(cols, sides):
    with col:
        if s["ok"]:
            render_signature(s, c)
