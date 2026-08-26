"""
Graph Tokenizer -- Candidate Selection Method Comparison (Streamlit app).

Ported from https://github.com/yy-suisse/graph_tokenizer/blob/main/app_new_tokenizer.py,
adapted to this project's current pipeline/API:
  - `load_scores()` builds the comparison table from this project's two persisted result
    files (results/greedy_tree_performance.parquet + results/baseline_performance.parquet,
    the latter averaged over `iter` for k_random_all_samples) instead of a single unified
    metrics_result.parquet -- exactly the same concat/aggregate the notebook's plotting
    cell (4.comparison_candidate_set.ipynb) does.
  - `load_graph_data()` also precomputes the transition matrix (A/node_to_idx) and
    out-adjacency (adj) ONCE via tokenizer.build_coverage_transition/build_out_adjacency
    -- the same optimization applied to the notebook sweep -- instead of rebuilding them
    on every drill-down/distribution tokenize call.
  - `LOWER_IS_BETTER` drops `distance_score`: this project's distance_score was changed to
    higher-is-better (1 - mean_distance/(D+1), dead-ends counted at D+1), unlike the source
    app's older lower-is-better version.
"""

import polars as pl
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

import src.graph_tokenizer_gd_tree_dev.config as config
from src.graph_tokenizer_gd_tree_dev import drilldown, graph_viz
from src.graph_tokenizer_gd_tree_dev.tokenizer import build_coverage_transition, build_out_adjacency

st.set_page_config(page_title="Graph Tokenizer — Candidate Selection Comparison", layout="wide")

D = config.TokenizerParam().max_dist_candidate

METRIC_COLS = [
    "semantic_coverage",
    "conciseness",
    "distance_score",
    "uniqueness_entropy",
    "unk_rate",
    "uncovered_rate",
    "tree_complexity",
    "exact_rate",
    "unique_rate",
]


@st.cache_data
def load_scores() -> pl.DataFrame:
    perf_df = pl.read_parquet(config.Results().perf_greedy_tree_path).sort(["category", "file_type", "k"])
    perf_df_k = (
        pl.read_parquet(config.Results().perf_baseline_path)
        .sort(["category", "file_type", "k"])
        .group_by(["category", "file_type", "k"])
        .agg(pl.col(METRIC_COLS).mean())
    )
    df = pl.concat([perf_df, perf_df_k], how="diagonal_relaxed").sort(["category", "file_type", "k"])
    return df.with_columns((pl.col("category") + "/" + pl.col("file_type")).alias("method"))


@st.cache_resource(show_spinner="Loading combined subgraph + mapped concepts...")
def load_graph_data():
    G = drilldown.load_graph()
    df_mapped = drilldown.load_mapped_concepts()
    mapped_ids = df_mapped["id"].unique().to_list()
    id_to_label = drilldown.load_id_to_label()
    all_candidates = drilldown.load_all_candidates()
    # Built once and shared across every drill-down/distribution call instead of being
    # rebuilt per (method, k) -- see tokenizer.build_coverage_transition's docstring.
    A, node_to_idx = build_coverage_transition(G)
    adj = build_out_adjacency(G)
    return G, mapped_ids, id_to_label, df_mapped, all_candidates, A, node_to_idx, adj


@st.cache_data
def load_mapped_concept_options():
    _G, _mapped_ids, _id_to_label, df_mapped, *_ = load_graph_data()
    return df_mapped.select("id", "label").to_pandas()


@st.cache_data(show_spinner="Tokenizing with this method/k...")
def cached_tokenize_for(method: str, k: int, iter: int = 0):
    category, file_type = method.split("/", 1)
    G, mapped_ids, id_to_label, _df_mapped, all_candidates, A, node_to_idx, adj = load_graph_data()
    T, trees, concept_scores = drilldown.tokenize_for(
        A, node_to_idx, adj, mapped_ids, id_to_label, D, all_candidates, category, file_type, k, iter,
    )
    return T, trees, concept_scores.to_pandas()


SCORE_COLS = METRIC_COLS
LOWER_IS_BETTER = {"conciseness", "tree_complexity", "unk_rate", "uncovered_rate"}

df = load_scores()
k_values = sorted(df["k"].unique().to_list())
methods = sorted(df["method"].unique().to_list())

st.title("Graph Tokenizer — Candidate Selection Method Comparison")
st.caption(
    "Browse precomputed evaluation metrics (`4.comparison_candidate_set.ipynb`) for every "
    "candidate-selection method (`2.greedy_tree_candidate_seledction.ipynb` / "
    "`3.baseline_candidate_selection.ipynb`), on the "
    f"**{len(load_graph_data()[1]):,} mapped concepts**. Charts use **k** directly on the "
    "x-axis — every method's candidate list is evaluated at the same nominal k values."
)

with st.sidebar:
    st.header("Filters")
    k_range = st.select_slider(
        "k range",
        options=k_values,
        value=(min(k_values), max(k_values)),
    )

# --- Method toggle panel ---
with st.container(border=True):
    toggle_cols = st.columns([1, 1] + [2] * len(methods))
    with toggle_cols[0]:
        if st.button("All", width="stretch"):
            for m in methods:
                st.session_state[f"method_toggle_{m}"] = True
    with toggle_cols[1]:
        if st.button("None", width="stretch"):
            for m in methods:
                st.session_state[f"method_toggle_{m}"] = False
    selected_methods = []
    for col, method in zip(toggle_cols[2:], methods):
        checked = col.checkbox(
            method,
            value=st.session_state.get(f"method_toggle_{method}", True),
            key=f"method_toggle_{method}",
        )
        if checked:
            selected_methods.append(method)

if not selected_methods:
    st.warning("No methods selected — select at least one above.")
    st.stop()

filtered = df.filter(
    pl.col("method").is_in(selected_methods)
    & (pl.col("k") >= k_range[0])
    & (pl.col("k") <= k_range[1])
)

tab_compare, tab_table, tab_dist, tab_drilldown = st.tabs(
    ["Compare scores", "Raw table", "Score distributions", "Concept drill-down"],
)

with tab_compare:
    st.subheader("All scores vs k")
    compare_cols = st.columns(2)
    for i, score in enumerate(SCORE_COLS):
        with compare_cols[i % 2]:
            direction = "lower is better" if score in LOWER_IS_BETTER else "higher is better"
            fig_small = px.line(
                filtered.sort("k").to_pandas(),
                x="k",
                y=score,
                color="method",
                markers=True,
                title=f"{score} ({direction})",
            )
            fig_small.update_layout(
                height=300,
                showlegend=(i == 0),
                legend_itemclick=False,
                legend_itemdoubleclick=False,
                margin={"t": 40, "b": 10},
            )
            st.plotly_chart(fig_small, use_container_width=True, key=f"compare_small_{score}")

with tab_table:
    st.subheader("Filtered raw scores")
    st.dataframe(filtered.sort(["method", "k"]), width="stretch", hide_index=True)
    st.download_button(
        "Download filtered scores as CSV",
        filtered.write_csv(),
        file_name="filtered_scores.csv",
        mime="text/csv",
    )

with tab_dist:
    st.subheader("Per-concept score distributions")
    st.caption(
        "Choose a method and k to see how individual scores are spread across all mapped concepts "
        "— revealing whether good aggregate scores hide a long tail of poorly-covered concepts.",
    )

    col_dm, col_dk = st.columns(2)
    with col_dm:
        dist_method = st.selectbox("Method", methods, key="dist_method")
    with col_dk:
        dist_method_rows = df.filter(pl.col("method") == dist_method).sort("k")
        dist_k_options = dist_method_rows["k"].to_list()
        dist_k = st.select_slider("k", options=dist_k_options, key="dist_k")

    if dist_method.endswith("k_random_all_samples"):
        st.info(
            "`k_random_all_samples` uses a single representative draw (iter 0) here, not the "
            "multi-draw average shown in the leaderboard — scores will differ slightly.",
        )

    method_row = dist_method_rows.filter(pl.col("k") == dist_k)
    _T, _trees, concept_df = cached_tokenize_for(dist_method, dist_k)
    st.markdown("**Method-level aggregate scores at this k**")
    agg_cols = st.columns(len(SCORE_COLS))
    for col, name in zip(agg_cols, SCORE_COLS):
        val = method_row[name][0] if method_row.height else None
        col.metric(name.replace("_", " "), f"{val:.3f}" if val is not None else "—")

    st.divider()

    # --- distributions ---
    DIST_SPECS = [
        ("frac_sem_cov",          "Semantic coverage per concept",   "S_D(c, T) — 1.0 = full coverage, 0.0 = fully uncovered",    True),
        ("mean_distance",         "Mean token distance per concept",  "Average hop distance to found tokens; fully uncovered concepts excluded",   False),
        ("num_tokens",            "Number of tokens per concept",     "How many token leaves a concept's context tree resolves to",                     False),
        ("redundancy_group_size", "Concepts sharing the same context-tree signature — how many mapped concepts have an identical representation", "Number of concepts in the same signature group (1 = fully unique representation)", False),
    ]

    for col_a, col_b in zip(DIST_SPECS[::2], DIST_SPECS[1::2]):
        left, right = st.columns(2)
        for pane, (col_name, title, x_label, show_unk_note) in zip([left, right], [col_a, col_b]):
            with pane:
                plot_df = concept_df[["mapped_id", col_name]].dropna()

                # Binary summary bar for the token-set uniqueness metric
                if col_name == "redundancy_group_size":
                    n_unique = int((plot_df[col_name] == 1).sum())
                    n_shared = int((plot_df[col_name] > 1).sum())
                    total = n_unique + n_shared
                    import plotly.graph_objects as go
                    fig_bar = go.Figure(data=[
                        go.Bar(name="Unique signature", x=["Token set uniqueness"], y=[n_unique],
                               marker_color="#2ecc71",
                               text=[f"{n_unique} ({100*n_unique/total:.1f}%)"], textposition="outside"),
                        go.Bar(name="Shared signature (≥2 concepts)", x=["Token set uniqueness"], y=[n_shared],
                               marker_color="#e74c3c",
                               text=[f"{n_shared} ({100*n_shared/total:.1f}%)"], textposition="outside"),
                    ])
                    fig_bar.update_layout(
                        barmode="stack",
                        title="Unique vs. shared context-tree signatures",
                        height=260,
                        showlegend=True,
                        margin={"t": 40, "b": 10},
                        yaxis_title="Number of concepts",
                    )
                    st.plotly_chart(fig_bar, use_container_width=True, key="dist_uniqueness_bar")

                if col_name == "redundancy_group_size":
                    max_val = int(plot_df[col_name].max())
                    x_cap = 50
                    tail_n = int((plot_df[col_name] > x_cap).sum())
                    fig = px.histogram(
                        plot_df,
                        x=col_name,
                        title=title,
                        labels={col_name: x_label},
                        color_discrete_sequence=["#4a90d9"],
                    )
                    fig.update_traces(xbins=dict(start=0.5, end=max_val + 0.5, size=1))
                    fig.update_layout(height=340, bargap=0.05, showlegend=False,
                                      margin={"t": 40, "b": 10}, xaxis_range=[0, x_cap])
                else:
                    fig = px.histogram(
                        plot_df,
                        x=col_name,
                        nbins=40,
                        title=title,
                        labels={col_name: x_label},
                        color_discrete_sequence=["#4a90d9"],
                    )
                    fig.update_layout(height=340, bargap=0.05, showlegend=False,
                                      margin={"t": 40, "b": 10})
                st.plotly_chart(fig, use_container_width=True, key=f"dist_{col_name}")
                if col_name == "redundancy_group_size" and tail_n > 0:
                    st.caption(f"{tail_n} concept(s) with group size > {x_cap} not shown (max = {max_val}).")
                unk_n = (concept_df["frac_sem_cov"] == 0.0).sum()
                if show_unk_note and unk_n > 0:
                    st.caption(f"{unk_n} fully-uncovered concept(s) (frac_sem_cov = 0) included at the left edge.")
                null_n = concept_df[col_name].isna().sum()
                if null_n > 0:
                    st.caption(f"{null_n} concept(s) with no value for this metric are excluded.")

    st.divider()
    st.markdown("**Joint distribution: semantic coverage vs mean distance**")
    joint_df = concept_df[["frac_sem_cov", "mean_distance", "num_tokens"]].dropna()
    fig_joint = px.scatter(
        joint_df,
        x="mean_distance",
        y="frac_sem_cov",
        color="num_tokens",
        color_continuous_scale="Viridis",
        labels={
            "mean_distance": "Mean token distance",
            "frac_sem_cov": "Semantic coverage",
            "num_tokens": "# tokens",
        },
        opacity=0.4,
    )
    fig_joint.update_traces(marker_size=4)
    fig_joint.update_layout(height=420)
    st.plotly_chart(fig_joint, use_container_width=True)

with tab_drilldown:
    st.subheader("How is a single concept tokenized?")
    st.caption(
        "This tab re-runs the tokenizer for one method/k on demand (not precomputed), "
        "so picking a new method/k takes several seconds the first time.",
    )

    col_method, col_k = st.columns(2)
    with col_method:
        dd_method = st.selectbox("Method", methods, key="dd_method")
    with col_k:
        dd_method_rows = df.filter(pl.col("method") == dd_method).sort("k")
        dd_k_options = dd_method_rows["k"].to_list()
        dd_k = st.select_slider("k", options=dd_k_options, key="dd_k")

    if dd_method.endswith("k_random_all_samples"):
        st.info(
            "`k_random_all_samples` uses a single representative draw (iter 0) here, not the "
            "multi-draw average shown in the leaderboard — scores will differ slightly.",
        )

    mapped_options = load_mapped_concept_options()

    search = st.text_input("Search mapped concept by id or label", "")
    if search:
        mask = mapped_options["id"].str.contains(search, case=False, regex=False) | mapped_options["label"].str.contains(
            search, case=False, regex=False,
        )
        matches = mapped_options[mask].head(200)
    else:
        matches = mapped_options.head(200)

    if matches.empty:
        st.warning("No mapped concept matches that search.")
    else:
        option_labels = [f"{row.label} ({row.id})" for row in matches.itertuples()]
        picked = st.selectbox("Mapped concept", option_labels, key="dd_concept")
        picked_id = matches.iloc[option_labels.index(picked)]["id"]

        T, trees, concept_df = cached_tokenize_for(dd_method, dd_k)
        G, _mapped_ids, id_to_label, _df_mapped, *_ = load_graph_data()

        tree = trees.get(picked_id)

        st.markdown(f"### {id_to_label.get(picked_id, picked_id)} (`{picked_id}`)")

        if tree is None:
            st.warning("This concept was not found in the tokenization output.")
        else:
            token_rows, uncovered_rows = drilldown.flatten_concept_tree(tree)

            method_row = df.filter((pl.col("method") == dd_method) & (pl.col("k") == dd_k))

            def _method_avg(col: str):
                return method_row[col][0] if method_row.height else None

            if drilldown.is_fully_unk(tree):
                uncovered_avg = _method_avg("uncovered_rate")
                st.error(
                    "This concept is **fully uncovered** — no selected candidate covers any of its relation "
                    "types. Counts toward `uncovered_rate`"
                    + (f" (method average at k={dd_k}: {uncovered_avg:.3f})." if uncovered_avg is not None else "."),
                )
            elif drilldown.is_exact_match(tree, picked_id):
                st.success("This concept is an **exact match** — it is itself in the candidate vocabulary.")
            else:
                st.info(f"This concept is tokenized via **{len(token_rows)} candidate assignment(s)**, shown below.")
                if uncovered_rows:
                    unk_avg = _method_avg("unk_rate")
                    st.caption(
                        f"{len(uncovered_rows)} branch(es) of this concept's context tree remain uncovered "
                        "— counts toward `unk_rate` (but not `uncovered_rate`, since other branches still "
                        "found tokens)"
                        + (f". Method average `unk_rate` at k={dd_k}: {unk_avg:.3f}." if unk_avg is not None else "."),
                    )

            if token_rows:
                detail = (
                    pl.DataFrame(token_rows)
                    .with_columns(
                        pl.col("candidate_id").replace_strict(id_to_label, default=None).alias("candidate_label"),
                    )
                    .select("candidate_id", "candidate_label", "relation", "distance")
                    .sort("relation", "distance")
                )
            else:
                detail = pl.DataFrame(
                    schema={"candidate_id": pl.Utf8, "candidate_label": pl.Utf8, "relation": pl.Utf8, "distance": pl.Float64},
                )
            st.dataframe(detail, width="stretch", hide_index=True)

            st.markdown("#### Tokenized context tree")
            st.caption(
                "IS_A hops stay inside the same branch; every other relation opens a nested one. "
                "`•` = a token found; `∅` = a branch that died without one.",
            )
            st.code(drilldown.render_context_tree(tree, id_to_label), language=None)

            st.markdown("#### This concept's scores")
            concept_row = concept_df[concept_df["mapped_id"] == picked_id]

            frac_sem_cov = concept_row["frac_sem_cov"].iloc[0] if len(concept_row) else None
            mean_distance = concept_row["mean_distance"].iloc[0] if len(concept_row) else None
            distance_score = concept_row["distance_score"].iloc[0] if len(concept_row) else None
            num_tokens = concept_row["num_tokens"].iloc[0] if len(concept_row) else None
            tree_complexity = concept_row["tree_complexity"].iloc[0] if len(concept_row) else None
            redundancy_group_size = concept_row["redundancy_group_size"].iloc[0] if len(concept_row) else None

            # The four continuous-valued metrics that are meaningful for a single concept (each is
            # exactly this concept's own contribution to the corresponding aggregate metric averaged
            # in the Compare-scores tab). uniqueness_entropy/exact_rate aren't included here: they're
            # only defined over the whole concept population, not a single concept. unk_rate/
            # uncovered_rate/unique_rate are per-concept booleans, not continuous values -- shown via
            # the banner above and the signature-sharing metric below instead of a numeric tile here.
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Semantic coverage",
                f"{frac_sem_cov:.2f}" if frac_sem_cov is not None else "—",
                help=f"S_D(c, T) for this concept. Method average at k={dd_k}: {_method_avg('semantic_coverage'):.3f}" if _method_avg("semantic_coverage") is not None else None,
            )
            m2.metric(
                "Conciseness",
                int(num_tokens) if num_tokens is not None and num_tokens == num_tokens else "—",
                help=f"Token leaves in this concept's tree. Method average tokens/concept at k={dd_k}: {_method_avg('conciseness'):.2f}" if _method_avg("conciseness") is not None else None,
            )
            m3.metric(
                "Distance score",
                f"{distance_score:.2f}" if distance_score is not None and distance_score == distance_score else "—",
                help=(
                    f"1 - mean_distance/(D+1), dead-end branches counted at D+1. Raw mean token "
                    f"distance: {mean_distance:.2f}" if mean_distance is not None and mean_distance == mean_distance else "All branches uncovered (dead-end penalty only)."
                ) + (f" Method average at k={dd_k}: {_method_avg('distance_score'):.3f}" if _method_avg("distance_score") is not None else ""),
            )
            m4.metric(
                "Tree complexity",
                int(tree_complexity) if tree_complexity is not None and tree_complexity == tree_complexity else "—",
                help=f"Contexts (root + subcontexts) in this concept's tree. Method average at k={dd_k}: {_method_avg('tree_complexity'):.2f}" if _method_avg("tree_complexity") is not None else None,
            )

            unique_avg = _method_avg("unique_rate")
            st.metric(
                "Concepts sharing this exact context-tree signature",
                int(redundancy_group_size) if redundancy_group_size is not None and redundancy_group_size == redundancy_group_size else "—",
                help=(
                    "Higher = more concepts collapse onto the same representation as this one (lower "
                    "uniqueness). A value of 1 means this concept counts toward `unique_rate`"
                    + (f" (method average at k={dd_k}: {unique_avg:.3f})." if unique_avg is not None else ".")
                ),
            )

            if len(concept_row) and redundancy_group_size and redundancy_group_size > 1:
                sig_id = concept_row["signature_id"].iloc[0]
                other_ids = concept_df.loc[
                    (concept_df["signature_id"] == sig_id) & (concept_df["mapped_id"] != picked_id),
                    "mapped_id",
                ].tolist()
                sharing_df = (
                    pl.DataFrame({"id": other_ids})
                    .with_columns(pl.col("id").replace_strict(id_to_label, default=None).alias("label"))
                    .select("label", "id")
                    .sort("label")
                )
                with st.expander(f"Other {len(other_ids)} concept(s) sharing this exact representation"):
                    st.dataframe(sharing_df, width="stretch", hide_index=True)

            st.markdown("#### Graph view")
            st.caption(
                "Blue = the tokenized concept · Green = candidates actually used to tokenize it · "
                "Red = a branch that hit the depth limit or a dead end without finding a candidate · "
                f"Gray = other concepts reachable within {D} hops that were **not** selected as "
                "tokenizing candidates.",
            )

            html = graph_viz.build_new_concept_graph_html(picked_id, G, T, D, id_to_label)
            components.html(html, height=780, scrolling=True)
