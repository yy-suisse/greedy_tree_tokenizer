"""
pyvis graph rendering for app_new_tokenizer.py's concept drill-down tab.

Ported from https://github.com/yy-suisse/graph_tokenizer/blob/main/src/graph_viz.py
(build_new_concept_graph_html only -- the older build_concept_graph_html counterpart in
that file targets a different, class-based tokenizer this project doesn't have).
"""

import networkx as nx
from pyvis.network import Network

COLOR_CENTER = "#1f3a93"
COLOR_CANDIDATE = "#2ecc71"
COLOR_NOT_CANDIDATE = "#b0b0b0"
COLOR_UNK = "#e74c3c"

NODE_FONT = {"size": 26, "face": "arial", "strokeWidth": 3, "strokeColor": "#ffffff"}
EDGE_FONT = {"size": 20, "face": "arial", "strokeWidth": 3, "strokeColor": "#ffffff", "align": "middle"}


def _label(concept_id: str, id_to_label: dict) -> str:
    label = id_to_label.get(concept_id)
    return f"{label}\n({concept_id})" if label else concept_id


def build_new_concept_graph_html(
    concept_id: str,
    G: nx.MultiDiGraph,
    T,
    D: int,
    id_to_label: dict,
    max_nodes: int = 200,
    height: str = "780px",
) -> str:
    """
    Walks combined_subgraphs directly from concept_id (the same graph tokenizer.expand()
    draws candidates from), following every out-edge up to D hops -- IS_A included, since
    there's no separate is_a-only graph to reconstruct chains from here.

    Colors: blue = concept_id itself; green = a node in T actually reached (a token);
    red = a branch that died (depth budget D exhausted, or a dead end) without hitting a
    token; gray = an intermediate node still being explored (not itself a candidate).

    This traversal is a simplified visualization aid, not a re-run of tokenize_all_rel's
    exact context-tree semantics (no relation-based subcontext dedup) -- the token/relation
    table shown alongside it (drilldown.flatten_concept_tree) is the ground truth for what
    the tokenizer actually assigned.
    """
    net = Network(height=height, width="100%", directed=True, notebook=False, cdn_resources="remote")
    net.barnes_hut(spring_length=320, spring_strength=0.02)

    T_set = set(T)

    added_nodes: set[str] = set()
    added_edges: set[tuple[str, str, str]] = set()
    visited: set[str] = set()

    def add_node(node_id: str, **kwargs) -> None:
        if node_id in added_nodes:
            return
        added_nodes.add(node_id)
        net.add_node(node_id, **kwargs)

    def add_edge(src: str, dst: str, relation: str, **kwargs) -> None:
        key = (src, dst, relation)
        if key in added_edges:
            return
        added_edges.add(key)
        net.add_edge(src, dst, **kwargs)

    add_node(
        concept_id,
        label=_label(concept_id, id_to_label),
        color=COLOR_CENTER,
        size=55,
        font=NODE_FONT,
        title="Tokenized concept",
    )

    def walk(u: str, d: int) -> None:
        if u in visited or len(added_nodes) >= max_nodes:
            return
        visited.add(u)
        if u in T_set or d == D or G.out_degree(u) == 0:
            # Mirrors expand()'s own top-of-function check: once u is itself a token
            # (or a dead end), expand() stops there and never looks at u's neighbors --
            # most importantly for u == concept_id, an exact match (concept_id in T)
            # means the tokenizer never explored anything beyond concept_id itself.
            return

        for _, v, data in sorted(G.out_edges(u, data=True), key=lambda e: e[2]["relation"] == "IS_A"):
            if len(added_nodes) >= max_nodes:
                break
            if v == concept_id:
                continue
            relation = data["relation"]

            is_token = v in T_set
            is_dead_end = (not is_token) and (d + 1 == D or G.out_degree(v) == 0)

            if is_token:
                color, title, size = COLOR_CANDIDATE, "Tokenizing candidate", 46
            elif is_dead_end:
                color, title, size = COLOR_UNK, "Depth limit / dead end without a candidate", 36
            else:
                color, title, size = COLOR_NOT_CANDIDATE, "Not used as a tokenizing candidate", 40

            add_node(v, label=_label(v, id_to_label), size=size, font=NODE_FONT, color=color, title=title)
            add_edge(
                u, v, relation,
                label=relation, font=EDGE_FONT, color=color,
                width=3 if is_token else 2, dashes=not is_token,
            )

            if not is_token and not is_dead_end:
                walk(v, d + 1)

    walk(concept_id, 0)

    return net.generate_html(notebook=False)
