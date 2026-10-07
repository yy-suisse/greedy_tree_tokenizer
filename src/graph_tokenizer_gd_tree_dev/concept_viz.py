"""
Shared helpers for the concept-level visualization apps:
  7.visualization_example.py            (one tokenizer, one concept)
  8.compare_visualization_our_baseline.py (our method next to a baseline)

Nothing here depends on Streamlit, so the helpers can also be used from a notebook.

  - tokenize_traced: the recursion of tokenizer.expand (Algorithm 3), plus a record
    of every concept and relationship it visits, for drawing the explored subgraph
  - build_graph_html / legend_html: pyvis rendering of that subgraph
  - context_entries / nested_expression / percentile: per-concept quantities in the
    paper's notation (Sec. 3.6)
  - build_search_index / fuzzy_rank: fuzzy search over the mapped concepts
  - dataframe_width_kwargs: full-width st.dataframe on old and new Streamlit versions
"""

import difflib
import inspect
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from pyvis.network import Network

from src.graph_tokenizer_gd_tree_dev.tokenizer import Context, collapse_redundant

# Node colours (fill, border, font)
COLOR_MAPPED = ("#dc2626", "#7f1d1d", "#ffffff")  # the mapped concept: red
COLOR_TOKEN = ("#2563eb", "#1e3a8a", "#ffffff")  # tokens: blue
COLOR_INNER = ("#d1d5db", "#6b7280", "#111827")  # concepts that are not tokens: gray
COLOR_DEAD = ("#facc15", "#a16207", "#111827")  # dead end of a branch: yellow
BORDER_DEAD_BRANCH = "#eab308"  # gray node whose every path dies: yellow border

# Edge colours
EDGE_ISA = "#9ca3af"
EDGE_ATTRIBUTE = "#475569"
EDGE_DEAD = "#eab308"
EDGE_MERGED = "#cbd5e1"


# ---------------------------------------------------------------------------
# Traced Expand: same recursion as tokenizer.expand, plus a record of what it visits
# ---------------------------------------------------------------------------


@dataclass
class Trace:
    min_depth: dict = field(default_factory=dict)  # node -> smallest depth it was reached at
    roles: dict = field(default_factory=lambda: defaultdict(set))  # node -> {"token", "expanded", "dead_depth", "dead_leaf"}
    found: dict = field(default_factory=dict)  # node -> some traversal through it reached a token
    edges: dict = field(default_factory=dict)  # (u, v, relation) -> {"expanded", "found", "merged"}


def _expand_traced(u, q, d, adj, T, D, trace):
    """Mirror of tokenizer.expand (Algorithm 3). Returns True if this call reached a token."""
    trace.min_depth[u] = min(trace.min_depth.get(u, d), d)
    if u in T:  # token test first: a token at depth D is accepted
        q.add_token(u, d)
        trace.roles[u].add("token")
        found = True
    else:
        edges = adj.get(u, ())
        if d == D or not edges:
            q.mark_uncovered()
            trace.roles[u].add("dead_leaf" if not edges else "dead_depth")
            found = False
        else:
            trace.roles[u].add("expanded")
            found = False
            for r, v in edges:  # attribute edges first, IS_A last (build_out_adjacency order)
                e = trace.edges.setdefault((u, v, r), {"expanded": False, "found": False, "merged": False})
                if r == "IS_A":
                    f = _expand_traced(v, q, d + 1, adj, T, D, trace)  # transparent: same context
                else:
                    child, is_new = q.open_subcontext(r, v, d + 1)
                    if not is_new:  # (r, v) already opened in this context: not expanded again
                        e["merged"] = True
                        continue
                    f = _expand_traced(v, child, d + 1, adj, T, D, trace)
                e["expanded"] = True
                e["found"] = e["found"] or f
                found = found or f
    trace.found[u] = trace.found.get(u, False) or found
    return found


def tokenize_traced(c, adj, T, D, id_to_label):
    """Context tree of concept c (identical to tokenizer.tokenize_all_rel) and its Trace."""
    trace = Trace()
    root = Context(id_to_label, "ROOT", distance=0)
    _expand_traced(c, root, 0, adj, T, D, trace)
    return collapse_redundant(root), trace


# ---------------------------------------------------------------------------
# Per-concept quantities (paper notation, Sec. 3.6)
# ---------------------------------------------------------------------------


def context_entries(tree, label, D):
    """Rows of P_m: every token with its depth, plus one row per uncovered context at depth D + 1."""
    rows = []

    def walk(ctx, path):
        where = " > ".join(path) if path else "initial context"
        for tok, d in ctx.tokens:
            rows.append({"entry": label(tok), "id": tok, "kind": "token", "context": where, "depth": int(d)})
        if ctx.uncovered:
            rows.append({"entry": "∅ (uncovered)", "id": "", "kind": "uncovered", "context": where, "depth": D + 1})
        for sub in ctx.subcontexts:
            walk(sub, path + [f"{sub.relation} → {label(sub.destination)}"])

    walk(tree, [])
    return rows


def nested_expression(ctx, name):
    """Paper notation, e.g. (t_A, R1(t_B, R2(∅)))."""
    items = [name(tok) for tok, _ in ctx.tokens]
    if ctx.uncovered:
        items.append("∅")
    items += [f"{sub.relation}({nested_expression(sub, name)})" for sub in ctx.subcontexts]
    return ", ".join(items)


def percentile(series, value):
    """Share (%) of a polars Series that is <= value; None if undefined."""
    arr = series.drop_nulls().to_numpy()
    if value is None or arr.size == 0:
        return None
    return 100.0 * float(np.mean(arr <= value))


# ---------------------------------------------------------------------------
# Graph rendering (pyvis)
# ---------------------------------------------------------------------------


def node_style(n, root, trace, T, D):
    roles = trace.roles.get(n, set())
    found = trace.found.get(n, False)
    if n == root:
        fill, border, font = COLOR_MAPPED
        width = 2
        if n in T:
            border, width = COLOR_TOKEN[0], 5
        elif not found:
            border, width = BORDER_DEAD_BRANCH, 4
        what = "mapped concept" + (" (itself a token)" if n in T else "")
        return fill, border, font, width, what
    if "token" in roles:
        return (*COLOR_TOKEN, 2, "token")
    if roles & {"dead_depth", "dead_leaf"} and "expanded" not in roles:
        why = "no outgoing relationship" if "dead_leaf" in roles else f"depth limit D = {D} reached"
        return (*COLOR_DEAD, 2, f"dead end ({why})")
    fill, border, font = COLOR_INNER
    if not found:
        return fill, BORDER_DEAD_BRANCH, font, 4, "not a token; every path through it dies"
    extra = "; depth limit also reached on a longer path" if "dead_depth" in roles else ""
    return fill, border, font, 2, f"not a token{extra}"


def build_graph_html(root, trace, T, label, D, pruned=False, layout="bottom_up",
                     show_isa_labels=True, show_ids=False, height=800):
    """Self-contained HTML (vis.js inlined) of the subgraph Expand explored from root.

    layout: "bottom_up" (hierarchical, general concepts on top), "left_right" or "force".
    pruned: keep only the concepts and relationships on paths that reach a token.
    Returns (html, number of concepts drawn).
    """
    net = Network(height=f"{height}px", width="100%", directed=True, notebook=False, cdn_resources="in_line")

    keep = {n for n in trace.min_depth if not pruned or n == root or trace.found.get(n, False)}
    for n in keep:
        d = trace.min_depth[n]
        fill, border, font, border_width, what = node_style(n, root, trace, T, D)
        text = label(n)
        if show_ids:
            text += f"\n{n}"
        if "token" in trace.roles.get(n, set()):
            text += f"\n[d = {d}]"
        net.add_node(
            n,
            label=text,
            shape="box",
            title=f"{label(n)}\n{n}\n{what}\nsmallest depth: {d}",
            level=d,
            color={"background": fill, "border": border, "highlight": {"background": fill, "border": "#000000"}},
            font={"color": font},
            borderWidth=border_width,
        )

    pair_count = defaultdict(int)
    for (u, v, r), e in trace.edges.items():
        if u not in keep or v not in keep:
            continue
        if pruned and not e["found"]:
            continue
        if e["expanded"] and not e["found"]:
            color, width, dashes, note = EDGE_DEAD, 2.5, False, "dead branch: no token reached"
        elif not e["expanded"]:
            color, width, dashes, note = EDGE_MERGED, 1.5, True, "merged: (relation, target) already opened in this context"
        elif r == "IS_A":
            color, width, dashes, note = EDGE_ISA, 1.5, False, "IS_A: stays in the same context"
        else:
            color, width, dashes, note = EDGE_ATTRIBUTE, 2.5, False, "attribute: opens a subcontext"
        kwargs = {}
        i = pair_count[(u, v)]
        pair_count[(u, v)] += 1
        if i:  # several relations between the same two concepts: curve them apart
            kwargs["smooth"] = {"enabled": True, "type": "curvedCW", "roundness": 0.15 * i}
        net.add_edge(
            u,
            v,
            label=r if (r != "IS_A" or show_isa_labels) else "",
            title=f"{label(u)} —{r}→ {label(v)}\n{note}",
            color=color,
            width=width,
            dashes=dashes,
            **kwargs,
        )

    options = {
        "nodes": {"shape": "box", "margin": 8, "widthConstraint": {"maximum": 200}, "font": {"size": 14, "face": "arial"}},
        "edges": {
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
            "font": {"size": 12, "align": "horizontal", "strokeWidth": 3, "strokeColor": "#ffffff", "color": "#334155"},
        },
        "interaction": {"hover": True, "navigationButtons": True, "keyboard": True, "tooltipDelay": 120},
    }
    if layout == "force":
        options["layout"] = {"hierarchical": {"enabled": False}}
        options["edges"]["smooth"] = {"enabled": True, "type": "dynamic"}
        options["physics"] = {
            "enabled": True,
            "solver": "barnesHut",
            "barnesHut": {"gravitationalConstant": -12000, "springLength": 170, "springConstant": 0.02, "avoidOverlap": 0.4},
            "stabilization": {"iterations": 300},
        }
    else:
        direction = {"bottom_up": "DU", "left_right": "LR"}[layout]
        options["layout"] = {
            "hierarchical": {
                "enabled": True,
                "direction": direction,
                "levelSeparation": 150 if direction == "DU" else 280,
                "nodeSpacing": 220 if direction == "DU" else 90,
                "treeSpacing": 220,
                "blockShifting": True,
                "edgeMinimization": True,
                "parentCentralization": True,
            }
        }
        options["edges"]["smooth"] = {
            "enabled": True,
            "type": "cubicBezier",
            "forceDirection": "vertical" if direction == "DU" else "horizontal",
            "roundness": 0.4,
        }
        options["physics"] = {"enabled": False}
    net.options = options  # a dict is serialized as-is by pyvis (set_options would strip spaces)
    return net.generate_html(notebook=False), len(keep)


def count_dead_ends(trace, root):
    return sum(
        1 for n, roles in trace.roles.items()
        if roles & {"dead_depth", "dead_leaf"} and "expanded" not in roles and "token" not in roles and n != root
    )


def legend_html():
    def chip(fill, border, text, border_width=2):
        return (
            f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:16px;">'
            f'<span style="width:14px;height:14px;border-radius:3px;background:{fill};'
            f'border:{border_width}px solid {border};display:inline-block"></span>{text}</span>'
        )

    def line(color, text, dashed=False):
        style = "dashed" if dashed else "solid"
        return (
            f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:16px;">'
            f'<span style="width:24px;border-top:3px {style} {color};display:inline-block"></span>{text}</span>'
        )

    return (
        '<div style="font-size:0.85rem;line-height:2;">'
        + chip(COLOR_MAPPED[0], COLOR_MAPPED[1], "mapped concept")
        + chip(COLOR_TOKEN[0], COLOR_TOKEN[1], "token")
        + chip(COLOR_INNER[0], COLOR_INNER[1], "concept, not a token")
        + chip(COLOR_DEAD[0], COLOR_DEAD[1], "dead end (uncovered)")
        + chip(COLOR_INNER[0], BORDER_DEAD_BRANCH, "on a dead branch only", 3)
        + "<br>"
        + line(EDGE_ISA, "IS_A (same context)")
        + line(EDGE_ATTRIBUTE, "attribute (opens a subcontext)")
        + line(EDGE_DEAD, "dead branch")
        + line(EDGE_MERGED, "merged restatement", dashed=True)
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Fuzzy search over mapped concepts
# ---------------------------------------------------------------------------


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def build_search_index(ids, labels):
    """(ids, labels, word -> concept indices, sorted vocabulary) for fuzzy_rank."""
    word_to_idx = defaultdict(set)
    for i, lab in enumerate(labels):
        for w in _norm(lab).split():
            word_to_idx[w].add(i)
    return list(ids), list(labels), word_to_idx, sorted(word_to_idx)


def _fallback_scores(q, labels, word_to_idx, vocab):
    """Without rapidfuzz: share of query words found in the label, by prefix or close spelling."""
    words = _norm(q).split()
    if not words:
        return {}
    hits = defaultdict(float)
    for w in words:
        prefix = {x for x in vocab if x.startswith(w)} if len(w) >= 3 else ({w} & word_to_idx.keys())
        typos = set(difflib.get_close_matches(w, vocab, n=5, cutoff=0.8)) - prefix
        best = defaultdict(float)  # a typo match counts less than an exact or prefix match
        for x in prefix:
            for i in word_to_idx[x]:
                best[i] = 1.0
        for x in typos:
            for i in word_to_idx[x]:
                best[i] = max(best[i], 0.7)
        for i, v in best.items():
            hits[i] += v
    return {i: 100.0 * h / len(words) - 0.05 * len(labels[i]) for i, h in hits.items()}


def fuzzy_rank(query, index, limit=50):
    """Indices of the best-matching mapped concepts (by label, or by id prefix)."""
    ids, labels, word_to_idx, vocab = index
    q = query.strip()
    if not q:
        return []
    scores = {}
    if q.isdigit():
        for i, cid in enumerate(ids):
            if cid.startswith(q):
                scores[i] = 101.0 if cid == q else 100.0
    try:
        from rapidfuzz import fuzz, process, utils

        for _lab, score, i in process.extract(q, labels, scorer=fuzz.WRatio, processor=utils.default_process, limit=limit):
            scores[i] = max(scores.get(i, 0.0), float(score))
    except ImportError:
        for i, score in _fallback_scores(q, labels, word_to_idx, vocab).items():
            scores[i] = max(scores.get(i, 0.0), score)
    ranked = sorted(scores, key=lambda i: (-scores[i], len(labels[i])))
    return ranked[:limit]


# ---------------------------------------------------------------------------
# Streamlit compatibility
# ---------------------------------------------------------------------------


def dataframe_width_kwargs(st):
    """Full-width tables on any Streamlit version: newer ones take width="stretch",
    older ones only an int width plus use_container_width."""
    default = inspect.signature(st.dataframe).parameters["width"].default
    return {"width": "stretch"} if isinstance(default, str) else {"use_container_width": True}
