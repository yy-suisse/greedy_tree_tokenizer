"""
Per-concept drill-down helpers for app_new_tokenizer.py.

Ported from https://github.com/yy-suisse/graph_tokenizer/blob/main/src/drilldown_new.py,
adapted to this project's current tokenizer.py/eval.py API: the precomputed transition
matrix (build_coverage_transition/compute_semantic_coverage) and out-adjacency
(build_out_adjacency) instead of passing G directly, "token" as the candidate column
(this project's parquet schema, vs. drilldown_new's multi-name _pick_col guess), and
the glob-discovered `all_candidates` dict (config.CandidateLists has no unified
all_candidate_to_test registry here) instead of a hardcoded one.
"""

import glob
import os
import pickle
from collections import Counter

import polars as pl

import src.graph_tokenizer_gd_tree_dev.config as config
from src.graph_tokenizer_gd_tree_dev.eval import _iter_contexts, build_context_trees
from src.graph_tokenizer_gd_tree_dev.tokenizer import compute_semantic_coverage, signature


def load_graph():
    with open(config.ProcessedGraph().combined_subgraphs, "rb") as f:
        return pickle.load(f)


def load_id_to_label() -> dict:
    with open(config.ProcessedGraph().id_to_label, "rb") as f:
        return pickle.load(f)


def load_mapped_concepts() -> pl.DataFrame:
    return pl.read_parquet(config.BasicConfig().mapped_path).select("id", "label")


def load_all_candidates() -> dict:
    """{category: {file_type: path}} -- same glob-based discovery as 4.comparison_candidate_set.ipynb."""

    def to_type_dict(file_list):
        return {os.path.splitext(os.path.basename(f))[0]: f for f in file_list}

    gd_tree_list = glob.glob(config.CandidateLists().path_greedy_tree + "/*.parquet")
    baseline_list = glob.glob(config.CandidateLists().baseline_path + "/*.parquet")
    return {
        "greedy_tree_margin": to_type_dict(gd_tree_list),
        "baseline": to_type_dict(baseline_list),
    }


def get_candidate_ids(all_candidates: dict, category: str, file_type: str, k: int, iter: int = 0) -> list:
    """Top-k candidate ids for one (category, file_type) candidate list, in the file's own
    rank order -- mirrors the head(k) selection used in 4.comparison_candidate_set.ipynb.
    """
    path = all_candidates[category][file_type]
    df = pl.read_parquet(path)
    if file_type == "k_random_all_samples":
        return df.filter((pl.col("k") == k) & (pl.col("iter") == iter))["token"].to_list()
    return df.head(k)["token"].to_list()


def get_all_concept_scores(trees: dict, S, node_to_idx: dict, mapped_ids: list, D: int) -> pl.DataFrame:
    """Per-concept score table -- one row per mapped concept.

    Columns: mapped_id, frac_sem_cov (S_D(c,T), the semantic-coverage matrix's own per-concept
    value -- this concept's contribution to the aggregate semantic_coverage metric),
    mean_distance (raw mean hop-distance to found tokens, None if fully uncovered, uncovered
    branches excluded), distance_score (this concept's contribution to the aggregate
    distance_score metric -- same 1 - mean_distance/(D+1) transform as eval.py's
    distance_score, with uncovered branches counted at D+1 rather than excluded), num_tokens
    (== this concept's contribution to the aggregate conciseness metric: token leaves in its
    tree), tree_complexity (this concept's contribution to the aggregate tree_complexity
    metric: contexts, root + subcontexts, in its tree), redundancy_group_size (how many mapped
    concepts share this concept's exact context-tree signature), signature_id (group key --
    concepts with the same signature_id are interchangeable representations under this T).
    """
    sigs = {c: signature(trees[c]) for c in mapped_ids if c in trees}
    sig_counts = Counter(sigs.values())
    sig_to_gid: dict = {}

    rows = []
    for c in mapped_ids:
        tree = trees.get(c)
        if tree is None:
            continue
        contexts = list(_iter_contexts(tree))
        distances = [d for ctx in contexts for _, d in ctx.tokens]
        distances_w_penalty = distances + [D + 1 for ctx in contexts if ctx.uncovered]
        mean_distance_w_penalty = sum(distances_w_penalty) / len(distances_w_penalty) if distances_w_penalty else D + 1
        sig = sigs[c]
        rows.append({
            "mapped_id": c,
            "frac_sem_cov": float(S[-1, node_to_idx[c]]) if c in node_to_idx else None,
            "mean_distance": sum(distances) / len(distances) if distances else None,
            "distance_score": 1 - mean_distance_w_penalty / (D + 1),
            "num_tokens": sum(len(ctx.tokens) for ctx in contexts),
            "tree_complexity": len(contexts),
            "redundancy_group_size": sig_counts[sig],
            "signature_id": sig_to_gid.setdefault(sig, len(sig_to_gid)),
        })
    return pl.DataFrame(rows)


def flatten_concept_tree(root) -> tuple[list[dict], list[dict]]:
    """Flatten a concept's context tree into display rows.

    Returns (token_rows, uncovered_rows):
      - token_rows: {candidate_id, relation, distance} for every token found, `relation`
        being the full chain of relations opened to reach that context (" > "-joined;
        "ROOT" for tokens found without leaving the IS_A chain).
      - uncovered_rows: {relation} for every branch that died without finding a token.
    """
    token_rows, uncovered_rows = [], []

    def walk(ctx, path):
        rel_path = " > ".join(path) if path else "ROOT"
        for tok, d in ctx.tokens:
            token_rows.append({"candidate_id": tok, "relation": rel_path, "distance": d})
        if ctx.uncovered:
            uncovered_rows.append({"relation": rel_path})
        for sub in ctx.subcontexts:
            walk(sub, path + [sub.relation])

    walk(root, [])
    return token_rows, uncovered_rows


def render_context_tree(root, id_to_label: dict) -> str:
    """ASCII tree view of a concept's context tree, for display alongside
    flatten_concept_tree's flat table -- each relation the tree opens becomes a nested
    branch, with token leaves and uncovered (`∅`) leaves shown at whatever depth they
    were actually found.
    """
    lines: list[str] = []

    def label(cid: str) -> str:
        name = id_to_label.get(cid)
        return f"{name} ({cid})" if name else cid

    def render(node, prefix: str, is_last: bool, is_root: bool) -> None:
        head = node.relation
        if node.destination is not None:
            head += f" -> {label(node.destination)}"
        connector = "" if is_root else ("└── " if is_last else "├── ")
        lines.append(f"{prefix}{connector}{head}")

        child_prefix = prefix if is_root else prefix + ("    " if is_last else "│   ")

        leaves = [("token", tok, d) for tok, d in node.tokens]
        if node.uncovered:
            leaves.append(("uncovered", None, None))
        entries = len(leaves) + len(node.subcontexts)

        i = 0
        for kind, tok, d in leaves:
            i += 1
            conn = "└── " if i == entries else "├── "
            if kind == "token":
                lines.append(f"{child_prefix}{conn}• {label(tok)}  d={d}")
            else:
                lines.append(f"{child_prefix}{conn}∅ uncovered")

        for sub in node.subcontexts:
            i += 1
            render(sub, child_prefix, i == entries, is_root=False)

    render(root, "", True, is_root=True)
    return "\n".join(lines)


def is_exact_match(tree, concept_id: str) -> bool:
    return (
        len(tree.tokens) == 1
        and tree.tokens[0][0] == concept_id
        and tree.tokens[0][1] == 0.0
        and not tree.subcontexts
        and not tree.uncovered
    )


def is_fully_unk(tree) -> bool:
    return sum(len(ctx.tokens) for ctx in _iter_contexts(tree)) == 0


def tokenize_for(A, node_to_idx, adj, mapped_ids: list, id_to_label: dict, D: int,
                  all_candidates: dict, category: str, file_type: str, k: int, iter: int = 0):
    """End-to-end: pick the candidate set for (category, file_type, k[, iter]), build every
    mapped concept's context tree against it, and compute the per-concept score table.

    A/node_to_idx/adj are the precomputed, T-independent graph structures from
    tokenizer.build_coverage_transition/build_out_adjacency -- build them once (e.g. in a
    st.cache_resource) and pass them in rather than rebuilding per call.
    """
    T = get_candidate_ids(all_candidates, category, file_type, k, iter)
    trees = build_context_trees(mapped_ids, adj, T, D, id_to_label)
    S = compute_semantic_coverage(A, node_to_idx, T, D)
    concept_scores = get_all_concept_scores(trees, S, node_to_idx, mapped_ids, D)
    return T, trees, concept_scores
