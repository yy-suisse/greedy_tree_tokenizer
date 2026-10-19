"""
Per-concept distributions on the mapped concepts M for our method, at one token budget per
compression level and four values of the decay lambda.
    (a) proximity-coverage score   (box: IQR, line: median, whiskers: P10-P90, dot: mean)
    (b) tokens per concept         (same)

Step 1 (per-concept table, on the full candidate set of M, not HERO):
    python plot_dist_k_lambda.py compute <repo_root> <out.parquet>
Step 2 (figure):
    python plot_dist_k_lambda.py plot <per_concept.parquet> <out.pdf>
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Patch

KS = [3500, 10000, 16500]
LEVEL = {3500: "high", 10000: "mid", 16500: "low"}
LAMS = [0.3, 0.6, 0.9, 1.0]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
_VIR = plt.cm.viridis
LAM_COLOR = {0.3: _VIR(0.05), 0.6: _VIR(0.36), 0.9: _VIR(0.62), 1.0: MUTED}   # lambda = 1: no decay


def compute(repo, out, graph, mapped, cand_dir, D=9):
    import pickle
    sys.path.insert(0, repo)
    from src.graph_tokenizer_gd_tree_dev import tokenizer
    from src.graph_tokenizer_gd_tree_dev.fidelity_eval import evaluate_token_set
    G = pickle.load(open(graph, "rb"))
    M = pl.read_parquet(mapped)["id"].unique().to_list()
    adj = tokenizer.build_out_adjacency(G)
    rows = []
    for lam in LAMS:
        toks = pl.read_parquet(f"{cand_dir}{lam}.parquet")["token"].to_list()
        for k in KS:
            _, pc = evaluate_token_set(M, adj, toks[:k], D, return_per_concept=True)
            rows.append(pc.select("concept", "fidelity", "n_tokens", "unique", "unknown", "group_size")
                          .with_columns(pl.lit(k).alias("k"), pl.lit(lam).alias("lam")))
    pl.concat(rows).write_parquet(out)


def _boxes(ax, d, col, width):
    for i, k in enumerate(KS):
        for j, lam in enumerate(LAMS):
            v = d.filter((pl.col("k") == k) & (pl.col("lam") == lam))[col].to_numpy()
            x = i + (j - (len(LAMS) - 1) / 2) * width
            p10, p25, p50, p75, p90 = np.percentile(v, [10, 25, 50, 75, 90])
            c = LAM_COLOR[lam]
            ax.add_patch(plt.Rectangle((x - width * 0.42, p25), width * 0.84, max(p75 - p25, 1e-9),
                                       facecolor=c, edgecolor=c, alpha=0.35, lw=0, zorder=2))
            ax.add_patch(plt.Rectangle((x - width * 0.42, p25), width * 0.84, max(p75 - p25, 1e-9),
                                       fill=False, edgecolor=c, lw=0.9, zorder=3))
            ax.plot([x, x], [p10, p25], color=c, lw=0.9, zorder=2)
            ax.plot([x, x], [p75, p90], color=c, lw=0.9, zorder=2)
            ax.plot([x - width * 0.42, x + width * 0.42], [p50, p50], color=c, lw=2.0, zorder=4,
                    solid_capstyle="butt")
            ax.plot(x, v.mean(), marker="o", ms=2.6, color="white", markeredgecolor=c,
                    markeredgewidth=0.8, zorder=5)


def plot(per_concept, fname):
    d = pl.read_parquet(per_concept)
    plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
    w = 0.2
    _boxes(axes[0], d, "fidelity", w)
    axes[0].set_title("(a) Proximity-coverage score $\\uparrow$", fontsize=7.5, color=INK, pad=5)
    axes[0].set_ylim(0.38, 1.03)
    _boxes(axes[1], d, "n_tokens", w)
    axes[1].set_title("(b) Tokens per concept $\\downarrow$", fontsize=7.5, color=INK, pad=5)
    axes[1].set_ylim(0, None)

    for ax in axes:
        ax.set_xlim(-0.6, len(KS) - 0.4)
        ax.set_xticks(range(len(KS)))
        ax.set_xticklabels([f"{k:,}\n({LEVEL[k]})" for k in KS])
        ax.set_xlabel("Token budget $k$ (compression level)", fontsize=7, color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=2.5, width=0.6, labelsize=6.8)

    handles = [Patch(facecolor=LAM_COLOR[l], edgecolor=LAM_COLOR[l], alpha=0.8,
                     label=(f"$\\lambda = {l}$" if l < 1 else "$\\lambda = 1$ (no decay)")) for l in LAMS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               bbox_transform=fig.transFigure, ncol=4, frameon=False, fontsize=7.2)
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    if fname.endswith(".pdf"):
        fig.savefig(fname[:-4] + ".png", dpi=300, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    if sys.argv[1] == "compute":
        # defaults = paths on the user's PC (config.py)
        repo, out = sys.argv[2], sys.argv[3]
        compute(repo, out,
                graph="D:/greedy_graph_data/processed_graph/combined_subgraphs.gpickle",
                mapped="D:/HUG_graph_data/2026-07-03/mapped_concepts.parquet",
                cand_dir="D:/greedy_graph_data/greedy_tree_candidates/")
    else:
        plot(sys.argv[2], sys.argv[3])
