"""
Share of primitive concepts among the top-k selected tokens, hop-based vs status-dependent decay.
One panel per lambda (rows), x = token budget k, y = share of primitive tokens in the top k.

Run from decay_with_prim_fd/ (same setup as the notebooks):
    python plot_primitive_share.py
"""
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter

sys.path.insert(0, str(Path.cwd().parent))  # repo root, where src/ lives

import config_prim_fd as config  # noqa: E402
import src.graph_tokenizer_gd_tree_dev.config as config_hop  # noqa: E402

LAMBDAS = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
SERIES = {  # label, folder, colour (categorical slots 1-2, validated light-mode pair)
    "status": ("Status-dependent decay", config.CandidateLists().path_greedy_tree, "#2a78d6"),
    "hop": ("Hop-based decay", config_hop.CandidateLists().path_greedy_tree, "#eb6834"),
}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SATURATION_K = 942  # lambda = 1: semantic coverage reaches 1 here, later picks are tie-breaks


def primitive_share(tokens, is_primitive, step=100):
    """share of primitive concepts among the first k tokens, for k = step, 2*step, ..., len(tokens)"""
    flags = np.array([is_primitive.get(t, True) for t in tokens], dtype=float)
    k = np.arange(step, len(flags) + 1, step)
    return k, np.cumsum(flags)[k - 1] / k


def plot_primitive_share(fname=None):
    st = pl.read_parquet(config.ProcessedGraph.concept_and_status, columns=["id", "status"])
    is_primitive = dict(zip(st["id"].to_list(), (st["status"] == "primitive").to_list()))

    # reference: share of primitive concepts among all nodes of the graph (= a random selection)
    with open(config_hop.ProcessedGraph().combined_subgraphs, "rb") as f:
        G = pickle.load(f)
    graph_share = np.mean([is_primitive.get(u, True) for u in G.nodes()])

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#c3c2b7", "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(len(LAMBDAS), 1, figsize=(6.0, 0.95 * len(LAMBDAS)), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, lam in zip(axes, LAMBDAS):
        ax.axhline(graph_share, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
        for key, (label, folder, color) in SERIES.items():
            tokens = pl.read_parquet(f"{folder}{lam}.parquet")["token"].to_list()
            k, share = primitive_share(tokens, is_primitive)
            ax.plot(k, share, color=color, lw=1.6, label=label, zorder=3 if key == "status" else 2,
                    solid_capstyle="round")
        if lam == "1.0":
            ax.axvspan(SATURATION_K, k.max(), color=GRID, alpha=0.6, lw=0, zorder=0)
            ax.text(SATURATION_K + 300, 0.93, "coverage = 1: later picks are tie-breaks", fontsize=6,
                    color=MUTED, va="top")
        ax.text(-0.11, 0.5, f"$\\lambda = {lam}$", transform=ax.transAxes, ha="right", va="center", color=INK)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].text(1.0, graph_share + 0.04, f"all graph concepts ({graph_share:.0%})", transform=axes[0].get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=6, color=MUTED)
    axes[-1].xaxis.set_major_locator(MultipleLocator(5000))
    axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    axes[-1].set_xlabel("Token budget $k$", color=INK2)
    axes[-1].set_xlim(0, None)
    fig.supylabel("Share of primitive concepts among the top-$k$ tokens", color=INK2, fontsize=8, x=-0.04)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               fontsize=7.5, labelcolor=INK2, handlelength=2.4)
    if fname:
        fig.savefig(fname, dpi=400, bbox_inches="tight")
    return fig


def plot_status_composition(fname=None):
    """Status-dependent decay only: 100% stacked area of primitive vs fully defined tokens in the top k."""
    st = pl.read_parquet(config.ProcessedGraph.concept_and_status, columns=["id", "status"])
    is_primitive = dict(zip(st["id"].to_list(), (st["status"] == "primitive").to_list()))
    with open(config_hop.ProcessedGraph().combined_subgraphs, "rb") as f:
        G = pickle.load(f)
    graph_share = np.mean([is_primitive.get(u, True) for u in G.nodes()])

    C_PRIM, C_FD = "#2a78d6", "#eb6834"   # categorical slots 1-2 (validated pair)
    SURFACE = "#fcfcfb"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#c3c2b7", "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(len(LAMBDAS), 1, figsize=(6.0, 0.95 * len(LAMBDAS)), sharex=True, sharey=True,
                             constrained_layout=True)
    folder = config.CandidateLists().path_greedy_tree
    for ax, lam in zip(axes, LAMBDAS):
        tokens = pl.read_parquet(f"{folder}{lam}.parquet")["token"].to_list()
        k, share = primitive_share(tokens, is_primitive)
        ax.fill_between(k, 0, share, color=C_PRIM, lw=0, label="Primitive")
        ax.fill_between(k, share, 1, color=C_FD, lw=0, label="Fully defined")
        ax.plot(k, share, color=SURFACE, lw=1.2)                       # surface gap between the two fills
        ax.axhline(graph_share, color=INK, lw=0.6, ls=(0, (2, 2)), alpha=0.55)
        if lam == "1.0":
            ax.axvline(SATURATION_K, color=INK, lw=0.6, alpha=0.55)
            ax.text(SATURATION_K + 300, 0.93, "coverage = 1: later picks are tie-breaks", fontsize=6,
                    color=SURFACE, va="top")
        ax.text(-0.11, 0.5, f"$\\lambda = {lam}$", transform=ax.transAxes, ha="right", va="center", color=INK)
        ax.set_ylim(0, 1)
        ax.set_xlim(k.min(), k.max())
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].text(0.01, graph_share + 0.04, f"dotted: all graph concepts, {graph_share:.0%} primitive",
                 transform=axes[0].get_yaxis_transform(), ha="left", va="bottom", fontsize=6, color=SURFACE)
    axes[-1].xaxis.set_major_locator(MultipleLocator(5000))
    axes[-1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    axes[-1].set_xlabel("Token budget $k$", color=INK2)
    fig.supylabel("Share of the top-$k$ tokens (status-dependent decay)", color=INK2, fontsize=8, x=-0.04)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               fontsize=7.5, labelcolor=INK2, handlelength=1.6)
    if fname:
        fig.savefig(fname, dpi=400, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    Path("figures").mkdir(exist_ok=True)
    plot_primitive_share("figures/primitive_share_by_k.png")
    plot_status_composition("figures/status_composition_by_k.png")
