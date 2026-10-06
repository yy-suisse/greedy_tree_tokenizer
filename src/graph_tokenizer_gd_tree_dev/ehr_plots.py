"""
Figures for the EHR section, from the tables of ehr_tokenization.py.

    from src.graph_tokenizer_gd_tree_dev.ehr_plots import plot_ehr_summary, plot_borrowed_strength
    # fig:k-sweep-real: coverage, cost, uniqueness and long tail vs the token budget k
    plot_ehr_summary(et.coverage(per_code), et.cost(per_code), et.uniqueness(per_code),
                     et.longtail(per_code), fname="figures/ehr_summary.pdf")
    # fig:borrowed-strength: events of each concept vs events of its rarest token, at one k
    fig, on_diag = plot_borrowed_strength(per_code, k=5000, fname="figures/borrowed_strength.pdf")
"""
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
OURS, TRUNC = "#2a78d6", "#eb6834"            # categorical slots 1-2 (validated, CVD dE 24.7)
# sequential blue ramp, steps 100 -> 700 (one hue, light = few, dark = many)
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
STYLE = {
    "ours":       dict(color=OURS,  lw=1.8, marker="o", ms=3.6, ls="-", zorder=3,
                       markeredgecolor="white", markeredgewidth=0.6),
    "ours_flat":  dict(color=OURS,  lw=1.4, marker="o", ms=3.0, ls=(0, (4, 2)), zorder=2,
                       mfc="white", markeredgecolor=OURS, markeredgewidth=0.8),
    "truncation": dict(color=TRUNC, lw=1.4, marker="s", ms=3.2, ls="-", zorder=2,
                       markeredgecolor="white", markeredgewidth=0.6),
}
LABEL = {
    "ours": "Ours ($\\lambda{=}0.9$), context tree",
    "ours_flat": "Ours ($\\lambda{=}0.9$), flat token set",
    "truncation": "Frequency truncation (top-$k$ concepts + UNK)",
}


def _rare_summary(lt, rare_bins=("<10", "10-99"), rare=10):
    """Pool the rare frequency bins: share of rare concepts represented (ours, truncation) and
    share of represented rare concepts whose rarest token occurs in at least `rare` events (ours)."""
    col_ge = f"pct_rarest_token_ge_{rare}"
    d = lt.filter(pl.col("freq_bin").cast(pl.String).is_in(list(rare_bins)))
    n, rep = pl.col("n_codes"), pl.col("n_codes") * pl.col("ours_pct_represented") / 100
    return (d.group_by("k").agg(
        ((n * pl.col("ours_pct_represented")).sum() / n.sum()).alias("ours_rep"),
        ((n * pl.col("trunc_pct_represented")).sum() / n.sum()).alias("trunc_rep"),
        ((rep * pl.col(col_ge)).sum() / rep.sum()).alias("ours_ge"),
        n.sum().alias("n_rare"),
    ).sort("k"))


def plot_ehr_summary(cov, cst, uniq, lt, rare_bins=("<10", "10-99"), rare=10, fname=None,
                     notes=None):
    rs = _rare_summary(lt, rare_bins, rare)
    n_rare = int(rs["n_rare"][0])
    panels = [
        # (title, [(table, column, style key)]); letters (a)-(f) match the paper caption
        ("(a) Concepts represented (%) $\\uparrow$",
         [(cov, "trunc_pct_codes", "truncation"), (cov, "ours_pct_codes", "ours")]),
        ("(b) Events represented (%) $\\uparrow$",
         [(cov, "trunc_pct_events", "truncation"), (cov, "ours_pct_events", "ours")]),
        # order follows the Results text: coverage, uniqueness, rare concepts, then the cost
        ("(c) Unique concepts (%) $\\uparrow$",
         [(uniq, "trunc_pct_codes", "truncation"), (uniq, "ours_flat_pct_codes", "ours_flat"),
          (uniq, "ours_tree_pct_codes", "ours")]),
        ("(d) Rare concepts (<100 events)\nrepresented (%) $\\uparrow$",
         [(rs, "trunc_rep", "truncation"), (rs, "ours_rep", "ours")]),
        (f"(e) Rare concepts whose rarest token\noccurs in $\\geq${rare} events (%) $\\uparrow$",
         [(rs, "ours_ge", "ours")]),
        ("(f) Tokens per event $\\downarrow$",
         [(cst, "trunc_tokens_per_event", "truncation"), (cst, "ours_tokens_per_event", "ours")]),
    ]
    k_all = sorted(cov["k"].to_list())
    k_ticks = [k for k in (250, 1000, 5000) if k_all[0] <= k <= k_all[-1]] + [k_all[-1]]
    k_lab = {k: (str(k) if k < 1000 else f"{k // 1000}k" if k % 1000 == 0 else f"{k / 1000:.1f}k")
             for k in k_ticks}

    plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9), constrained_layout=True)
    for i, (ax, (title, series)) in enumerate(zip(axes.flat, panels)):
        for tbl, col, key in series:
            t = tbl.sort("k")
            ax.plot(t["k"].to_list(), t[col].to_list(), **STYLE[key])
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(k_ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: k_lab.get(int(round(v)), "")))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_title(title, fontsize=7.5, color=INK, pad=5)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=2.5, width=0.6, labelsize=6.8)
        if i >= 3:
            ax.set_xlabel("Token budget $k$ (log scale)", fontsize=7, color=INK2)
        if notes and i in notes:                          # e.g. {1: "preview data"} while drafting
            ax.text(0.5, 0.5, notes[i], transform=ax.transAxes, ha="center", va="center",
                    fontsize=7, color=MUTED, alpha=0.9)

    # direct labels on the first panel, so series identity never relies on colour alone
    ax0, c0 = axes.flat[0], cov.sort("k")
    ax0.annotate("Ours", (c0["k"][1], c0["ours_pct_codes"][1]), xytext=(0, -10),
                 textcoords="offset points", fontsize=6.8, color=INK2, ha="left", va="top")
    i_mid = len(c0) // 2
    ax0.annotate("Truncation", (c0["k"][i_mid], c0["trunc_pct_codes"][i_mid]), xytext=(6, -4),
                 textcoords="offset points", fontsize=6.8, color=INK2, ha="left", va="top")

    handles = [Line2D([0], [0], **STYLE[k], label=LABEL[k]) for k in ("ours", "ours_flat", "truncation")]
    # legend above the panels; works on matplotlib < 3.7 (no "outside" loc) because
    # savefig(bbox_inches="tight") extends the saved area to include it
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               bbox_transform=fig.transFigure, ncol=3, frameon=False, fontsize=7.2, handlelength=3.0)
    if fname:
        fig.savefig(fname, dpi=300, bbox_inches="tight")
    return fig


def plot_borrowed_strength(per_code, k=5000, rare=10, fname=None):
    """
    fig:borrowed-strength. One point per represented concept at budget k:
      x = events of the concept, y = events of its rarest token (column rarest_token_occ).
    A token occurs in every event of every concept that uses it, so y >= x: concepts lie on or
    above the diagonal, and the height above it is the gain. Concepts ON the diagonal gain nothing.
    No title inside the plot (the caption carries it). Returns (fig, share of concepts on the
    diagonal), so the share can go into the text or caption.
    """
    d = per_code.filter((pl.col("k") == k) & ~pl.col("unknown") & (pl.col("occurrences") > 0))
    x = d["occurrences"].to_numpy().astype(float)
    y = d["rarest_token_occ"].to_numpy().astype(float)
    assert (y >= x).all(), "rarest token rarer than the concept: check the token table"
    on_diag = float((y == x).mean())

    plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(3.4, 2.9), constrained_layout=True)
    hb = ax.hexbin(x, y, xscale="log", yscale="log", gridsize=40, bins="log", cmap=SEQ_BLUE,
                   mincnt=1, linewidths=0.15, edgecolors="white")
    hi = max(x.max(), y.max()) * 1.5
    ax.set_xlim(0.7, hi)
    ax.set_ylim(0.7, hi)
    ax.plot([0.7, hi], [0.7, hi], color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=3)
    ax.axhline(rare, color=MUTED, lw=0.6, ls=(0, (1, 1.5)), zorder=3)
    ax.axvline(rare, color=MUTED, lw=0.6, ls=(0, (1, 1.5)), zorder=3)
    # direct labels: the diagonal and the 10-event lines
    mid = hi ** 0.5                                   # geometric middle of the diagonal
    ax.annotate("no gain ($y = x$)", xy=(mid, mid), xytext=(6, -4),
                textcoords="offset points", fontsize=6.6, color=INK2, ha="left", va="top")
    ax.annotate(f"{rare} events", xy=(hi / 2, rare), xytext=(0, 2), textcoords="offset points",
                fontsize=6.4, color=MUTED, ha="right", va="bottom")
    ax.set_aspect("equal", adjustable="box")          # same scale on both axes: a decade is a decade
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Events of the concept", color=INK2)
    ax.set_ylabel("Events of its rarest token", color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5, width=0.6, labelsize=6.8)
    cb = fig.colorbar(hb, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("Concepts per hexagon", color=INK2, fontsize=7)
    cb.ax.tick_params(labelsize=6.4, color=INK2, labelcolor=INK2)
    cb.outline.set_visible(False)
    if fname:
        fig.savefig(fname, dpi=400, bbox_inches="tight")
    return fig, on_diag
