"""
Figure 9 (Section 4.4): our tokenizer (lambda = 0.9) vs frequency truncation on the HERO data,
three panels against the token budget k.
    (a) concepts (solid) and events (dashed) represented
    (b) rare concepts (< 100 events) represented
    (c) tokens per event

Inputs are the tables written by the (final)ehr_application_analysis notebook:
    ehr/coverage.csv, ehr/longtail.csv, ehr/cost.csv

    from plot_ehr_rare_summary import plot_ehr_rare_summary
    plot_ehr_rare_summary("D:/greedy_graph_data/final_hero/",
                          fname="D:/greedy_graph_data/final_hero/figures/ehr_rare_summary.pdf")
"""
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
OURS, TRUNC = "#2a78d6", "#eb6834"
MARK = {"ours": dict(color=OURS, marker="o", ms=3.6, markeredgecolor="white", markeredgewidth=0.6),
        "trunc": dict(color=TRUNC, marker="s", ms=3.2, markeredgecolor="white", markeredgewidth=0.6)}
SOLID, DASHED = "-", (0, (4, 2))


def rare_by_k(lt, rare_bins=("<10", "10-99")):
    """Share (%) of rare concepts represented, pooled over the rare frequency bins."""
    d = lt.filter(pl.col("freq_bin").cast(pl.String).is_in(list(rare_bins)))
    n = pl.col("n_codes")
    return (d.group_by("k").agg(
        ((n * pl.col("ours_pct_represented")).sum() / n.sum()).alias("ours"),
        ((n * pl.col("trunc_pct_represented")).sum() / n.sum()).alias("trunc"),
    ).sort("k"))


def plot_ehr_rare_summary(root, fname=None):
    cov = pl.read_csv(f"{root}ehr/coverage.csv").sort("k")
    cst = pl.read_csv(f"{root}ehr/cost.csv").sort("k")
    rare = rare_by_k(pl.read_csv(f"{root}ehr/longtail.csv"))
    k_all = cov["k"].to_list()
    k_ticks = [250, 1000, 5000, k_all[-1]]
    k_lab = {k: (str(k) if k < 1000 else f"{k // 1000}k" if k % 1000 == 0 else f"{k / 1000:.1f}k")
             for k in k_ticks}

    plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    a, b, c = axes

    # (a) concepts (solid) and events (dashed) represented
    a.plot(k_all, cov["trunc_pct_events"], ls=DASHED, lw=1.4, zorder=2, **MARK["trunc"])
    a.plot(k_all, cov["trunc_pct_codes"], ls=SOLID, lw=1.4, zorder=2, **MARK["trunc"])
    a.plot(k_all, cov["ours_pct_events"], ls=DASHED, lw=1.8, zorder=3, **MARK["ours"])
    a.plot(k_all, cov["ours_pct_codes"], ls=SOLID, lw=1.8, zorder=4, **MARK["ours"])
    a.set_title("(a) Concepts and events represented (%) $\\uparrow$", fontsize=7.5, color=INK, pad=5)
    a.set_ylim(-4, 116)
    a.set_yticks([0, 20, 40, 60, 80, 100])
    a.annotate("Ours: concepts and events", (k_all[0], 100), xytext=(0, 4), textcoords="offset points",
               fontsize=6.6, color=INK2, ha="left", va="bottom")
    a.annotate("Truncation: events", (1000, cov["trunc_pct_events"][2]), xytext=(2, -9),
               textcoords="offset points", fontsize=6.6, color=INK2, ha="left", va="top")
    i = 4
    a.annotate("Truncation: concepts", (k_all[i], cov["trunc_pct_codes"][i]), xytext=(-6, 6),
               textcoords="offset points", fontsize=6.6, color=INK2, ha="right", va="bottom")

    # (b) rare concepts represented
    b.plot(rare["k"], rare["trunc"], ls=SOLID, lw=1.4, zorder=2, **MARK["trunc"])
    b.plot(rare["k"], rare["ours"], ls=SOLID, lw=1.8, zorder=3, **MARK["ours"])
    b.set_title("(b) Rare concepts (<100 events)\nrepresented (%) $\\uparrow$", fontsize=7.5, color=INK, pad=5)
    b.set_ylim(-4, 116)
    b.set_yticks([0, 20, 40, 60, 80, 100])

    # (c) tokens per event
    c.plot(cst["k"], cst["trunc_tokens_per_event"], ls=SOLID, lw=1.4, zorder=2, **MARK["trunc"])
    c.plot(cst["k"], cst["ours_tokens_per_event"], ls=SOLID, lw=1.8, zorder=3, **MARK["ours"])
    c.set_title("(c) Tokens per event $\\downarrow$", fontsize=7.5, color=INK, pad=5)
    c.set_ylim(0, None)

    for ax in axes:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(k_ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: k_lab.get(int(round(v)), "")))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=2.5, width=0.6, labelsize=6.8)
        ax.set_xlabel("Token budget $k$ (log scale)", fontsize=7, color=INK2)

    handles = [Line2D([0], [0], ls=SOLID, lw=1.8, **MARK["ours"], label="Ours ($\\lambda{=}0.9$)"),
               Line2D([0], [0], ls=SOLID, lw=1.4, **MARK["trunc"], label="Frequency truncation (top-$k$ + UNK)"),
               Line2D([0], [0], ls=SOLID, lw=1.2, color=MUTED, label="concepts"),
               Line2D([0], [0], ls=DASHED, lw=1.2, color=MUTED, label="events (panel a)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               bbox_transform=fig.transFigure, ncol=4, frameon=False, fontsize=7.2, handlelength=3.0)
    if fname:
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        if fname.endswith(".pdf"):
            fig.savefig(fname[:-4] + ".png", dpi=300, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    import sys
    plot_ehr_rare_summary(sys.argv[1], fname=sys.argv[2])
