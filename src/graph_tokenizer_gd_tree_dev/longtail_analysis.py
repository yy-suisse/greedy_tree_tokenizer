import numpy as np
import polars as pl
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def prepare(df_tokenized, df_freq, freq_col="len"):
    """
    One row per original code: its frequency and its token set (empty list = no token at all).
    Token frequency = sum of the frequencies of the codes whose representation contains it.
    With event volume ("len") this sum is exact: every occurrence of a code is one occurrence
    of each of its tokens. (With num_patients it over-counts patients shared between codes.)
    """
    codes = (df_tokenized
             .join(df_freq.select("code", freq_col), left_on="original", right_on="code", how="inner")
             .rename({freq_col: "code_freq"})
             .with_columns(pl.col("tokenized_list").list.len().alias("n_tokens")))
    pairs = (codes.explode("tokenized_list").drop_nulls("tokenized_list")
             .rename({"tokenized_list": "token"})
             .select("original", "token", "code_freq"))
    tok = (pairs.group_by("token")
                .agg(pl.col("code_freq").sum().alias("tok_freq"),
                     pl.len().alias("n_codes_using")))
    return codes, pairs, tok


def vocab_table(codes, tok, k, rare=10):
    """Same vocabulary size k: raw codes vs frequency truncation (top-k + UNK) vs ours."""
    n_codes, total = codes.height, codes["code_freq"].sum()
    top_k = codes.sort("code_freq", descending=True).head(k)
    covered = codes.filter(pl.col("n_tokens") > 0)

    def row(name, item_freq, codes_repr, occ_repr):
        f = np.asarray(item_freq)
        return {"vocabulary": name, "size": len(f),
                f"% items < {rare}": round(100 * (f < rare).mean(), 1),
                "median item freq": float(np.median(f)),
                "% codes represented": round(100 * codes_repr, 1),
                "% occurrences represented": round(100 * occ_repr, 2)}

    return pl.DataFrame([
        row("Raw codes", codes["code_freq"], 1.0, 1.0),
        row(f"Frequency top-{k} + UNK", top_k["code_freq"], k / n_codes, top_k["code_freq"].sum() / total),
        row(f"Ours (k = {k})", tok["tok_freq"], covered.height / n_codes, covered["code_freq"].sum() / total),
    ])


def borrowed_strength(codes, pairs, tok, k, rare=10,
                      bins=(1, 10, 100, 1000, float("inf"))):
    """
    Per code: frequency of its rarest token ("weakest link"). Grouped by how frequent the
    code itself is. The key rows are the rare codes: does tokenization give them frequent tokens?
    """
    top_k_codes = set(codes.sort("code_freq", descending=True).head(k)["original"].to_list())
    weakest = (pairs.join(tok, on="token")
                    .group_by("original")
                    .agg(pl.col("tok_freq").min().alias("min_tok_freq")))
    df = (codes.join(weakest, on="original", how="left")
               .with_columns(pl.col("original").is_in(list(top_k_codes)).alias("in_top_k")))
    labels = [f"[{int(lo)}, {'inf' if hi == float('inf') else int(hi)})" for lo, hi in zip(bins[:-1], bins[1:])]
    rows = []
    for (lo, hi), lab in zip(zip(bins[:-1], bins[1:]), labels):
        b = df.filter((pl.col("code_freq") >= lo) & (pl.col("code_freq") < hi))
        if b.height == 0:
            continue
        rep = b.filter(pl.col("n_tokens") > 0)
        rows.append({
            "code freq bin": lab, "n codes": b.height,
            "% kept by top-k truncation": round(100 * b["in_top_k"].mean(), 1),
            "% represented by ours": round(100 * rep.height / b.height, 1),
            "median code freq": float(b["code_freq"].median()),
            "median freq of rarest token": float(rep["min_tok_freq"].median()) if rep.height else None,
            "median gain (rarest token / code)": float((rep["min_tok_freq"] / rep["code_freq"]).median()) if rep.height else None,
            "% with no gain (rarest token = code)": round(100 * (rep["min_tok_freq"] == rep["code_freq"]).mean(), 1) if rep.height else None,
            f"% whose rarest token >= {rare}": round(100 * (rep["min_tok_freq"] >= rare).mean(), 1) if rep.height else None,
        })
    return pl.DataFrame(rows), df


def rare_token_diagnosis(codes, tok, rare=10):
    """Which tokens stay rare? Typically a rare mapped code selected as its own token."""
    mapped = set(codes["original"].to_list())
    r = tok.filter(pl.col("tok_freq") < rare).with_columns(
        pl.col("token").is_in(list(mapped)).alias("is_mapped_code"))
    return {"n rare tokens": r.height,
            "% that are mapped codes themselves": round(100 * r["is_mapped_code"].mean(), 1) if r.height else None,
            "% used by a single code": round(100 * (r["n_codes_using"] == 1).mean(), 1) if r.height else None}


def unknown_codes(codes):
    """Codes with no token at all: how many, how much usage, how frequent."""
    u = codes.filter(pl.col("n_tokens") == 0)
    return {"n codes without any token": u.height,
            "% of codes": round(100 * u.height / codes.height, 1),
            "% of occurrences": round(100 * u["code_freq"].sum() / codes["code_freq"].sum(), 2),
            "median freq of these codes": float(u["code_freq"].median()) if u.height else None}


def plot_borrowed_strength(df, rare=10, fname=None):
    """x = frequency of the code, y = frequency of its rarest token.
    By construction y >= x (a token is seen at least as often as any code using it), so points
    can only lie on or above the diagonal. What matters is the distance above it (borrowed
    strength) and the share lying exactly ON it (codes that are their own, unshared token)."""
    d = df.filter(pl.col("n_tokens") > 0)
    x, y = d["code_freq"].to_numpy(), d["min_tok_freq"].to_numpy()
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#c3c2b7",
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(3.4, 3.0), constrained_layout=True)
    hb = ax.hexbin(x, y, xscale="log", yscale="log", gridsize=40, bins="log", cmap="Blues", mincnt=1)
    lim = [1, max(x.max(), y.max()) * 1.5]
    ax.plot(lim, lim, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.axhline(rare, color=MUTED, lw=0.6, ls=(0, (1, 1.5)))
    ax.axvline(rare, color=MUTED, lw=0.6, ls=(0, (1, 1.5)))
    ax.set_xlabel("Occurrences of the code", color=INK2)
    ax.set_ylabel("Occurrences of its rarest token", color=INK2)
    on_diag = (y == x).mean()
    ax.set_title(f"{on_diag:.0%} of represented codes gain nothing (on the diagonal)",
                 fontsize=7.5, color=INK)
    cb = fig.colorbar(hb, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("Number of codes", color=INK2)
    cb.outline.set_visible(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if fname:
        fig.savefig(fname, bbox_inches="tight", dpi=400)
    return fig


# ------------------------------------------------------------------
# usage (notebook 6, after df_tokenized is built)
# ------------------------------------------------------------------
# K, RARE = 5000, 10
# codes, pairs, tok = prepare(df_tokenized, df_vocab_freq, freq_col="len")   # event volume: exact sums
#
# print(unknown_codes(codes))                                   # codes that get no token at all
# print(vocab_table(codes, tok, k=K, rare=RARE))                # raw vs top-k truncation vs ours
# table, per_code = borrowed_strength(codes, pairs, tok, k=K, rare=RARE)
# print(table)                                                  # the key evidence, per code-frequency bin
# print(rare_token_diagnosis(codes, tok, rare=RARE))            # why some tokens stay rare
# plot_borrowed_strength(per_code, rare=RARE, fname="borrowed_strength.pdf")
#
# # are the token-less codes even in the graph?
# no_tok = codes.filter(pl.col("n_tokens") == 0)["original"].to_list()
# print(sum(c not in combined_subgraphs for c in no_tok), "of", len(no_tok), "are not nodes of the subgraph")