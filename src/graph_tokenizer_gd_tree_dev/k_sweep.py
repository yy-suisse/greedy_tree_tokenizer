"""
Real-data k sweep: how the vocabulary budget k trades token frequency (data per token)
against representation cost (sequence length) and fidelity (coverage, uniqueness), for our
tokenizer and for frequency truncation (top-k codes + one UNK token), optionally under
simulated smaller datasets (binomial thinning of code occurrences).
"""
from collections import Counter

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from src.graph_tokenizer_gd_tree_dev.eval import build_context_trees, _iter_contexts
from src.graph_tokenizer_gd_tree_dev.tokenizer import signature


def _code_stats(tree):
    """Per code: flat token set, sequence cost with relations, partly-unknown flag."""
    ctxs = list(_iter_contexts(tree))
    per_ctx = [len({t for t, _ in c.tokens}) for c in ctxs]       # dedup tokens within a context
    tokens = {t for c in ctxs for t, _ in c.tokens}
    seq_cost = sum(per_ctx) + (len(ctxs) - 1)                     # tokens + one marker per relation subcontext
    partly_unknown = any(c.uncovered for c in ctxs)
    return tokens, seq_cost, partly_unknown


def _reuse_stats(tok_sets, T):
    """Frequency-free token sharing: how many concepts use each token (each concept counted once)."""
    use = Counter(t for s in tok_sets for t in s)
    used = np.array([use[t] for t in T if use.get(t, 0) > 0])
    return {
        "median_concepts_per_token": float(np.median(used)) if len(used) else 0.0,
        "mean_concepts_per_token": float(used.mean()) if len(used) else 0.0,
        "pct_tokens_shared": 100 * float((used >= 2).mean()) if len(used) else 0.0,
        "median_tokens_per_concept": float(np.median([len(s) for s in tok_sets])) if tok_sets else 0.0,
    }


def _freq_stats(counts, min_occ):
    counts = np.asarray(counts)
    out = {"median_occ_per_token": float(np.median(counts)) if len(counts) else 0.0}
    for m in min_occ:
        out[f"pct_tokens_ge_{m}"] = 100 * float((counts >= m).mean()) if len(counts) else 0.0
    return out


def sweep_k(Ks, ranked_tokens, codes, code_freq, adj, D, id_to_label=None,
            fractions=(0.1, 0.25, 0.5, 1.0), min_occ=(10, 100), seed=0):
    """
    Ks            : token budgets to test
    ranked_tokens : greedy selection order (e.g. the 'token' column of the lambda = 0.9 run)
    codes         : codes to tokenize (e.g. mapped_ids_overlap)
    code_freq     : {code: number of occurrences in the data} (event volume, e.g. 'len')
    fractions     : simulated share of the data available for training (binomial thinning)
    Returns one row per (method, k, fraction).
    """
    rng = np.random.default_rng(seed)
    codes = list(codes)
    f_full = np.array([code_freq.get(c, 0) for c in codes], dtype=np.int64)
    thinned = {p: (f_full if p == 1.0 else rng.binomial(f_full, p)) for p in fractions}
    rows = []

    for k in Ks:
        T = set(ranked_tokens[:k])
        trees = build_context_trees(codes, adj, T, D, id_to_label)
        stats = [_code_stats(trees[c]) for c in codes]
        tok_sets = [s[0] for s in stats]
        seq_cost = np.array([s[1] for s in stats], dtype=float)
        partly_unk = np.array([s[2] for s in stats])
        represented = np.array([len(s) > 0 for s in tok_sets])

        sig = [signature(trees[c]) for c in codes]
        sig_n = Counter(sig)
        flat_n = Counter(frozenset(s) for s in tok_sets)
        unique_rel = 100 * np.mean([sig_n[s] == 1 for s in sig])
        unique_flat = 100 * np.mean([flat_n[frozenset(s)] == 1 and len(s) > 0 for s in tok_sets])
        reuse = _reuse_stats(tok_sets, T)
        pct_self = 100 * np.mean([c in T for c in codes])              # concept is itself a token

        for p, f in thinned.items():
            total = f.sum()
            # ---- ours: token occurrences = sum of occurrences of the codes that use the token
            cnt = Counter()
            for s, n in zip(tok_sets, f):
                if n:
                    for t in s:
                        cnt[t] += int(n)
            tok_counts = [cnt.get(t, 0) for t in T]          # unused tokens count as 0
            rows.append({
                "method": "ours", "k": k, "fraction": p,
                **_freq_stats(tok_counts, min_occ),
                "pct_codes_represented": 100 * represented.mean(),
                "pct_occ_represented": 100 * f[represented].sum() / total if total else 0.0,
                "seq_cost_per_event": float((seq_cost * f).sum() / total) if total else 0.0,
                "pct_codes_partly_unknown": 100 * partly_unk.mean(),
                "unique_rate_relations": unique_rel,
                "unique_rate_flat": unique_flat,
                "n_unused_tokens": sum(1 for t in T if cnt.get(t, 0) == 0),
                "pct_concepts_self_token": pct_self,
                **reuse,
            })
            # ---- frequency truncation: top-k codes of the (thinned) training data + one UNK
            order = np.argsort(-f)
            kept = np.zeros(len(codes), dtype=bool)
            kept[order[:k]] = True
            rows.append({
                "method": "truncation", "k": k, "fraction": p,
                **_freq_stats(f[kept], min_occ),
                "pct_codes_represented": 100 * kept.mean(),
                "pct_occ_represented": 100 * f[kept].sum() / total if total else 0.0,
                "seq_cost_per_event": 1.0,
                "pct_codes_partly_unknown": 0.0,
                "unique_rate_relations": 100 * kept.mean(),
                "unique_rate_flat": 100 * kept.mean(),
                "n_unused_tokens": int((f[kept] == 0).sum()),
                "pct_concepts_self_token": 100 * kept.mean(),
                "median_concepts_per_token": 1.0,                   # UNK excluded: not a semantic token
                "mean_concepts_per_token": 1.0,
                "pct_tokens_shared": 0.0,
                "median_tokens_per_concept": 1.0,
            })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
LINE = "#0d366b"
REF = "#898781"                                                 # raw-concept reference
METHOD_LS = {"ours": "-", "truncation": (0, (3, 2))}
K_TICKS = [250, 1000, 2000, 5000, 10000]


def _k_label(v):
    return f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"


def plot_sweep(res, m=100, show_freq=False, show_events=True, fname=None):
    """
    Full data only (fraction == 1.0). Solid = ours, dashed = truncation.
    Panels: [token frequency (share of tokens seen >= m times, raw concepts dotted)] if show_freq,
    then tokens per event, concepts represented, [events represented] if show_events,
    unique representations. Letters follow the order. An event is represented if its concept
    has at least one token (for truncation: if its concept is among the top-k).
    m must be one of the min_occ values passed to sweep_k.
    """
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#c3c2b7",
                         "xtick.color": INK2, "ytick.color": INK2})
    full = res.filter(pl.col("fraction") == 1.0) if "fraction" in res.columns else res
    by_method = {meth: full.filter(pl.col("method") == meth).sort("k") for meth in METHOD_LS}

    panels = [("seq_cost_per_event", "Tokens per event"),
              ("pct_codes_represented", "Concepts represented (%)"),
              ("unique_rate_relations", "Unique representations (%)")]
    if show_events:
        panels.insert(2, ("pct_occ_represented", "Events represented (%)"))
    if show_freq:
        col_a = f"pct_tokens_ge_{m}"
        if col_a not in full.columns:
            raise ValueError(f"m={m} not computed: rerun sweep_k with {m} in min_occ")
        panels.insert(0, (col_a, f"Tokens seen $\\geq${m:,} times (%)"))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(1.8 * n, 2.3), constrained_layout=True)
    for i, (ax, (col, title)) in enumerate(zip(axes, panels)):
        for meth, ls in METHOD_LS.items():
            d = by_method[meth]
            ax.plot(d["k"], d[col], color=LINE, lw=1.4, ls=ls, label=meth)
        ax.set_title(f"({chr(97 + i)}) {title}", fontsize=8, color=INK)

    if show_freq:
        last = by_method["truncation"].tail(1)
        if last.height and last["pct_codes_represented"][0] >= 100:
            ref = last[panels[0][0]][0]
            axes[0].axhline(ref, color=REF, lw=1.0, ls=":")
            axes[0].text(K_TICKS[0], ref + 2.5, "raw concepts", color=MUTED, fontsize=6.5, va="bottom")
        axes[0].set_ylim(0, 105)
    for ax, (col, _) in zip(axes, panels):
        if col in ("pct_codes_represented", "pct_occ_represented", "unique_rate_relations"):
            ax.set_ylim(-3, 105)                                   # same 0-100 scale for shares
    seq_ax = axes[1] if show_freq else axes[0]
    seq_ax.legend(frameon=False, fontsize=6.5, labelcolor=INK2, loc="upper right")

    for ax in axes:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(K_TICKS))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _k_label(v)))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Token budget $k$", color=INK2)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    if fname:
        fig.savefig(fname, bbox_inches="tight", dpi=400)
    return fig


# ---------------------------------------------------------------------------------------------
# usage (notebook 6, after mapped_ids_overlap, adj, D, id_to_label exist)
# ---------------------------------------------------------------------------------------------
# ranked = pl.read_parquet(f"{config.HERODataVocabsTest.greedy_candidate_path}0.9.parquet")["token"].to_list()
# code_freq = dict(zip(df_vocab_freq["code"], df_vocab_freq["len"]))          # event volume
# Ks = [250, 500, 1000, 2000, 3500, 5000, 7500, 10000, len(mapped_ids_overlap)]
# res = sweep_k(Ks, ranked, mapped_ids_overlap, code_freq, adj, D, id_to_label,
#               fractions=(1.0,), min_occ=(10, 100, 1000))
# res.write_csv("k_sweep_real_data.csv")
# plot_sweep(res, freq_panel=False, fname="k_sweep_real_data.pdf")  # main text, 3 panels
# plot_sweep(res, m=100, fname="k_sweep_real_data_4p.pdf")           # with token-frequency panel
# plot_sweep(res, m=1000, fname="k_sweep_real_data_m1000.pdf")     # appendix