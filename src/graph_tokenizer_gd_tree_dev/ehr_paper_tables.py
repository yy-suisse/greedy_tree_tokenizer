"""
LaTeX rows for the two EHR tables of the paper, computed straight from per_code
(output of ehr_tokenization.tokenize_all_k), so that every number quoted in the text
can be found in a table.

    import importlib, src.graph_tokenizer_gd_tree_dev.ehr_paper_tables as ept
    importlib.reload(ept)
    print(ept.ehr_rows(per_code))                 # tab:ehr       one row per k
    print(ept.longtail_rows(per_code, k=5000))    # tab:longtail  one row per frequency bin

Definitions (same as ehr_tokenization.py and ehr_plots.py):
  represented   the concept has at least one token (not unknown)
  uncovered     at least one branch of its context tree ends without a token
  unique        represented, and no other concept has the same tree (or flat token set)
  rare          fewer than 100 events
  rarest token  events of the least frequent token of a represented concept
"""
import polars as pl

RARE = 100          # a concept is rare below this number of events
MIN_EVENTS = 10     # "every token occurs in at least 10 events"


def _pct(v):
    """Percentage with one decimal; never rounds a non-zero value to 0.0 or a partial one to 100.0."""
    if v is None:
        return "n/a"
    r = round(v, 1)
    if v > 0 and r == 0:
        return "$<$0.1"
    if v < 100 and r >= 100:
        return "$>$99.9"
    return f"{r:.1f}"


def _share(series_bool):
    """100 * mean of a boolean Series, None if empty."""
    m = series_bool.mean()
    return None if m is None else 100 * m


def _num(v, nd=0):
    return "n/a" if v is None else f"{v:,.{nd}f}"


def ehr_rows(per_code):
    """tab:ehr: k | concepts (ours, trunc) | events (ours, trunc) | uncovered | unique (tree, flat,
    tree weighted by events) | rare concepts whose every token has >= 10 events | tokens per event."""
    occ = pl.col("occurrences").cast(pl.Float64)
    rep = ~pl.col("unknown")
    rare = (pl.col("occurrences") < RARE) & rep
    t = (per_code.group_by("k").agg(
        (100 * rep.mean()).alias("c_ours"),
        (100 * pl.col("trunc_kept").mean()).alias("c_trunc"),
        (100 * occ.filter(rep).sum() / occ.sum()).alias("e_ours"),
        (100 * occ.filter(pl.col("trunc_kept")).sum() / occ.sum()).alias("e_trunc"),
        (100 * pl.col("partly_unknown").mean()).alias("uncov"),
        (100 * pl.col("unique_tree").mean()).alias("u_tree"),
        (100 * pl.col("unique_flat").mean()).alias("u_flat"),
        (100 * occ.filter(pl.col("unique_tree")).sum() / occ.sum()).alias("u_tree_ev"),
        (100 * (pl.col("rarest_token_occ") >= MIN_EVENTS).filter(rare).mean()).alias("rare_ge"),
        ((pl.col("seq_cost") * occ).sum() / occ.sum()).alias("tok_ev"),
    ).sort("k"))
    lines = []
    for r in t.iter_rows(named=True):
        cells = [_num(r["k"])] + [_pct(r[c]) for c in
                 ("c_ours", "c_trunc", "e_ours", "e_trunc", "uncov", "u_tree", "u_flat",
                  "u_tree_ev", "rare_ge")] + [_num(r["tok_ev"], 1)]
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines)


BINS = [(0, 10, r"$< 10$"), (10, 100, r"$[10, 100)$"), (100, 1000, r"$[100, 1000)$"),
        (1000, None, r"$\ge 1000$")]


def longtail_rows(per_code, k=5000):
    """tab:longtail at one k: events of the concept | concepts | share of all events | truncation
    represented | unique tree | rarest token: median events, median gain, >= 10 events, no gain."""
    d = per_code.filter(pl.col("k") == k)
    total = d["occurrences"].cast(pl.Float64).sum()
    lines = []
    for lo, hi, label in BINS:
        m = pl.col("occurrences") >= lo
        if hi is not None:
            m = m & (pl.col("occurrences") < hi)
        b = d.filter(m)
        r = b.filter(~pl.col("unknown"))
        rt = r["rarest_token_occ"].cast(pl.Float64)
        oc = r["occurrences"].cast(pl.Float64)
        cells = [
            label,
            _num(b.height),
            _pct(100 * b["occurrences"].cast(pl.Float64).sum() / total),
            _pct(_share(b["trunc_kept"])),
            _pct(_share(b["unique_tree"])),
            _num(rt.median()),
            _num((rt / oc).median() if r.height else None, 1),
            _pct(_share(rt >= MIN_EVENTS)),
            _pct(_share(rt == oc)),
        ]
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines)
