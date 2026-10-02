"""
EHR application of the tokenizer, from scratch.

Step 1  tokenize_all_k(...)  -> per_code: one row per (k, code) with
          tokens, unknown flags, sequence cost, signature id, occurrences in the data,
          and the frequency-truncation baseline (is the code among the k most frequent?).
Step 2  four analyses, each returning one tidy table (ours vs truncation):
          coverage(per_code)    code coverage and event coverage
          cost(per_code)        tokens per event, total tokens in the corpus
          uniqueness(per_code)  codes with a representation no other code shares (tree / flat set)
          longtail(per_code)    rare-code expressivity, by code-frequency bin
        plus token_stats(per_code): how often each token occurs in the data,
        and distribution_stats(per_code): skew of the token frequency distribution under four
        vocabularies (raw codes, truncation + UNK, truncation with the tail removed, ours).

Truncation baseline throughout: the k most frequent codes keep their own token, every other
code becomes one shared UNK token.
"""
import numpy as np
import polars as pl
from tqdm import tqdm

from src.graph_tokenizer_gd_tree_dev.eval import build_context_trees, _iter_contexts
from src.graph_tokenizer_gd_tree_dev.tokenizer import signature


# ---------------------------------------------------------------------------------------------
# 0. data: one row per code
# ---------------------------------------------------------------------------------------------
def code_frequencies(df_vocab_freq, freq_col="len"):
    """Total occurrences (event volume) per code. Sums over every row of a code (e.g. one row
    per date), so the result has exactly one row per code."""
    # cast before summing: counts often come as UInt32 (e.g. from pl.len()), and 100 * a sum of
    # hundreds of millions of events overflows 32-bit integers silently
    df = df_vocab_freq.group_by("code").agg(pl.col(freq_col).cast(pl.Int64).sum().alias("occurrences"))
    assert df["code"].n_unique() == df.height
    return df


# ---------------------------------------------------------------------------------------------
# 1. tokenized version of every code at every k
# ---------------------------------------------------------------------------------------------
def _record(code, tree, T):
    """Everything we need from one context tree, as plain Python values (no Context objects
    in the DataFrame: polars would store them as slow, fragile Object columns)."""
    ctxs = list(_iter_contexts(tree))
    tok_per_ctx = [{t for t, _ in c.tokens} for c in ctxs]       # tokens are deduplicated per context
    tokens = sorted(set().union(*tok_per_ctx))
    unknown = len(tokens) == 0
    return {
        "code": code,
        "tokens": tokens,                                         # flat token set (relations ignored)
        "n_tokens": len(tokens),
        "unknown": unknown,                                       # no token at all -> UNK
        "partly_unknown": any(c.uncovered for c in ctxs),         # >= 1 branch without a token
        "n_contexts": len(ctxs),
        # sequence cost of one occurrence: distinct tokens of each context + one marker per
        # relation subcontext; an unknown code costs one UNK token
        "seq_cost": 1 if unknown else sum(len(s) for s in tok_per_ctx) + len(ctxs) - 1,
        "is_self_token": code in T,                               # the code itself was selected
    }


def tokenize_all_k(codes, ranked, Ks, adj, D, df_freq, id_to_label=None):
    """
    codes   : codes to tokenize (e.g. mapped_ids_overlap)
    ranked  : greedy selection order (token column of the lambda run)
    df_freq : output of code_frequencies (columns code, occurrences)
    Returns per_code, one row per (k, code).
    """
    frames = []
    for K in tqdm(Ks):
        T = set(ranked[:K])
        trees = build_context_trees(codes, adj, T, D, id_to_label)
        sig_ids, flat_ids = {}, {}                                # representation -> integer id, per k
        recs = []
        for c, tree in trees.items():
            r = _record(c, tree, T)
            r["sig_id"] = sig_ids.setdefault(signature(tree), len(sig_ids))
            r["flat_id"] = flat_ids.setdefault(frozenset(r["tokens"]), len(flat_ids))
            recs.append(r)
        frames.append(pl.DataFrame(recs, infer_schema_length=None).with_columns(pl.lit(K).alias("k")))

    per_code = (
        pl.concat(frames)
        .join(df_freq, on="code", how="left")
        .with_columns(pl.col("occurrences").fill_null(0).cast(pl.Int64))
        .with_columns(
            # truncation baseline: the K most frequent codes are kept, all others share UNK
            (pl.col("occurrences").rank("ordinal", descending=True).over("k") <= pl.col("k"))
                .alias("trunc_kept"),
            # uniqueness: a represented code whose representation no other code shares
            ((pl.col("code").count().over(["k", "sig_id"]) == 1) & ~pl.col("unknown"))
                .alias("unique_tree"),
            ((pl.col("code").count().over(["k", "flat_id"]) == 1) & ~pl.col("unknown"))
                .alias("unique_flat"),
        )
    )
    if id_to_label is not None:
        per_code = per_code.with_columns(
            pl.col("code").replace_strict(id_to_label, default=None).alias("label"))

    # rarest token of each code: the token of its representation seen least often in the data
    tok = token_table(per_code)
    rarest = (per_code.select("k", "code", "tokens").explode("tokens").drop_nulls("tokens")
              .join(tok, left_on=["k", "tokens"], right_on=["k", "token"])
              .group_by(["k", "code"]).agg(pl.col("token_occ").min().alias("rarest_token_occ")))
    return (per_code.join(rarest, on=["k", "code"], how="left")
                    .sort(["k", "occurrences"], descending=[False, True]))


def token_table(per_code):
    """One row per (k, used token): occurrences in the data (sum over the codes that use it,
    each occurrence of a code is one occurrence of each of its tokens) and number of codes using it."""
    return (per_code.select("k", "tokens", "occurrences").explode("tokens").drop_nulls("tokens")
            .group_by(["k", "tokens"])
            .agg(pl.col("occurrences").sum().alias("token_occ"), pl.len().alias("n_codes_using"))
            .rename({"tokens": "token"}))


# ---------------------------------------------------------------------------------------------
# 2. analyses (one row per k unless stated otherwise)
# ---------------------------------------------------------------------------------------------
def _pct(expr):
    return 100 * expr.mean()


def _pct_events(mask):
    occ = pl.col("occurrences").cast(pl.Float64)                  # float: no integer overflow
    return 100 * occ.filter(mask).sum() / occ.sum()


def coverage(per_code):
    """Code coverage (share of distinct codes) and event coverage (share of data volume)."""
    represented = ~pl.col("unknown")
    return (per_code.group_by("k").agg(
        _pct(represented).alias("ours_pct_codes"),
        _pct_events(represented).alias("ours_pct_events"),
        _pct(pl.col("trunc_kept")).alias("trunc_pct_codes"),
        _pct_events(pl.col("trunc_kept")).alias("trunc_pct_events"),
        _pct(pl.col("partly_unknown")).alias("ours_pct_codes_partly_unknown"),
    ).sort("k"))


def cost(per_code):
    """Tokens needed to encode the data. Truncation always uses one token per event."""
    w = pl.col("occurrences").cast(pl.Int64)
    return (per_code.group_by("k").agg(
        w.sum().alias("events"),
        (pl.col("seq_cost") * w).sum().alias("ours_total_tokens"),
        ((pl.col("seq_cost") * w).sum() / w.sum()).alias("ours_tokens_per_event"),
        pl.col("seq_cost").median().alias("ours_tokens_per_code_median"),
        pl.col("n_tokens").median().alias("ours_distinct_tokens_per_code_median"),
        pl.lit(1.0).alias("trunc_tokens_per_event"),
    ).sort("k"))


def uniqueness(per_code):
    """Share of codes (and of events) whose representation is unique, and number of distinct
    representations. Truncation: each kept code is unique, all dropped codes share UNK."""
    return (per_code.group_by("k").agg(
        _pct(pl.col("unique_tree")).alias("ours_tree_pct_codes"),
        _pct(pl.col("unique_flat")).alias("ours_flat_pct_codes"),
        _pct(pl.col("trunc_kept")).alias("trunc_pct_codes"),
        _pct_events(pl.col("unique_tree")).alias("ours_tree_pct_events"),
        _pct_events(pl.col("trunc_kept")).alias("trunc_pct_events"),
        pl.col("sig_id").n_unique().alias("ours_tree_n_distinct"),
        pl.col("flat_id").n_unique().alias("ours_flat_n_distinct"),
        (pl.col("trunc_kept").sum() + (~pl.col("trunc_kept")).any().cast(pl.UInt32))
            .alias("trunc_n_distinct"),
    ).sort("k"))


def token_stats(per_code, min_occ=100):
    """How often tokens occur in the data (training signal per token). Unused tokens are the
    selected tokens that no code reaches; count = k - number of used tokens."""
    tok = token_table(per_code)
    return (tok.group_by("k").agg(
        pl.len().alias("n_used_tokens"),
        pl.col("token_occ").median().alias("median_token_occ"),
        _pct(pl.col("token_occ") >= min_occ).alias(f"pct_tokens_ge_{min_occ}"),
        _pct(pl.col("n_codes_using") >= 2).alias("pct_tokens_shared"),
        pl.col("n_codes_using").median().alias("median_codes_per_token"),
    ).with_columns((pl.col("k") - pl.col("n_used_tokens")).alias("n_unused_tokens")).sort("k"))


def longtail(per_code, breaks=(10, 100, 1000), rare=10):
    """
    Rare-code expressivity, per k and code-frequency bin.
      represented / unique : can the code be expressed, and told apart from the others?
      rarest token         : occurrences of the least frequent token of its representation
                             (computed over represented codes only)
      gain                 : rarest token occurrences / code occurrences (borrowed strength)
      no gain              : the rarest token is as rare as the code (typically: the code is its
                             own, unshared token)
    """
    labels = [f"<{breaks[0]}"] + [f"{lo}-{hi - 1}" for lo, hi in zip(breaks[:-1], breaks[1:])] \
             + [f">={breaks[-1]}"]
    rt, occ = pl.col("rarest_token_occ"), pl.col("occurrences")
    return (per_code
            .with_columns(occ.cut(list(breaks), labels=labels, left_closed=True).alias("freq_bin"))
            .group_by(["k", "freq_bin"]).agg(
                pl.len().alias("n_codes"),
                occ.min().alias("_min_occ"),
                _pct(pl.col("trunc_kept")).alias("trunc_pct_represented"),
                _pct(~pl.col("unknown")).alias("ours_pct_represented"),
                _pct(pl.col("unique_tree")).alias("ours_pct_unique_tree"),
                rt.median().alias("median_rarest_token_occ"),
                (rt / occ).median().alias("median_gain"),
                _pct(rt >= rare).alias(f"pct_rarest_token_ge_{rare}"),
                _pct(rt == occ).alias("pct_no_gain"),
            )
            .sort(["k", "_min_occ"]).drop("_min_occ"))


# ---------------------------------------------------------------------------------------------
# 3. skew of the token frequency distribution, under four vocabularies
# ---------------------------------------------------------------------------------------------
VOCABULARIES = {
    "raw_codes":  "every code is its own token (no compression; independent of k)",
    "trunc_unk":  "top-k codes kept, all other codes mapped to one UNK token",
    "trunc_drop": "top-k codes kept, events of all other codes removed from the data",
    "ours":       "our k tokens; unused tokens count 0; codes without any token map to UNK",
}


def _vocab_counts(per_code, tok, K):
    """Occurrences of every vocabulary unit at budget K, for the four vocabularies.
    Ours: a token occurs once per occurrence of every code whose flat token set contains it
    (relation markers are structure, not vocabulary, and are not counted)."""
    d = per_code.filter(pl.col("k") == K)
    occ = d["occurrences"].to_numpy().astype(float)
    kept = d["trunc_kept"].to_numpy().astype(bool)
    unknown = d["unknown"].to_numpy().astype(bool)

    used = tok.filter(pl.col("k") == K)["token_occ"].to_numpy().astype(float)
    ours = np.concatenate([used, np.zeros(K - len(used))])            # unused tokens: 0
    if occ[unknown].sum() > 0:
        ours = np.append(ours, occ[unknown].sum())                    # UNK for codes without token
    unk = occ[~kept].sum()
    return {
        "raw_codes": occ,
        "trunc_unk": np.append(occ[kept], unk) if unk > 0 else occ[kept],
        "trunc_drop": occ[kept],
        "ours": ours,
    }


def _gini(x):
    """Gini coefficient of the counts (0 = every unit equally frequent, -> 1 = one unit has all)."""
    x = np.sort(np.asarray(x, dtype=float))
    n, s = len(x), x.sum()
    if n == 0 or s == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(2 * (i * x).sum() / (n * s) - (n + 1) / n)


def _dist_row(c, rare):
    c = np.asarray(c, dtype=float)
    V, total = len(c), c.sum()
    p = c[c > 0] / total
    H = float(-(p * np.log(p)).sum())                                 # entropy in nats
    pos = np.sort(c[c > 0])[::-1]
    slope = (float(np.polyfit(np.log(np.arange(1, len(pos) + 1)), np.log(pos), 1)[0])
             if len(pos) >= 3 else float("nan"))
    row = {
        "vocab_size": V,                                              # incl. UNK and unused tokens
        "token_occurrences": float(total),
        "max_token_share": float(c.max() / total),                    # e.g. how dominant UNK is
        "median_token_occ": float(np.median(c)),
        "gini": _gini(c),
        "entropy_norm": H / np.log(V) if V > 1 else float("nan"),     # 1 = perfectly even
        "effective_vocab": float(np.exp(H)),                          # perplexity of the unigram
        "zipf_slope": slope,                                          # log-log rank-frequency slope
    }
    for r in rare:
        row[f"pct_tokens_lt_{r}"] = 100 * float((c < r).mean())       # size of the rare region
    return row


def distribution_stats(per_code, rare=(10, 100)):
    """
    One row per (k, vocabulary). Read the skew metrics together with pct_events_kept:
    trunc_drop looks less skewed only because it deletes the tail from the data.
    Zipf slope: least-squares fit over all units with at least one occurrence (unstable
    for small vocabularies; report as secondary).
    """
    tok = token_table(per_code)
    rows = []
    for K in sorted(per_code["k"].unique().to_list()):
        counts = _vocab_counts(per_code, tok, K)
        total_events = counts["raw_codes"].sum()
        for name, c in counts.items():
            r = {"k": K, "vocabulary": name}
            r["pct_events_kept"] = 100 * (c.sum() / total_events if name == "trunc_drop" else 1.0)
            r.update(_dist_row(c, rare))
            rows.append(r)
    return pl.DataFrame(rows)


def rank_frequency(per_code, K):
    """Rank-frequency curves at budget K for the four vocabularies (units with >= 1 occurrence),
    for a log-log plot. Long format: vocabulary, rank, occurrences."""
    tok = token_table(per_code)
    frames = []
    for name, c in _vocab_counts(per_code, tok, K).items():
        pos = np.sort(c[c > 0])[::-1]
        frames.append(pl.DataFrame({"vocabulary": name, "rank": np.arange(1, len(pos) + 1),
                                    "occurrences": pos}))
    return pl.concat(frames)
