"""Unblind and compare the experts' answers.

Every expert rates the same concepts in the same form. Their CSV files only know the
positions A / B. The ground-truth file says, per concept, which position shows our
tokenizer and which shows the baseline (plus the category and the official status).

Usage
-----
    python unblind.py GROUND_TRUTH.csv ANSWERS_FOLDER [OUT_PREFIX]

    GROUND_TRUTH.csv  columns: study_id, item, concept_id, concept_name, category, status, A, B
    ANSWERS_FOLDER    folder holding the experts' answers_*.csv files (any number of experts)
    OUT_PREFIX        default: results  ->  results_long.csv, results_wide.csv

Outputs
-------
    results_long.csv  one row per (expert, concept): preferred = ours / baseline / no_difference,
                      score (+1 ours better, 0 no difference, -1 baseline better), problems per method
    results_wide.csv  one row per concept: one score column per expert, mean score, agreement

If an expert sent several downloads, the most recent file wins (file names end with a local
timestamp). Standard library only, no install needed.
"""
import csv
import glob
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations
from statistics import mean

ISSUES = ["incorrect_or_ambiguous", "information_absent", "precision_lost", "redundant"]
SCORE = {"ours": 1, "no_difference": 0, "baseline": -1}


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def unblind_row(r, k, source):
    if {k["A"], k["B"]} != {"ours", "baseline"}:
        raise SystemExit(f"Ground truth for concept {k['concept_id']} must contain 'ours' and 'baseline'.")
    ours = "A" if k["A"] == "ours" else "B"
    base = "B" if ours == "A" else "A"
    preferred = {ours: "ours", base: "baseline", "equal": "no_difference"}.get(r["choice"], "unanswered")
    out = {
        "rater_id": r["rater_id"].strip().upper(),
        "item": k.get("item", ""),
        "concept_id": r["concept_id"],
        "concept_name": r["concept_name"],
        "category": k.get("category", ""),
        "status": k.get("status", ""),
        "ours_shown_as": ours,
        "preferred": preferred,
        "score": SCORE.get(preferred, ""),
    }
    for method, side in (("ours", ours), ("baseline", base)):
        for issue in ISSUES:
            out[f"{method}_{issue}"] = r[f"{side}_{issue}"]
    out.update(definition_insufficient=r["definition_insufficient"], comment=r["comment"],
               seconds_spent=r["seconds_spent"], view_at_answer=r.get("view_at_answer", ""),
               position_for_rater=r.get("position", ""), source_file=source)
    return out


def krippendorff_alpha_ordinal(units):
    """units: list of lists of ordinal values (one list per concept, one value per expert).
    Krippendorff's alpha with the ordinal difference function; units with < 2 values are ignored."""
    units = [u for u in units if len(u) >= 2]
    if not units:
        return None
    values = sorted({v for u in units for v in u})
    idx = {v: i for i, v in enumerate(values)}
    m = len(values)
    o = [[0.0] * m for _ in range(m)]                      # coincidence matrix
    for u in units:
        mu = len(u)
        for a, b in ((a, b) for i, a in enumerate(u) for j, b in enumerate(u) if i != j):
            o[idx[a]][idx[b]] += 1.0 / (mu - 1)
    n_c = [sum(row) for row in o]
    n = sum(n_c)
    if n <= 1:
        return None

    def delta2(c, k):                                       # ordinal metric
        lo, hi = min(c, k), max(c, k)
        return (sum(n_c[g] for g in range(lo, hi + 1)) - (n_c[lo] + n_c[hi]) / 2) ** 2

    d_o = sum(o[c][k] * delta2(c, k) for c in range(m) for k in range(m))
    d_e = sum(n_c[c] * n_c[k] * delta2(c, k) for c in range(m) for k in range(m))
    return None if d_e == 0 else 1 - (n - 1) * d_o / d_e


def main(truth_path, answers_dir, out_prefix="results"):
    truth = {(t["study_id"], t["concept_id"]): t for t in read_csv(truth_path)}
    files = sorted(glob.glob(os.path.join(answers_dir, "answers_*.csv")))
    if not files:
        raise SystemExit(f"No answers_*.csv files in {answers_dir}")

    latest = {}  # (rater, concept) -> row; later files overwrite earlier ones
    for path in files:
        for r in read_csv(path):
            k = truth.get((r["study_id"], r["concept_id"]))
            if k is None:
                raise SystemExit(f"{os.path.basename(path)}: concept {r['concept_id']} is not in the ground truth")
            row = unblind_row(r, k, os.path.basename(path))
            latest[(row["rater_id"], row["concept_id"])] = row

    rows = sorted(latest.values(), key=lambda x: (x["rater_id"], int(x["item"] or 0)))
    raters = sorted({r["rater_id"] for r in rows})
    with open(f"{out_prefix}_long.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # wide table: one row per concept, one score column per expert
    by_concept = defaultdict(dict)
    for r in rows:
        if r["score"] != "":
            by_concept[r["concept_id"]][r["rater_id"]] = r["score"]
    wide = []
    for t in sorted(truth.values(), key=lambda t: int(t.get("item") or 0)):
        sc = by_concept.get(t["concept_id"], {})
        vals = list(sc.values())
        wide.append({"item": t.get("item", ""), "concept_id": t["concept_id"], "concept_name": t["concept_name"],
                     "category": t.get("category", ""), "status": t.get("status", ""),
                     **{f"score_{r}": sc.get(r, "") for r in raters},
                     "n_experts": len(vals), "mean_score": round(mean(vals), 3) if vals else "",
                     "all_agree": (len(set(vals)) == 1) if len(vals) >= 2 else ""})
    with open(f"{out_prefix}_wide.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(wide[0].keys()))
        w.writeheader()
        w.writerows(wide)

    # ---------------- summary ----------------
    print(f"Experts: {', '.join(raters)} | wrote {out_prefix}_long.csv ({len(rows)} rows) and {out_prefix}_wide.csv ({len(wide)} concepts)\n")
    print("Per expert:")
    for rater in raters:
        c = Counter(r["preferred"] for r in rows if r["rater_id"] == rater)
        print(f"  {rater:<8} ours {c['ours']:>3} | baseline {c['baseline']:>3} | no difference {c['no_difference']:>3} | unanswered {c['unanswered']:>3}")

    print("\nPer stratum (all experts):")
    for key in sorted({(r["category"], r["status"]) for r in rows}):
        c = Counter(r["preferred"] for r in rows if (r["category"], r["status"]) == key)
        print(f"  {key[0]:<15} {key[1]:<10} ours {c['ours']:>3} | baseline {c['baseline']:>3} | no difference {c['no_difference']:>3}")

    if len(raters) >= 2:
        units = [list(sc.values()) for sc in by_concept.values()]
        alpha = krippendorff_alpha_ordinal(units)
        print(f"\nAgreement between experts: Krippendorff's alpha (ordinal) = {alpha:.3f}" if alpha is not None else "")
        for r1, r2 in combinations(raters, 2):
            pairs = [(sc[r1], sc[r2]) for sc in by_concept.values() if r1 in sc and r2 in sc]
            if pairs:
                print(f"  {r1} vs {r2}: same answer on {sum(a == b for a, b in pairs)}/{len(pairs)} concepts")
    print("\nWeighting by stratum size (strata_*.csv) gives the population-level result; see the notebook.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:4])
