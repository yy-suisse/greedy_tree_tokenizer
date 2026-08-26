import polars as pl 
import os 

def to_ranked_df(id_to_label, scores: dict, score_col: str) -> pl.DataFrame:
    """dict{node: score} -> the same token/score/label/index schema notebook 3's baselines use,
    sorted so .head(k) at eval time gives the top-k under this ranking."""
    return (
        pl.DataFrame({"token": list(scores.keys()), score_col: list(scores.values())})
        .with_columns(pl.col("token").replace_strict(id_to_label, default=None).alias("label"))
        .sort(score_col, descending=True)
        .with_row_index()
    )

def to_type_dict(file_list):
    return {os.path.splitext(os.path.basename(f))[0]: f for f in file_list}