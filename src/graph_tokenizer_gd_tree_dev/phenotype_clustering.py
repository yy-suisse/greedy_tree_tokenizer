"""
Phenotype-discovery clustering for the IA case study (6.4).

Lives in a real module, not defined inline in the notebook, because the multiprocessing pool
below must use the `forkserver` start method rather than the Jupyter kernel's default `fork`.
Forking directly from a live ipykernel process is a known deadlock trap: ipykernel itself runs
several background threads (heartbeat, iopub, shell, control), and forking a multi-threaded
process can hand child processes an inherited-but-dead lock, which then hangs at 0% CPU
forever. `forkserver` sidesteps this by forking from a clean, single-threaded server process
started once up front -- but it requires worker functions to be importable by module path, not
defined ad hoc in a notebook cell, hence this file.
"""

import polars as pl
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score


def cluster_and_characterize(df_feat, T, id_to_label, k_range=range(3, 9), top_n=10, seed=0):
    tok_cols = [c for c in df_feat.columns if c.startswith("tok_")]
    X = df_feat.select(tok_cols).to_numpy()

    n_components = min(50, X.shape[0] - 1, X.shape[1])
    X_pca = PCA(n_components=n_components, random_state=seed).fit_transform(X)

    best = None  # (n_clusters, silhouette, labels)
    for n_clusters in k_range:
        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        if best is None or score > best[1]:
            best = (n_clusters, score, labels)
    n_clusters, silhouette, labels = best

    is_case = df_feat["is_case"].to_numpy()
    gender_female = df_feat["gender_female"].to_numpy()
    pop_freq = X.mean(axis=0)

    rows = []
    for cl in range(n_clusters):
        mask = labels == cl
        cluster_freq = X[mask].mean(axis=0)
        enrichment = (cluster_freq + 1e-3) / (pop_freq + 1e-3)
        top_idx = np.argsort(-enrichment)[:top_n]
        top_tokens = [id_to_label.get(T[i], T[i]) for i in top_idx]
        rows.append({
            "cluster": cl,
            "size": int(mask.sum()),
            "case_rate": float(is_case[mask].mean()),
            "gender_female_rate": float(gender_female[mask].mean()),
            "top_enriched_tokens": top_tokens,
        })

    # (unique_patient_id, cluster) so callers can align cluster assignments across different
    # (method, k) combos by patient id -- row order isn't guaranteed to match across combos.
    df_assignment = pl.DataFrame({
        "unique_patient_id": df_feat["unique_patient_id"],
        "cluster": labels,
    })

    return n_clusters, silhouette, pl.DataFrame(rows), df_assignment


_CLUSTER_WORKER_STATE = {}


def init_cluster_worker(id_to_label, method_specs, assign_ref_dict):
    _CLUSTER_WORKER_STATE.update(id_to_label=id_to_label, method_specs=method_specs, assign_ref_dict=assign_ref_dict)


def cluster_worker(task, n_seeds=5):
    """
    Runs cluster_and_characterize n_seeds times (seed=0..n_seeds-1) and reports mean/std of
    silhouette + ARI/NMI-vs-reference across seeds, instead of a single run's point estimate.
    The reference assignment (`assign_ref_dict`) is fixed at the reference's own seed=0
    clustering -- only this combo's side is re-seeded -- so all 5 numbers measure this
    method's stability against one fixed comparison point, not joint reference+combo noise.
    The saved characterization table (top enriched tokens per cluster) is from seed=0 only;
    averaging that across seeds isn't meaningful since cluster identity/order isn't stable
    across seeds the way scalar metrics are.
    """
    import src.graph_tokenizer_gd_tree_dev.config as config  # forkserver re-imports fresh, keep local

    method_name, k = task
    st = _CLUSTER_WORKER_STATE

    df_feat = pl.read_parquet(f"{config.IAFeatures().path}{method_name}_k{k}.parquet")
    T = pl.read_parquet(st["method_specs"][method_name]).head(k)["token"].to_list()
    ref = st["assign_ref_dict"]

    n_clusters_list, silhouettes, aris, nmis = [], [], [], []
    df_clusters_seed0 = None
    for seed in range(n_seeds):
        n_clusters, silhouette, df_clusters, df_assign = cluster_and_characterize(df_feat, T, st["id_to_label"], seed=seed)
        if seed == 0:
            df_clusters_seed0 = df_clusters

        pids = df_assign["unique_patient_id"].to_list()
        combo_labels = df_assign["cluster"].to_list()
        paired = [(combo_labels[i], ref[pid]) for i, pid in enumerate(pids) if pid in ref]
        combo_l, ref_l = zip(*paired)

        n_clusters_list.append(n_clusters)
        silhouettes.append(silhouette)
        aris.append(adjusted_rand_score(combo_l, ref_l))
        nmis.append(normalized_mutual_info_score(combo_l, ref_l))

    df_clusters_seed0.write_parquet(f"{config.IAFeatures().path}phenotype_clusters/{method_name}_k{k}.parquet")

    return {
        "method": method_name,
        "k": k,
        "n_clusters_mode": max(set(n_clusters_list), key=n_clusters_list.count),
        "silhouette_mean": float(np.mean(silhouettes)),
        "silhouette_std": float(np.std(silhouettes)),
        "ari_vs_no_tokenizer_mean": float(np.mean(aris)),
        "ari_vs_no_tokenizer_std": float(np.std(aris)),
        "nmi_vs_no_tokenizer_mean": float(np.mean(nmis)),
        "nmi_vs_no_tokenizer_std": float(np.std(nmis)),
    }
