from dataclasses import dataclass

import numpy as np


@dataclass
class BasicConfig:
    graph_path: str = "D:/HUG_graph_data/2026-07-03/"

    relation_path: str = f"{graph_path}connectivity.parquet"
    concept_path: str = f"{graph_path}concept_snomed_hug.parquet"
    mapped_path: str = f"{graph_path}mapped_concepts.parquet"
    official_release_path: str = f"{graph_path}released_version.parquet"


class TokenizerParam:
    # token-search horizon D (IS_A generalization hops): covers 98.4% of concept depth in
    # combined_subgraphs, capped below dag_longest_path_length (30, a handful of outlier
    # chains) since depth 10+ hits are already generic SNOMED "grouper" categories
    # (e.g. "Wound", "Inspection (procedure)") rather than clinically specific concepts
    max_dist_candidate: int = 9
    Ks: np.ndarray = np.arange(500, 20000, 500)
    rnd_iters: np.ndarray = np.arange(0, 20, 1)
    exclude_cpt: list = ["138875005"]

    # distance decay grid for the status-dependent variant (same grid as the hop-based results)
    lambdas: list = sorted({float(x) for x in np.round(np.arange(0.0, 1.05, 0.1), 1)}
                           | {float(x) for x in np.round(np.arange(0.91, 1.0, 0.01), 2)})
    lambdas_real_data: list = [float(x) for x in np.round(np.arange(0.6, 1.05, 0.1), 1)]


class ProcessedGraph:
    path: str = "D:/greedy_graph_data/processed_graph_prim_fd/"
    combined_subgraphs: str = f"{path}combined_subgraphs.gpickle"
    id_to_label: str = f"{path}id_to_label.pkl"

    all_rel_distance_subgraph: str = f"{path}all_rel_distance_subgraph.parquet"
    is_a_distance_subgraph: str = f"{path}is_a_distance_subgraph.parquet"
    concept_and_status: str = f"{path}concept_and_status.parquet"

class CandidateLists:
    path: str = "D:/greedy_graph_data/"

    path_greedy_tree: str = f"{path}greedy_tree_candidates_prim_fd/"  # was greedy_tree_candidates/: notebook 2 would have overwritten the hop-based runs

    baseline_path: str = f"{path}baseline_candidates/"
    highest_degree: str = f"{baseline_path}highest_degree.parquet"
    highest_degree_dist_1: str = f"{baseline_path}highest_degree_dist_1.parquet"
    k_random_all_samples: str = f"{baseline_path}k_random_all_samples.parquet"
    most_children: str = f"{baseline_path}most_children.parquet"
    pagerank: str = f"{baseline_path}pagerank.parquet"
    personalized_pagerank: str = f"{baseline_path}personalized_pagerank.parquet"
    closeness_centrality: str = f"{baseline_path}closeness_centrality.parquet"
    eigenvector_centrality: str = f"{baseline_path}eigenvector_centrality.parquet"
    discrete_set_cover: str = f"{baseline_path}discrete_set_cover.parquet"


class Results:
    path: str = "D:/greedy_graph_data/metrics_results_prim_fd/"

    perf_baseline_path: str = f"{path}baseline_performance.parquet"

    perf_greedy_tree_path: str = f"{path}greedy_tree_performance.parquet"
    perf_greedy_tree_path_append: str = f"{path}greedy_tree_performance_append.parquet"
    perf_greedy_tree_path_all: str = f"{path}greedy_tree_performance_all.parquet"
    perf_k_rdn_path: str = f"{path}k_rdn_performance.parquet"

    perf_all: str = f"{path}all_performance.parquet"
    perf_all_csv: str = f"{path}all_performance.csv"
    
    perf_ranked_csv : str = f"{path}ranked_performance.csv"

class HERODataVocabsTest:
    path: str = "D:/greedy_graph_data/test_vocabs_prim_fd/"
    vocab_count_patient: str = f"{path}vocab_preq_patient.parquet"
    vocab_count_volumn: str = f"{path}vocab_volume_2017.parquet"

    subgraph_path: str = f"{path}subgraph_info/"
    greedy_candidate_path: str = f"{path}candidates/"

    perf_greedy_tree_path: str = f"{path}greedy_tree_performance.parquet"
    rank_perf : str = f"{path}rank_perf.csv"
