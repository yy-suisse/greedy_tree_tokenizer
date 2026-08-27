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


class ProcessedGraph:
    path: str = "D:/greedy_graph_data/processed_graph/"
    combined_subgraphs: str = f"{path}combined_subgraphs.gpickle"
    id_to_label: str = f"{path}id_to_label.pkl"

    all_rel_distance_subgraph: str = f"{path}all_rel_distance_subgraph.parquet"
    is_a_distance_subgraph: str = f"{path}is_a_distance_subgraph.parquet"


class CandidateLists:
    path: str = "D:/greedy_graph_data/"

    path_greedy_tree: str = f"{path}greedy_tree_candidates/"

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
    path: str = "D:/greedy_graph_data/metrics_results/"

    perf_baseline_path: str = f"{path}baseline_performance.parquet"

    perf_greedy_tree_path: str = f"{path}greedy_tree_performance.parquet"
    perf_greedy_tree_path_append: str = f"{path}greedy_tree_performance_append.parquet"
    perf_greedy_tree_path_all: str = f"{path}greedy_tree_performance_all.parquet"


    perf_k_rdn_path: str = f"{path}k_rdn_performance.parquet"



class TimelineData:
    path: str = "/mnt/z/PREM_STAGE/hero_timeline_2026_06_25/"
    patient_list_all: str = "/mnt/z/PREM_STAGE/Liste patients totale.xlsx"
    raw_timeline: str = f"{path}timeline.parquet"
    hashed_patient_ids = f"{path}hashed_patient_ids.csv"

    missing_patient_ids = f"{path}missing_real_patient_ids.csv"
    patient_ids_and_index_lookup = f"{path}patient_ids_and_index_lookup.parquet"
    patient_gender: str = f"{path}patient_unique_id_gender.parquet"

    extracted_traj_path: str = f"{path}/data_after_preprocessing/"
    all_patient_traj: str = f"{extracted_traj_path}/all_patient_traj.parquet"
    used_cpt: str = f"{path}cpt_used.json"


class TimelineVocab:
    path: str = "/mnt/z/graph_tokenizer_greedy_tree/IA_patient_analysis/vocabs/"
    snomed_vocabs_original: str = f"{path}snomed_vocabs.parquet"

    gender_vocab: list = ["248153007", "248152002"]
    gender_label: list = ["Female (finding)", "Male (finding)"]

    ANEURYSM_CODES = [
        "I67.10",
        "I67.1",
        "Q28.20",
        "Q28.30",
        "128608001",
        "128609009",
        "783420001",
        "737159004",
        "783716004",
    ]


class IAProcessedGraph:
    path: str = "/mnt/z/graph_tokenizer_greedy_tree/IA_patient_analysis/processed_graph/"
    combined_subgraphs: str = f"{path}combined_subgraphs.gpickle"

    all_rel_distance_subgraph: str = f"{path}all_rel_distance_subgraph.parquet"
    is_a_distance_subgraph: str = f"{path}is_a_distance_subgraph.parquet"


class IACandidateLists:
    path: str = "/mnt/z/graph_tokenizer_greedy_tree/IA_patient_analysis/"

    path_greedy_tree: str = f"{path}greedy_tree_candidates/"

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


class IAResults:
    path: str = "/mnt/z/graph_tokenizer_greedy_tree/IA_patient_analysis/results/"

    perf_baseline_path: str = f"{path}baseline_performance.parquet"
    perf_greedy_tree_path: str = f"{path}greedy_tree_performance.parquet"


class IACohort:
    patient_list_path: str = "/mnt/z/PREM_STAGE/Liste patients totale.xlsx"  # PatID, Profil ("Patient"=case, "Control"=control)


class IAFeatures:
    path: str = "/mnt/z/graph_tokenizer_greedy_tree/IA_patient_analysis/features/"
