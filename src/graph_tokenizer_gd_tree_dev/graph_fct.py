import networkx as nx
import polars as pl
import pickle

import src.graph_tokenizer_gd_tree_dev.config as config
import src.graph_tokenizer_gd_tree_dev.graph_fct as graph_fct


def build_relations_graph(df_relations, col_src="src.id", col_dst="dst.id", col_relation="relation") -> nx.MultiDiGraph:
    """
    Build a directed graph with all relations as edges: src -> dst, labeled with relation type.
    Uses a MultiDiGraph since a (src, dst) pair can carry more than one relation type.
    """
    G = nx.MultiDiGraph()
    for src, dst, relation in df_relations.select(col_src, col_dst, col_relation).iter_rows():
        G.add_edge(src, dst, relation=relation)
    return G


def get_combined_subgraphs_from_nodes(G: nx.MultiDiGraph, nodes, D) -> dict:
    """
    For each node, return the subgraph reachable by following outgoing edges up to
    max_distance hops. Nodes absent from G map to None.
    """
    subgraphs = {node: nx.ego_graph(G, node, radius=D) if node in G else None for node in nodes}
    return nx.compose_all([sg for sg in subgraphs.values() if sg is not None])

def build_combined_subgraphs_and_id2label(D:int):
    df_relations = pl.read_parquet(f"{config.BasicConfig().relation_path}")
    df_mapped = pl.read_parquet(f"{config.BasicConfig().mapped_path}")
    mapped_ids = df_mapped["id"].to_list()

    # remove root concept
    df_relations = df_relations.filter(~pl.col("dst.id").is_in(config.TokenizerParam().exclude_cpt))

    # build graph and combine
    whole_graph = graph_fct.build_relations_graph(df_relations, col_src="src.id", col_dst="dst.id", col_relation="relation")
    combined_subgraphs = graph_fct.get_combined_subgraphs_from_nodes(whole_graph, mapped_ids, D)
    df_cpt = pl.read_parquet(config.BasicConfig().concept_path).select("id", "label")
    id_to_label = dict(zip(df_cpt["id"], df_cpt["label"]))
    
    with open(config.ProcessedGraph().id_to_label, "wb") as f:
        pickle.dump(id_to_label, f)

    with open(config.ProcessedGraph().combined_subgraphs, "wb") as f:
        pickle.dump(combined_subgraphs, f)

def get_combined_combined_subgraphs_and_id2label():
    with open(config.ProcessedGraph().id_to_label, "rb") as f:
        id_to_label = pickle.load(f)

    with open(config.ProcessedGraph().combined_subgraphs, "rb") as f:
        combined_subgraphs = pickle.load(f)

    return id_to_label, combined_subgraphs