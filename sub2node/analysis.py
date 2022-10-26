import csv
from itertools import product
from pprint import pprint
from typing import List, Tuple, Any, Dict

import pandas as pd
from omegaconf import OmegaConf
from termcolor import cprint
from torch_geometric.data import Data
from torch_geometric.utils import homophily, remove_self_loops

from data import SubgraphDataModule
from utils import multi_label_homophily, try_get_from_dict
from visualize import plot_scatter, plot_box

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError:
    pass


def to_dataset_repr(dataset_name, repr_format):
    if repr_format == "filename":
        return {
            "PPIBP": "ppi_bp",
            "HPONeuro": "hpo_neuro",
            "HPOMetab": "hpo_metab",
            "EMUser": "em_user"
        }[dataset_name]
    elif repr_format == "paper":
        return {
            "PPIBP": "PPI-BP",
            "HPONeuro": "HPO-Neuro",
            "HPOMetab": "HPO-Metab",
            "EMUser": "EM-User"
        }[dataset_name]


def _analyze_node_properties(data: Data):
    N = data.num_nodes

    properties = dict()

    if data.y.squeeze().dim() == 1:
        properties["Node homophily"] = homophily(data.edge_index, data.y, method="node")
        properties["Edge homophily"] = homophily(data.edge_index, data.y, method="edge")
        properties["# classes"] = data.y.max().item() + 1
        properties["Single- or multi-labels"] = "Single"
    else:
        properties["Node homophily"] = multi_label_homophily(data.edge_index, data.y, method="node")
        properties["Edge homophily"] = multi_label_homophily(data.edge_index, data.y, method="edge")
        properties["# classes"] = data.y.size(1)
        properties["Single- or multi-labels"] = "Multi"

    edge_index, edge_attr = remove_self_loops(data.edge_index, getattr(data, "edge_attr", None))
    properties["Density"] = edge_index.size(1) / (N * (N - 1))
    properties["# nodes"] = N
    properties["# edges"] = data.edge_index.size(1)

    return properties


def analyze_s2n_properties(dataset_path, dataset_and_model_name_list: List[Tuple[str, str]],
                           out_path=None):
    list_of_pps = []
    for (dataset_name, model_name) in dataset_and_model_name_list:
        sdm = SubgraphDataModule(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            embedding_type="gin",
            use_s2n=True,
            edge_thres=0.0,
            use_consistent_processing=True,
            edge_normalize="standardize_then_trunc_thres_max_linear",
            # edge_normalize_arg_1=0.75,
            # edge_normalize_arg_2=1.0,
            # s2n_target_matrix="adjacent_no_self_loops",
            s2n_is_weighted=True,
            subgraph_batching=None,
            batch_size=None,
            eval_batch_size=None,
            use_sparse_tensor=USE_SPARSE_TENSOR,
            pre_add_self_loops=False,
            replace_x_with_wl4pattern=False,
            wl4pattern_args=None,
            custom_splits=None,
            **load_s2n_datamodule_kwargs(dataset_name, model_name),
        )
        pps_dict = _analyze_node_properties(sdm.test_data)
        list_of_pps.append({"dataset_name": to_dataset_repr(dataset_name, "paper"),
                            "model_name": model_name, **pps_dict})
        pprint(list_of_pps)

    if out_path is not None:
        with open(out_path, "w") as f:
            cprint(f"Save properties at {out_path}", "blue")
            writer = csv.DictWriter(f, fieldnames=[k for k in list_of_pps[0].keys()])
            writer.writeheader()
            for pps in list_of_pps:
                writer.writerow(pps)

    return list_of_pps


def load_s2n_datamodule_kwargs(dataset_name, model_name) -> Dict[str, Any]:
    assert model_name in ["fa", "gat", "gcn", "gcn2", "gin", "linkx", "sage"]
    dataset_name = {
        "PPIBP": "ppi_bp",
        "HPONeuro": "hpo_neuro",
        "HPOMetab": "hpo_metab",
        "EMUser": "em_user"
    }[dataset_name]
    yaml_name = f"../configs/datamodule/s2n/{dataset_name}/for-{model_name}.yaml"
    cfg = OmegaConf.load(yaml_name)
    cprint(f"Load: {yaml_name}", "green")
    kwargs = try_get_from_dict(cfg, ["edge_normalize_arg_1", "edge_normalize_arg_2", "s2n_target_matrix"],
                               as_dict=True)
    return kwargs


def visualize_s2n_properties(dataset_path, out_path, dataset_and_model_name_list,
                             run_analysis=False, extension="png"):
    if run_analysis:
        analyze_s2n_properties(
            dataset_path=dataset_path, out_path=out_path,
            dataset_and_model_name_list=dataset_and_model_name_list,
        )
    df = pd.read_csv(out_path)

    df_s2n = df[df["Data structure"] == "S2N"]
    plot_box(
        xs=df_s2n["dataset_name"].to_numpy(),
        ys=df_s2n["Node homophily"].to_numpy(),
        xlabel="Dataset",
        ylabel="Node homophily",
        path="../_figures",
        key="s2n_properties",
        extension=extension,

        yticks=[0.0, 0.1, 0.2, 0.3, 0.4],
        orient="v", width=0.5,
        aspect=1.45,
    )

    plot_scatter(
        xs=df["# nodes"].to_numpy(),
        ys=df["# edges"].to_numpy(),
        xlabel="# Nodes (Log)",
        ylabel="# Edges (Log)",
        path="../_figures",
        key="s2n_properties",
        extension=extension,

        hues=df["Data structure"].to_numpy(), hue_name="Data structure",
        styles=df["dataset_name"].to_numpy(), style_name="Dataset",
        scales_kws={"yscale": "log", "xscale": "log"},
        xticks=[1e2, 1e3, 1e4, 1e5],
        yticks=[1e2, 1e3, 1e4, 1e5, 1e6, 1e7],
        alpha=0.8,
        s=200,
    )

    """
    # scatter plot of Node homophily and Density
    df = df[df["Data structure"] == "S2N"]
    plot_scatter(
        xs=df["Node homophily"].to_numpy(),
        ys=df["Density"].to_numpy(),
        xlabel="Node homophily",
        ylabel="Density",
        path="../_figures",
        key="s2n_properties",
        extension=extension,

        hues=df["dataset_name"].to_numpy(), hue_name="Dataset",
        styles=df["dataset_name"].to_numpy(), style_name="Dataset",
        scales_kws={"yscale": "log"},
        yticks=[0.001, 0.01, 0.1],
        alpha=0.8,
        s=200,
    )
    """


if __name__ == '__main__':

    try:
        sns.set(style="whitegrid")
        sns.set_context("talk")
    except NameError:
        pass

    # analyze_s2n_properties, visualize_s2n_properties
    METHOD = "visualize_s2n_properties"

    TARGETS = "REAL_WORLD"  # SYNTHETIC, REAL_WORLD, ALL
    if TARGETS == "REAL_WORLD":
        DATASET_NAME_LIST = ["PPIBP", "HPONeuro", "HPOMetab", "EMUser"]
    else:
        raise ValueError(f"Wrong targets: {TARGETS}")

    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    E_TYPE = "gin"  # gin, graphsaint_gcn
    USE_SPARSE_TENSOR = False

    DM_NAME_LIST = list(product(DATASET_NAME_LIST,
                                ["fa", "gat", "gcn", "gcn2", "gin", "linkx", "sage"]))
    if METHOD == "analyze_s2n_properties":
        analyze_s2n_properties(
            dataset_path=PATH, out_path="./_data_analysis.csv",
            dataset_and_model_name_list=DM_NAME_LIST,
        )
    elif METHOD == "visualize_s2n_properties":
        visualize_s2n_properties(
            dataset_path=PATH, out_path="./_data_analysis_w_original.csv",
            dataset_and_model_name_list=DM_NAME_LIST,
            extension="pdf",
            run_analysis=False,  # NOTE: True to run analyze_s2n_properties
        )
