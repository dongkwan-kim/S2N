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
from visualize import plot_scatter, plot_box, plot_line

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError:
    pass


FIGURE_PATH = "../_figures"


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


def visualize_s2n_properties(dataset_path, csv_path, dataset_and_model_name_list,
                             run_analysis=False, extension="png"):
    if run_analysis:
        analyze_s2n_properties(
            dataset_path=dataset_path, out_path=csv_path,
            dataset_and_model_name_list=dataset_and_model_name_list,
        )
    df = pd.read_csv(csv_path)

    df_s2n = df[df["Data structure"] == "S2N"]
    plot_box(
        xs=df_s2n["dataset_name"].to_numpy(),
        ys=df_s2n["Node homophily"].to_numpy(),
        xlabel="Dataset",
        ylabel="Node homophily",
        path=FIGURE_PATH,
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
        path=FIGURE_PATH,
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
        path=FIGURE_PATH,
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


def visualize_efficiency(csv_path,  extension="png"):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Performance", "Throughput (Train)"])

    plot_scatter(
        xs=df["# parameters"].to_numpy(),
        xlabel="# Parameters (Log)",
        ys=df["Max Allocated GPU Memory (MB)"].to_numpy(),
        ylabel="Max Allocated VRAM (MB, Log)",
        path=FIGURE_PATH,
        key="efficiency",
        extension=extension,

        hues=df["Data structure"].to_numpy(), hue_name="Data structure",
        styles=df["Model"].to_numpy(), style_name="Model",
        cols=df["Dataset"].to_numpy(), col_name="Dataset",
        # elm_sizes=df["Performance"].to_numpy(), elm_size_name="Performance",
        scales_kws={"yscale": "log", "xscale": "log"},
        xticks=[1e6, 1e7],
        yticks=[1e2, 1e3, 1e4],
        alpha=0.7,
        s=200,
    )

    plot_scatter(
        xs=df["Throughput (Train)"].to_numpy(),
        xlabel="Train Throughput (#/s, Log)",
        ys=df["Throughput (Eval)"].to_numpy(),
        ylabel="Eval Throughput (#/s, Log)",
        path=FIGURE_PATH,
        key="efficiency",
        extension=extension,

        hues=df["Data structure"].to_numpy(), hue_name="Data structure",
        styles=df["Model"].to_numpy(), style_name="Model",
        cols=df["Dataset"].to_numpy(), col_name="Dataset",
        # elm_sizes=df["Performance"].to_numpy(), elm_size_name="Performance",
        scales_kws={"yscale": "log", "xscale": "log"},
        xticks=[1e1, 1e2, 1e3, 1e4, 1e5],
        yticks=[1e1, 1e2, 1e3, 1e4, 1e5],
        alpha=0.8,
        s=200,
    )

    plot_scatter(
        xs=df["Latency (Train)"].to_numpy(),
        xlabel="Train Latency (s/forward)",
        ys=df["Latency (Eval)"].to_numpy(),
        ylabel="Eval Latency (s/forward)",
        path=FIGURE_PATH,
        key="efficiency",
        extension=extension,

        hues=df["Data structure"].to_numpy(), hue_name="Data structure",
        styles=df["Model"].to_numpy(), style_name="Model",
        cols=df["Dataset"].to_numpy(), col_name="Dataset",
        # elm_sizes=df["Performance"].to_numpy(), elm_size_name="Performance",
        # scales_kws={"yscale": "log", "xscale": "log"},
        xticks=[0.0, 0.1, 0.2, 0.3, 0.4],
        yticks=[0.0, 0.05, 0.1, 0.15],
        alpha=0.8,
        s=200,
    )

    """
    plot_scatter(
        xs=df["Throughput (Train)"].to_numpy(),
        xlabel="Throughput (Train)",
        ys=df["Throughput (Eval)"].to_numpy(),
        ylabel="Throughput (Eval)",
        path=FIGURE_PATH,
        key="efficiency_w_perf",
        extension=extension,

        hues=df["Data structure"].to_numpy(), hue_name="Data structure",
        styles=df["Model"].to_numpy(), style_name="Model",
        cols=df["Dataset"].to_numpy(), col_name="Dataset",
        elm_sizes=df["Performance"].to_numpy(), elm_size_name="Performance",
        scales_kws={"yscale": "log", "xscale": "log"},
        xticks=[1e2, 1e3, 1e4, 1e5],
        yticks=[1e2, 1e3, 1e4, 1e5],
        alpha=0.8,
        s=140,
    )

    plot_scatter(
        xs=df["Throughput (Train)"].to_numpy(),
        xlabel="Throughput (Train)",
        ys=df["Performance"].to_numpy(),
        ylabel="Performance",
        path=FIGURE_PATH,
        key="efficiency",
        extension=extension,

        hues=df["Data structure"].to_numpy(), hue_name="Data structure",
        styles=df["Model"].to_numpy(), style_name="Model",
        cols=df["Dataset"].to_numpy(), col_name="Dataset",
        scales_kws={"xscale": "log"},
        xticks=[1e2, 1e3, 1e4, 1e5],
        yticks=[0.5, 0.6, 0.7, 0.8, 0.9],
        alpha=0.8,
        s=140,
    )
    """


def visualize_efficiency_by_num_training(csv_path,  extension="png", dataset=None):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Performance", "Throughput (Train)"])

    if dataset != "all":  # dataset name
        df = df[df.Dataset == dataset]

        kws = dict(
            path=FIGURE_PATH,
            key=f"efficiency_{dataset.replace('-', '_')}",
            extension=extension,
            markers=True, dashes=False,
            aspect=1.0,
            markersize=13, alpha=0.8,
            hues=df["Data structure"].to_numpy(), hue_name="Data structure",
            styles=df["Data structure"].to_numpy(), style_name="Data structure",
        )
        plot_line(
            xs=df["r_train"].to_numpy(),
            xlabel="Training set ratio",
            ys=df["Performance"].to_numpy(),
            ylabel="Performance",

            yticks=[0.4, 0.5, 0.6],
            legend=False,
            **kws,
        )
        plot_line(
            xs=df["r_train"].to_numpy(),
            xlabel="Training set ratio",
            ys=df["Max Allocated GPU Memory (MB)"].to_numpy(),
            ylabel="Max Allocated VRAM (MB, Log)",

            scales_kws={"yscale": "log"},
            yticks=[1e1, 1e2, 1e3, 1e4],
            # legend=False,  NOTE: # is necessary
            **kws,
        )
        plot_line(
            xs=df["r_train"].to_numpy(),
            xlabel="Training set ratio",
            ys=df["Throughput (Train)"].to_numpy(),
            ylabel="Train Throughput (#/s, Log)",

            scales_kws={"yscale": "log"},
            yticks=[1e3, 1e4, 1e5],
            legend=False,
            **kws,
        )
        plot_line(
            xs=df["r_train"].to_numpy(),
            xlabel="Training set ratio",
            ys=df["Throughput (Eval)"].to_numpy(),
            ylabel="Eval Throughput (#/s, Log)",

            scales_kws={"yscale": "log"},
            yticks=[1e3, 1e4, 1e5],
            legend=False,
            **kws,
        )
        plot_line(
            xs=df["r_train"].to_numpy(),
            xlabel="Training set ratio",
            ys=df["Latency (Train)"].to_numpy(),
            ylabel="Train Latency (s/forward)",

            # scales_kws={"yscale": "log"},
            yticks=[0.0, 0.1, 0.2],
            legend=False,
            **kws,
        )

    else:
        kws = dict(
            path=FIGURE_PATH,
            key="efficiency",
            extension=extension,
            markers=True, dashes=False,
            aspect=1.2,
            markersize=13, alpha=0.8,
            hues=df["Data structure"].to_numpy(), hue_name="Data structure",
            styles=df["Data structure"].to_numpy(), style_name="Data structure",
            cols=df["Dataset"].to_numpy(), col_name="Dataset",
        )
        plot_line(
            xs=df["r_train"].to_numpy(),
            xlabel="Training set ratio",
            ys=df["Performance"].to_numpy(),
            ylabel="Performance",

            yticks=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            **kws,
        )
        plot_line(
            xs=df["r_train"].to_numpy(),
            xlabel="Training set ratio",
            ys=df["Max Allocated GPU Memory (MB)"].to_numpy(),
            ylabel="Max Allocated VRAM (MB, Log)",

            scales_kws={"yscale": "log"},
            yticks=[1e2, 1e3, 1e4],
            **kws,
        )
        plot_line(
            xs=df["r_train"].to_numpy(),
            xlabel="Training set ratio",
            ys=df["Throughput (Train)"].to_numpy(),
            ylabel="Train Throughput (#/s, Log)",

            scales_kws={"yscale": "log"},
            yticks=[1e2, 1e3, 1e4, 1e5],
            **kws,
        )
        plot_line(
            xs=df["r_train"].to_numpy(),
            xlabel="Training set ratio",
            ys=df["Latency (Train)"].to_numpy(),
            ylabel="Train Latency (s/forward)",

            # yticks=[1e2, 1e3, 1e4, 1e5],
            **kws,
        )


if __name__ == '__main__':

    try:
        sns.set(style="whitegrid")
        sns.set_context("talk")
    except NameError:
        pass

    # analyze_s2n_properties, visualize_s2n_properties,
    # visualize_efficiency, visualize_efficiency_by_num_training
    METHOD = "visualize_efficiency_by_num_training"

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
            dataset_path=PATH,
            out_path="./_data_analysis.csv",
            dataset_and_model_name_list=DM_NAME_LIST,
        )
    elif METHOD == "visualize_s2n_properties":
        visualize_s2n_properties(
            dataset_path=PATH,
            csv_path="./_data_analysis_w_original.csv",
            dataset_and_model_name_list=DM_NAME_LIST,
            extension="pdf",
            run_analysis=False,  # NOTE: True to run analyze_s2n_properties
        )
    elif METHOD == "visualize_efficiency":
        visualize_efficiency(
            csv_path="./_sub2node Table (new) - tab_efficiency.csv",
            extension="pdf",
        )
    elif METHOD == "visualize_efficiency_by_num_training":
        visualize_efficiency_by_num_training(
            csv_path="./_sub2node Table (new) - tab_efficiency_by_num_training.csv",
            extension="pdf",
            dataset="HPO-Metab",
        )
