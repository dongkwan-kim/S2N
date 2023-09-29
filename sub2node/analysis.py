import csv
from itertools import product
from pathlib import Path
from pprint import pprint
from typing import List, Tuple, Any, Dict

import pandas as pd
from omegaconf import OmegaConf
from termcolor import cprint
from torch_geometric.data import Data
from torch_geometric.utils import homophily, remove_self_loops

from data import SubgraphDataModule
from utils import multi_label_homophily, try_get_from_dict
from visualize import plot_scatter, plot_box, plot_line, plot_bar

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError:
    pass

FIGURE_PATH = "../_figures"

# S2N+0, S2N+A, Connected, Separated, Baseline
METHOD_COLORS = ["#1e88e5", "#3949ab", "#fb8c00", "#43a047", "#e53935"]

# GCN, GAT, SAGE, GCNII, SubGNN, GLASS
# MODEL_COLORS = ["#00acc1", "#00897b", "#43a047", "#7cb342", "#e53935", "#8e24aa"]

# GCN, GCNII, SubGNN, GLASS
MODEL_COLORS = ["#00acc1", "#7cb342", "#e53935", "#8e24aa"]


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


def to_s2n_repr(s2n_name):
    return {
        "s2n": "S2N+0",
        "sub_s2n": "S2N+A",
    }[s2n_name]


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


def analyze_s2n_properties(dataset_path, dataset_model_s2n_name_list: List[Tuple[str, str]],
                           out_path=None):
    list_of_pps = []
    for (dataset_name, model_name, s2n_name) in dataset_model_s2n_name_list:
        sdm = SubgraphDataModule(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            embedding_type="glass",
            use_s2n=True,
            s2n_mapping_matrix_type="unnormalized",
            s2n_set_sub_x_weight="original_sqrt_d_node_div_d_sub",
            s2n_use_sub_edge_index=False,
            s2n_add_sub_x_wl=False,
            edge_thres=0.0,
            use_consistent_processing=True,
            post_edge_normalize="standardize_then_trunc_thres_max_linear",
            # post_edge_normalize_arg_1=2.0,
            # post_edge_normalize_arg_2=2.0,
            s2n_target_matrix="adjacent_with_self_loops",
            s2n_is_weighted=True,
            subgraph_batching=None,
            batch_size=None,
            eval_batch_size=None,
            use_sparse_tensor=USE_SPARSE_TENSOR,
            pre_add_self_loops=False,
            replace_x_with_wl4pattern=False,
            wl4pattern_args=None,
            custom_splits=None,
            **load_s2n_datamodule_kwargs(dataset_name, model_name, s2n_name),
        )
        pps_dict = _analyze_node_properties(sdm.test_data)
        list_of_pps.append({"dataset_name": to_dataset_repr(dataset_name, "paper"),
                            "model_name": model_name,
                            **pps_dict,
                            "Data structure": to_s2n_repr(s2n_name)})
        pprint(list_of_pps)

    if out_path is not None:
        with open(out_path, "w") as f:
            cprint(f"Save properties at {out_path}", "blue")
            writer = csv.DictWriter(f, fieldnames=[k for k in list_of_pps[0].keys()])
            writer.writeheader()
            for pps in list_of_pps:
                writer.writerow(pps)

    return list_of_pps


def load_s2n_datamodule_kwargs(dataset_name, model_name, s2n_name) -> Dict[str, Any]:
    assert model_name in ["fa", "gat", "gcn", "gcn2", "gin", "linkx", "sage"]
    dataset_name = {
        "PPIBP": "ppi_bp",
        "HPONeuro": "hpo_neuro",
        "HPOMetab": "hpo_metab",
        "EMUser": "em_user"
    }[dataset_name]
    yaml_name = f"../configs/datamodule/{s2n_name}/{dataset_name}/for-{model_name}.yaml"
    cfg = OmegaConf.load(yaml_name)
    cprint(f"Load: {yaml_name}", "green")
    kwargs = try_get_from_dict(cfg, ["post_edge_normalize_arg_1", "post_edge_normalize_arg_2"],
                               as_dict=True)
    return kwargs


def visualize_s2n_properties(dataset_path, csv_path, dataset_model_s2n_name_list,
                             run_analysis=False, extension="png"):
    if run_analysis:
        analyze_s2n_properties(
            dataset_path=dataset_path, out_path=csv_path,
            dataset_model_s2n_name_list=dataset_model_s2n_name_list,
        )
    df = pd.read_csv(csv_path)

    df_s2n = df[(df["Data structure"] == "S2N+0") | (df["Data structure"] == "S2N+A")]
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

    sns.set_palette(METHOD_COLORS)
    df["Data structure"] = df["Data structure"].replace(["Connected"], "Original")
    plot_scatter(
        xs=df["# nodes"].to_numpy(),
        ys=df["# edges"].to_numpy(),
        xlabel="# Nodes",
        ylabel="# Edges",
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


def visualize_efficiency(csv_path, extension="png", queries: List[str] = None):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Performance", "Throughput (Train)"])

    if queries is not None:
        for query in queries:
            df = df.query(query)

    kws = dict(
        path=FIGURE_PATH,
        key="efficiency",
        extension=extension,
    )
    scatter_kws = dict(alpha=0.65, s=300, **kws)
    bar_kws = dict(aspect=2.0, **kws)

    sns.set_palette(MODEL_COLORS)

    """
    plot_bar(
        xs=df["Data structure"].to_numpy(),
        xlabel=None,
        ys=df["# parameters"].to_numpy(),
        ylabel="# parameters",

        hues=df["Model"].to_numpy(), hue_name="Model",
        cols=df["Dataset"].to_numpy(), col_name="Dataset",
        legend=True,
        **bar_kws,
    )
    plot_bar(
        xs=df["Data structure"].to_numpy(),
        xlabel=None,
        ys=df["Max Allocated GPU Memory (MB)"].to_numpy(),
        ylabel="Max Allocated VRAM (MB)",

        hues=df["Model"].to_numpy(), hue_name="Model",
        cols=df["Dataset"].to_numpy(), col_name="Dataset",
        yticks=[0, 1000, 2000, 3000, 4000, 5000, 6000],
        legend=True,
        **bar_kws,
    )

    plot_bar(
        xs=df["Data structure"].to_numpy(),
        xlabel=None,
        ys=df["Throughput (Train)"].to_numpy(),
        ylabel="Train Throughput (Log)",

        hues=df["Model"].to_numpy(), hue_name="Model",
        cols=df["Dataset"].to_numpy(), col_name="Dataset",
        scales_kws={"yscale": "log"},
        yticks=[1e1, 1e2, 1e3, 1e4, 1e5],
        legend=True,
        **bar_kws,
    )
    plot_bar(
        xs=df["Data structure"].to_numpy(),
        xlabel=None,
        ys=df["Throughput (Eval)"].to_numpy(),
        ylabel="Eval Throughput (Log)",

        hues=df["Model"].to_numpy(), hue_name="Model",
        cols=df["Dataset"].to_numpy(), col_name="Dataset",
        scales_kws={"yscale": "log"},
        legend=True,
        yticks=[1e1, 1e2, 1e3, 1e4, 1e5],
        **bar_kws,
    )

    plot_bar(
        xs=df["Data structure"].to_numpy(),
        xlabel=None,
        ys=df["Latency (Train)"].to_numpy(),
        ylabel="Train Latency",

        hues=df["Model"].to_numpy(), hue_name="Model",
        cols=df["Dataset"].to_numpy(), col_name="Dataset",
        legend=True,
        **bar_kws,
    )
    plot_bar(
        xs=df["Data structure"].to_numpy(),
        xlabel=None,
        ys=df["Latency (Eval)"].to_numpy(),
        ylabel="Eval Latency",

        hues=df["Model"].to_numpy(), hue_name="Model",
        cols=df["Dataset"].to_numpy(), col_name="Dataset",
        legend=True,
        **bar_kws,
    )

    """

    # scatter plots
    sns.set_palette(METHOD_COLORS)
    plot_scatter(
        xs=df["# parameters"].to_numpy(),
        xlabel="# parameters (Log)",
        ys=df["Max Allocated GPU Memory (MB)"].to_numpy(),
        ylabel="Max Allocated VRAM (MB)",

        hues=df["Data structure"].to_numpy(), hue_name="Data structure",
        styles=df["Model"].to_numpy(), style_name="Model",
        cols=df["Dataset"].to_numpy(), col_name="Dataset",
        # elm_sizes=df["Performance"].to_numpy(), elm_size_name="Performance",
        scales_kws={"yscale": "log", "xscale": "log"},
        xticks=[1e6, 1e7],
        yticks=[1e1, 1e2, 1e3, 1e4],
        legend=True,
        **scatter_kws,
    )

    plot_scatter(
        xs=df["Throughput (Train)"].to_numpy(),
        xlabel="Train Throughput (#/s)",
        ys=df["Throughput (Eval)"].to_numpy(),
        ylabel="Eval Throughput (#/s)",

        hues=df["Data structure"].to_numpy(), hue_name="Data structure",
        styles=df["Model"].to_numpy(), style_name="Model",
        cols=df["Dataset"].to_numpy(), col_name="Dataset",
        # elm_sizes=df["Performance"].to_numpy(), elm_size_name="Performance",
        scales_kws={"yscale": "log", "xscale": "log"},
        xticks=[1e1, 1e2, 1e3, 1e4, 1e5],
        yticks=[1e1, 1e2, 1e3, 1e4, 1e5],
        legend=True,
        **scatter_kws,
    )

    plot_scatter(
        xs=df["Latency (Train)"].to_numpy(),
        xlabel="Train Latency (s/forward)",
        ys=df["Latency (Eval)"].to_numpy(),
        ylabel="Eval Latency (s/forward)",

        hues=df["Data structure"].to_numpy(), hue_name="Data structure",
        styles=df["Model"].to_numpy(), style_name="Model",
        cols=df["Dataset"].to_numpy(), col_name="Dataset",
        # elm_sizes=df["Performance"].to_numpy(), elm_size_name="Performance",
        # scales_kws={"yscale": "log", "xscale": "log"},
        xticks=[0.0, 0.1, 0.2, 0.3, 0.4],
        yticks=[0.0, 0.05, 0.1, 0.15],
        legend=True,
        **scatter_kws,
    )


def visualize_performance_by_coarsening_ratio(path_to_csvs, extension="png", dataset=None):
    sns.set_palette(METHOD_COLORS)

    if dataset is None:
        files = Path(path_to_csvs).glob(f'_log*.csv')
    else:
        files = Path(path_to_csvs).glob(f'_log_{dataset}*.csv')

    li = []
    for filename in files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    df = pd.concat(li, axis=0, ignore_index=True)
    df = df.dropna(subset=["datamodule/coarsening_ratio"])

    data_names = [to_dataset_repr(d.split("-")[0], "paper")
                  for d in df["datamodule/dataset_subname"].to_numpy()]
    ds_names = ["Co" + to_s2n_repr(s if s == "sub_s2n" else "s2n")
                for s in df["datamodule/subgraph_batching"].to_numpy()]

    kws = dict(
        path=FIGURE_PATH,
        key=f"performance_{dataset}",
        extension=extension,
        markers=True, dashes=False,
        aspect=1.0,
        markersize=13, alpha=0.8,
        hues=ds_names, hue_name="Data structure",
        styles=ds_names, style_name="Data structure",
        cols=df["datamodule/custom_splits"].to_numpy(), col_name="# samples / class",
    )

    if dataset == "EMUser":
        xticks = [0.8, 0.9]
    elif dataset == "PPIBP":
        xticks = [0.2, 0.3, 0.4, 0.5, 0.6]

    plot_line(
        xs=df["datamodule/coarsening_ratio"].to_numpy(),
        xlabel="Coarsening ratio",
        ys=df["mean/test/micro_f1"].to_numpy(),
        ylabel="Performance",
        xticks=xticks,
        # legend=True,
        **kws,
    )


def visualize_efficiency_by_num_training(csv_path, extension="png", dataset=None):
    sns.set_palette(METHOD_COLORS)

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Performance", "Throughput (Train)"])

    metric_y_ticks = {
        "EM-User": {
            "Performance": [0.5, 0.6, 0.7, 0.8, 0.9],
            "Max Allocated GPU Memory (MB)": [1e2, 1e3, 1e4],
            "Throughput (Train)": [1e2, 1e3, 1e4],
            "Latency (Train)": [0.0, 0.1, 0.2],
            "Throughput (Eval)": [1e2, 1e3, 1e4],
            "Latency (Eval)": [0.0, 0.05, 0.1, 0.15],
        },
        "PPI-BP": {
            "Performance": [0.3, 0.4, 0.5],
            "Max Allocated GPU Memory (MB)": [1e1, 1e2, 1e3],
            "Throughput (Train)": [1e3, 1e4],
            "Latency (Train)": [0.0, 0.01, 0.02, 0.03],
            "Throughput (Eval)": [1e4, 2 * 1e4, 3 * 1e4],
            "Latency (Eval)": [0.0, 0.01, 0.02],
        }
    }

    if dataset != "all":  # dataset name
        df = df[df.Dataset == dataset]

        kws = dict(
            path=FIGURE_PATH,
            key=f"efficiency_{dataset.replace('-', '_')}",
            extension=extension,
            markers=True, dashes=False,
            aspect=1.0,
            markersize=13, alpha=0.8,
            xticks=[5, 10, 20, 30, 40],
            hues=df["Data structure"].to_numpy(), hue_name="Data structure",
            styles=df["Data structure"].to_numpy(), style_name="Data structure",
        )
        plot_line(
            xs=df["#train/C"].to_numpy(),
            xlabel="# training samples / class",
            ys=df["Performance"].to_numpy(),
            ylabel="Performance",

            yticks=metric_y_ticks[dataset]["Performance"],
            legend=False,
            **kws,
        )
        plot_line(
            xs=df["#train/C"].to_numpy(),
            xlabel="# training samples / class",
            ys=df["Max Allocated GPU Memory (MB)"].to_numpy(),
            ylabel="Max Allocated VRAM (MB)",

            scales_kws={"yscale": "log"},
            yticks=metric_y_ticks[dataset]["Max Allocated GPU Memory (MB)"],
            # legend=False,  NOTE: # is necessary
            **kws,
        )
        plot_line(
            xs=df["#train/C"].to_numpy(),
            xlabel="# training samples / class",
            ys=df["Throughput (Train)"].to_numpy().astype(float),
            ylabel="Train Throughput (#/s)",

            scales_kws={"yscale": "log"},
            yticks=metric_y_ticks[dataset]["Throughput (Train)"],
            legend=False,
            **kws,
        )
        plot_line(
            xs=df["#train/C"].to_numpy(),
            xlabel="# training samples / class",
            ys=df["Latency (Train)"].to_numpy().astype(float),
            ylabel="Train Latency (s/forward)",

            # scales_kws={"yscale": "log"},
            yticks=metric_y_ticks[dataset]["Latency (Train)"],
            legend=False,
            **kws,
        )
        plot_line(
            xs=df["#train/C"].to_numpy(),
            xlabel="# training samples / class",
            ys=df["Throughput (Eval)"].to_numpy().astype(float),
            ylabel="Eval Throughput (#/s)",

            scales_kws={"yscale": "log"},
            yticks=metric_y_ticks[dataset]["Throughput (Eval)"],
            legend=False,
            **kws,
        )
        plot_line(
            xs=df["#train/C"].to_numpy(),
            xlabel="# training samples / class",
            ys=df["Latency (Eval)"].to_numpy().astype(float),
            ylabel="Eval Latency (s/forward)",

            # scales_kws={"yscale": "log"},
            yticks=metric_y_ticks[dataset]["Latency (Eval)"],
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
            xs=df["#train/C"].to_numpy(),
            xlabel="# training samples / class",
            ys=df["Performance"].to_numpy(),
            ylabel="Performance",

            yticks=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            **kws,
        )
        plot_line(
            xs=df["#train/C"].to_numpy(),
            xlabel="# training samples / class",
            ys=df["Max Allocated GPU Memory (MB)"].to_numpy(),
            ylabel="Max Allocated VRAM (MB)",

            scales_kws={"yscale": "log"},
            yticks=[1e2, 1e3, 1e4],
            **kws,
        )
        plot_line(
            xs=df["#train/C"].to_numpy(),
            xlabel="# training samples / class",
            ys=df["Throughput (Train)"].to_numpy(),
            ylabel="Train Throughput (#/s)",

            scales_kws={"yscale": "log"},
            yticks=[1e2, 1e3, 1e4, 1e5],
            **kws,
        )
        plot_line(
            xs=df["#train/C"].to_numpy(),
            xlabel="# training samples / class",
            ys=df["Latency (Train)"].to_numpy(),
            ylabel="Train Latency (s/forward)",

            # yticks=[1e2, 1e3, 1e4, 1e5],
            **kws,
        )


if __name__ == '__main__':

    try:
        sns.set(style="whitegrid")
        sns.set_context("poster")
    except NameError:
        pass

    # analyze_s2n_properties, visualize_s2n_properties,
    # visualize_efficiency, visualize_efficiency_by_num_training
    # visualize_performance_by_coarsening_ratio
    METHOD = "visualize_s2n_properties"

    TARGETS = "REAL_WORLD"  # SYNTHETIC, REAL_WORLD, ALL
    if TARGETS == "REAL_WORLD":
        DATASET_NAME_LIST = ["PPIBP", "HPONeuro", "HPOMetab", "EMUser"]
    else:
        raise ValueError(f"Wrong targets: {TARGETS}")

    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    E_TYPE = "gin"  # gin, graphsaint_gcn
    USE_SPARSE_TENSOR = False

    DMS_NAME_LIST = list(product(DATASET_NAME_LIST,
                                 ["gcn", "gcn2"],
                                 ["s2n", "sub_s2n"]))
    if METHOD == "analyze_s2n_properties":
        analyze_s2n_properties(
            dataset_path=PATH,
            out_path="./_data_analysis.csv",
            dataset_model_s2n_name_list=DMS_NAME_LIST,
        )
    elif METHOD == "visualize_s2n_properties":
        sns.set_context("talk")
        visualize_s2n_properties(
            dataset_path=PATH,
            csv_path="./_data_analysis_w_original.csv",
            dataset_model_s2n_name_list=DMS_NAME_LIST,
            extension="pdf",
            run_analysis=False,  # NOTE: True to run analyze_s2n_properties
        )
    elif METHOD == "visualize_efficiency":
        sns.set_context("talk")
        visualize_efficiency(
            csv_path="./_sub2node Table (new3) - tab_efficiency.csv",
            extension="pdf",
            queries=[
                "dkey in ['hpo_metab', 'hpo_neuro', 'em_user', 'ppi_bp']",
                "mkey_s in ['gcn', 'gcn2', 'subgnn', 'glass']",  # NOTE: here!
            ],
        )
    elif METHOD == "visualize_efficiency_by_num_training":
        sns.set_context("talk")
        visualize_efficiency_by_num_training(
            csv_path="./_sub2node Table (new3) - tab_efficiency_by_num_training.csv",
            extension="pdf",
            dataset="EM-User",  # all, EM-User, PPI-BP
        )
    elif METHOD == "visualize_performance_by_coarsening_ratio":
        sns.set_context("poster")
        EXT = "pdf"
        visualize_performance_by_coarsening_ratio(
            "../_aggr_logs/_logs_csv_coarsening/by ratio",
            extension=EXT,
            dataset="EMUser",
        )
        visualize_performance_by_coarsening_ratio(
            "../_aggr_logs/_logs_csv_coarsening/by ratio",
            extension=EXT,
            dataset="PPIBP",
        )
