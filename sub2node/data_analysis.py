import csv
from pprint import pprint

from termcolor import cprint
from torch_geometric.data import Data
from torch_geometric.utils import homophily, remove_self_loops

from data import SubgraphDataModule
from utils import multi_label_homophily


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


def analyze_node_properties(dataset_path, name_list, name_to_best_kwargs,
                            out_path=None):
    list_of_pps = []
    for name in name_list:
        sdm = SubgraphDataModule(
            dataset_name=name,
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
            **name_to_best_kwargs[name],
        )
        pps_dict = _analyze_node_properties(sdm.test_data)
        list_of_pps.append({"name": name, **pps_dict})
        pprint(list_of_pps)

    if out_path is not None:
        with open(out_path, "w") as f:
            cprint(f"Save properties at {out_path}", "blue")
            writer = csv.DictWriter(f, fieldnames=[k for k in list_of_pps[0].keys()])
            writer.writeheader()
            for pps in list_of_pps:
                writer.writerow(pps)

    return list_of_pps


if __name__ == '__main__':
    TARGETS = "REAL_WORLD"  # SYNTHETIC, REAL_WORLD, ALL
    if TARGETS == "REAL_WORLD":
        NAME_LIST = ["PPIBP", "HPONeuro", "HPOMetab", "EMUser"]
    else:
        raise ValueError(f"Wrong targets: {TARGETS}")

    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    E_TYPE = "gin"  # gin, graphsaint_gcn
    USE_SPARSE_TENSOR = False

    NAME_TO_BEST_KWARGS = {
        "PPIBP": dict(
            edge_normalize_arg_1=0.0,
            edge_normalize_arg_2=0.5,
            s2n_target_matrix="adjacent_with_self_loops",
        ),
        "HPONeuro": dict(
            edge_normalize_arg_1=2.25,
            edge_normalize_arg_2=2.0,
            s2n_target_matrix="adjacent_no_self_loops",
        ),
        "HPOMetab": dict(
            edge_normalize_arg_1=3.25,
            edge_normalize_arg_2=2.0,
            s2n_target_matrix="adjacent_with_self_loops",
        ),
        "EMUser": dict(
            edge_normalize_arg_1=0.75,
            edge_normalize_arg_2=1.0,
            s2n_target_matrix="adjacent_no_self_loops",
        ),
    }

    analyze_node_properties(
        dataset_path=PATH, out_path="./_data_analysis.csv",
        name_list=NAME_LIST,
        name_to_best_kwargs=NAME_TO_BEST_KWARGS,
    )
