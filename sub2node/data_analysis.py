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


def analyze_node_properties(dataset_path, name_list, edge_thres_list, out_path=None, **kwargs):
    list_of_pps = []
    for name in name_list:
        for edge_thres in edge_thres_list:
            sdm = SubgraphDataModule(
                dataset_name=name,
                dataset_path=dataset_path,
                use_s2n=True,
                edge_thres=edge_thres,
                # embedding_type=E_TYPE,
                # use_sparse_tensor=USE_SPARSE_TENSOR,
                **kwargs,
            )
            pps_dict = _analyze_node_properties(sdm.test_data)
            list_of_pps.append({"name": name, "edge_thres": edge_thres, **pps_dict})
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
    TARGETS = "ALL"  # SYNTHETIC, REAL_WORLD, ALL

    if TARGETS == "SYNTHETIC":
        NAME_LIST = ["Density", "CC", "Coreness", "CutRatio"]
    elif TARGETS == "REAL_WORLD":
        NAME_LIST = ["PPIBP", "HPOMetab", "HPONeuro", "EMUser"]
    else:
        NAME_LIST = ["PPIBP", "HPOMetab", "HPONeuro", "EMUser",
                     "Density", "CC", "Coreness", "CutRatio"]

    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    E_TYPE = "graphsaint_gcn"  # gin, graphsaint_gcn
    USE_SPARSE_TENSOR = False

    analyze_node_properties(
        dataset_path=PATH, out_path="./_data_analysis.csv",
        name_list=NAME_LIST,
        edge_thres_list=[1.0, 0.5],
        embedding_type=E_TYPE, use_sparse_tensor=USE_SPARSE_TENSOR,
    )
