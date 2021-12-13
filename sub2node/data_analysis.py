import csv
from pprint import pprint

from termcolor import cprint
from torch_geometric.data import Data
from torch_geometric.utils import homophily, remove_self_loops

from data import SubgraphDataModule


def _analyze_node_properties(data: Data):

    N = data.num_nodes

    properties = dict()

    if data.y.squeeze().dim() == 1:
        properties["homophily"] = homophily(data.edge_index, data.y)
        properties["num_classes"] = data.y.max().item() + 1
    else:
        properties["num_classes"] = data.y.size(1)

    edge_index, edge_attr = remove_self_loops(data.edge_index, getattr(data, "edge_attr", None))
    properties["density"] = edge_index.size(1) / (N * (N - 1))

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
    NAME = "Density"
    # PPIBP, HPOMetab, HPONeuro, EMUser
    # Density, CC, Coreness, CutRatio

    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    E_TYPE = "graphsaint_gcn"  # gin, graphsaint_gcn
    USE_SPARSE_TENSOR = False

    analyze_node_properties(
        dataset_path=PATH, out_path="./_data_analysis.csv",
        name_list=["PPIBP", "HPOMetab", "HPONeuro", "EMUser"],
        edge_thres_list=[1.0, 0.5],
        embedding_type=E_TYPE, use_sparse_tensor=USE_SPARSE_TENSOR,
    )
