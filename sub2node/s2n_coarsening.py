import base64
import hashlib
from pprint import pprint
from typing import List, Union, Callable, Optional, Tuple
from collections import Counter
import os

import torch
from termcolor import cprint
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected, subgraph
from torch_scatter import scatter
from tqdm import tqdm

from data_sub import SubgraphDataset
from s2n_coarsening_utils import coarsening_pyg, coarsening_pyg_batch
from sub2node import SubgraphToNode, dist_by_shared_nodes
from utils import repr_kvs
from utils_fscache import fscaches
from visualize import plot_hist_from_counter, plot_kde_from_counter


class SubgraphToNodePlusCoarsening(SubgraphToNode):
    METHOD_NAMES = ["variation_neighborhoods", "variation_edges", "variation_cliques",
                    "heavy_edge", "algebraic_JC", "affinity_GS", "kron"]
    METHOD_NAMES_SCALABLE = ["variation_neighborhoods", "variation_edges",  # "variation_cliques",
                             "heavy_edge", "algebraic_JC", "affinity_GS", "kron"]
    METHOD_NAMES_SELECTED = ["variation_neighborhoods", "variation_edges"]

    def __init__(self,
                 global_data: Data,
                 subgraph_data_list: List[Data],
                 name: str,
                 path: str,
                 splits: List[int],

                 coarsening_ratio: float,
                 coarsening_method: str,
                 min_num_node_for_coarsening: Optional[int] = None,

                 num_start: int = 0,
                 target_matrix: str = "adjacent_no_self_loops",
                 edge_aggr: Union[Callable[[Tensor], Tensor], str] = None,
                 num_workers: int = None,
                 undirected: bool = None):
        """
        :param global_data: Single Data(edge_index=[2, *], x=[*, F])
        :param subgraph_data_list: List of Data(x=[*, 1], edge_index=[2, *], y=[1])
        :param splits: [num_train, num_train + num_val]

          num_start
          ↓  [+] num_train
          ↓   ↓  [+] num_train + num_coarsened_nodes
          ↓   ↓   ↓  [+] num_train + num_coarsened_nodes + num_val
          ↓   ↓   ↓   ↓     num_subgraphs
          ↓   ↓   ↓   ↓     ↓
        @ @ @ X X # # + + +
        @ @ @ X X # # + + +
        @ @ @ X X # # + + +
        X X X X X # # + + +
        X X X X X # # + + +
        # # # # # # # + + +
        # # # # # # # + + +
        + + + + + + + + + +
        + + + + + + + + + +
        + + + + + + + + + +
        """
        self.global_data = global_data
        self.path = path

        self.coarsening_ratio = round(coarsening_ratio, 1)
        self.coarsening_method = coarsening_method
        self.min_num_node_for_coarsening = min_num_node_for_coarsening

        subgraph_data_list, splits, self.meta_info = self.process_args_by_coarsening(subgraph_data_list, splits)
        super().__init__(global_data, subgraph_data_list, name, path, splits, num_start, target_matrix, edge_aggr,
                         num_workers, undirected, node_spl_cutoff=None)

    @property
    def node_task_name(self):
        return f"{super().node_task_name}-{self.coarsening_method}" \
               f"-{self.coarsening_ratio}-{self.min_num_node_for_coarsening}"

    def process_args_by_coarsening(self, subgraph_data_list, splits):
        for sd in subgraph_data_list:
            del sd.split
        num_train, num_val = splits[0], splits[1] - splits[0]
        data_train = subgraph_data_list[:num_train]
        data_val = subgraph_data_list[num_train:num_train + num_val]
        data_test = subgraph_data_list[num_train + num_val:]

        data_batch_coarsened, meta_info = self.generate_and_cache_subgraphs_by_coarsening(
            path=os.path.join(self.path, "../s2n_coarsening_cache"),
            data=self.global_data,
            coarsening_ratio=self.coarsening_ratio,
            coarsening_method=self.coarsening_method,
            min_num_node_for_coarsening=self.min_num_node_for_coarsening,
        )
        data_coarsened = Batch.to_data_list(data_batch_coarsened)

        num_coarsened_nodes = len(data_coarsened)
        num_living_nodes_after_coarsening = sum(d.x.size(0) for d in data_coarsened)

        cprint(f"\t- num_nodes_after_coarsening: {meta_info['num_nodes_after_coarsening']}", "yellow")
        cprint(f"\t- num_living_nodes_after_coarsening: {num_living_nodes_after_coarsening}", "yellow")
        cprint(f"\t- num_coarsened_nodes: {num_coarsened_nodes}", "yellow")

        new_subgraph_data_list = data_train + data_coarsened + data_val + data_test
        new_splits = [
            num_train,
            num_train + num_coarsened_nodes,
            num_train + num_coarsened_nodes + num_val,
        ]
        assert str(data_coarsened[0]) == str(new_subgraph_data_list[new_splits[0]])
        return new_subgraph_data_list, new_splits, meta_info

    @staticmethod
    @fscaches(path_attrname_in_kwargs="path", verbose=True)
    def generate_and_cache_subgraphs_by_coarsening(
            path, data: Data, coarsening_ratio, coarsening_method,
            min_num_node_for_coarsening,
    ) -> (Batch, dict):

        data_coarsened = SubgraphToNodePlusCoarsening.generate_subgraphs_by_coarsening(
            data, coarsening_ratio, coarsening_method)

        num_nodes_after_coarsening = Counter([d.x.size(0) for d in data_coarsened])

        if min_num_node_for_coarsening is not None:
            data_coarsened = [d for d in data_coarsened if d.x.size(0) >= min_num_node_for_coarsening]

        data_batch_coarsened = Batch.from_data_list(data_coarsened)
        return (
            data_batch_coarsened,
            {"num_nodes_after_coarsening": num_nodes_after_coarsening}
        )

    @staticmethod
    def generate_subgraphs_by_coarsening(data: Data, coarsening_ratio, coarsening_method):
        assert coarsening_method in SubgraphToNodePlusCoarsening.METHOD_NAMES
        print(f"Target Data: {data}")
        x_ids, batch, sub_x_index = coarsening_pyg_batch(data, coarsening_ratio, coarsening_method)
        sub_x = x_ids[sub_x_index]

        num_nodes_per_coarsened_nodes = scatter(torch.ones(batch.size(0)).long(), batch, dim=0, reduce="sum")
        sizes = [0] + torch.cumsum(num_nodes_per_coarsened_nodes, dim=0).tolist()

        new_subgraph_data_list = []
        for i in range(len(sizes) - 1):
            sz_i, sz_j = sizes[i], sizes[i + 1]
            x = sub_x[sz_i:sz_j].view(-1, 1)
            edge_index, _ = subgraph(x, data.edge_index)
            new_subgraph_data_list.append(Data(
                x=x, edge_index=edge_index, y=torch.Tensor([-1]).long(),
            ))
        return new_subgraph_data_list


def analyze_coarsening_results(
        dataset_name,
        s2n_list: List[SubgraphToNodePlusCoarsening],
        extension="png",
):
    info_list = []
    ratio_list, method_list = [], []
    for s2n in s2n_list:
        info_list.append(s2n.meta_info)
        ratio_list.append(s2n.coarsening_ratio)
        method_list.append(s2n.coarsening_method)

    for ratio in set(ratio_list):
        kws = dict(
            path="../_figures", extension=extension,
            key=f"num_nodes_after_coarsening_{dataset_name}_{ratio}",
            x_counter_list=[s2n.meta_info["num_nodes_after_coarsening"] for s2n in s2n_list
                            if s2n.coarsening_ratio == ratio],
            cols=[s2n.coarsening_method for s2n in s2n_list if s2n.coarsening_ratio == ratio], col_name="method",
        )
        # plot_kde_from_counter(xlabel="# nodes / subgraph", **kws)
        plot_kde_from_counter(
            xlabel="Log(#) nodes / subgraph",
            scales_kws={"xscale": "log"},
            **kws,
        )
    for meth in set(method_list):
        kws = dict(
            path="../_figures", extension=extension,
            key=f"num_nodes_after_coarsening_{dataset_name}_{meth}",
            x_counter_list=[s2n.meta_info["num_nodes_after_coarsening"] for s2n in s2n_list
                            if s2n.coarsening_method == meth],
            cols=[str(s2n.coarsening_ratio) for s2n in s2n_list if s2n.coarsening_method == meth], col_name="ratio",
        )
        # plot_kde_from_counter(xlabel="# nodes / subgraph", **kws)
        plot_kde_from_counter(
            xlabel="Log(#) nodes / subgraph",
            scales_kws={"xscale": "log"},
            **kws,
        )


if __name__ == "__main__":

    from data_sub import HPOMetab, HPONeuro, PPIBP, EMUser, Density, Component, Coreness, CutRatio

    MODE = "node_task_data_splits"
    # CROSS, NUM_TRAIN_PER_CLASS, SAVE_PRECURSOR, analyze_coarsening_results, node_task_data_splits

    NAME = "PPIBP"
    # TEST
    # PPIBP, HPOMetab, HPONeuro, EMUser
    # Density, Component, Coreness, CutRatio, WLKSRandomTree

    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    TARGET_MATRIX = "adjacent_with_self_loops"

    if NAME.startswith("WL"):
        E_TYPE = "no_embedding"
    elif NAME in ["Density", "Component", "Coreness", "CutRatio"]:
        E_TYPE = "one"
    else:
        E_TYPE = "glass"  # gin, graphsaint_gcn, glass
    DEBUG = False

    print(f"NAME={NAME} | MODE={MODE}")
    if MODE == "analyze_coarsening_results":
        dts: SubgraphDataset = eval(NAME)(root=PATH, name=NAME, embedding_type=E_TYPE, debug=DEBUG)
        _subgraph_data_list = dts.get_data_list_with_split_attr()
        _global_data = dts.global_data

        _s2n_list = []
        for cr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for method in SubgraphToNodePlusCoarsening.METHOD_NAMES_SELECTED:  # NOTE: METHOD_NAMES_SELECTED
                _s2n = SubgraphToNodePlusCoarsening(
                    _global_data, _subgraph_data_list,
                    coarsening_ratio=cr,
                    coarsening_method=method,
                    min_num_node_for_coarsening=2,  # NOTE: important
                    name=NAME,
                    path=f"{PATH}/{NAME.upper()}/sub2node_coarsening/",
                    undirected=True,
                    splits=dts.splits,
                    target_matrix=TARGET_MATRIX,
                )
                _s2n_list.append(_s2n)

        analyze_coarsening_results(NAME, _s2n_list)

    elif MODE == "SAVE_PRECURSOR":
        dts: SubgraphDataset = eval(NAME)(
            root=PATH, name=NAME, embedding_type=E_TYPE, debug=DEBUG,
            num_training_tails_to_tile_per_class=40,  # NOTE: IMPORTANT
        )
        _subgraph_data_list = dts.get_data_list_with_split_attr()
        _global_data = dts.global_data

        for cr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for method in SubgraphToNodePlusCoarsening.METHOD_NAMES_SELECTED:  # NOTE: METHOD_NAMES_SELECTED
                for usei in [True, False]:
                    _s2n = SubgraphToNodePlusCoarsening(
                        _global_data, _subgraph_data_list,
                        coarsening_ratio=cr,
                        coarsening_method=method,
                        min_num_node_for_coarsening=2,  # NOTE: important
                        name=NAME,
                        path=f"{PATH}/{NAME.upper()}/sub2node_coarsening/",
                        splits=dts.splits,
                        num_start=dts.num_start,
                        target_matrix=TARGET_MATRIX,
                        edge_aggr=None,
                        undirected=True,
                    )
                    _s2n.node_task_data_precursor(
                        matrix_type="unnormalized",
                        use_sub_edge_index=usei,
                        save=True,
                    )

    elif MODE == "NUM_TRAIN_PER_CLASS":
        dts: SubgraphDataset = eval(NAME)(
            root=PATH, name=NAME, embedding_type=E_TYPE, debug=DEBUG,
            num_training_tails_to_tile_per_class=40,  # NOTE: IMPORTANT
        )

        # ----- NOTE: IMPORTANT
        dts.set_num_start_train_by_num_train_per_class(5)
        # -----

        _subgraph_data_list = dts.get_data_list_with_split_attr()
        _global_data = dts.global_data

        cr = 0.3
        _s2n_list = []
        for method in SubgraphToNodePlusCoarsening.METHOD_NAMES_SELECTED:  # NOTE: METHOD_NAMES_SELECTED
            _s2n = SubgraphToNodePlusCoarsening(
                _global_data, _subgraph_data_list,
                coarsening_ratio=cr,
                coarsening_method=method,
                min_num_node_for_coarsening=2,  # NOTE: important
                name=NAME,
                path=f"{PATH}/{NAME.upper()}/sub2node_coarsening/",
                splits=dts.splits,
                num_start=dts.num_start,
                target_matrix=TARGET_MATRIX,
                edge_aggr=None,
                undirected=True,
            )
            data_list = _s2n.node_task_data_splits(
                mapping_matrix_type="unnormalized",
                set_sub_x_weight=None,
                use_sub_edge_index=False,
                post_edge_normalize="standardize_then_trunc_thres_max_linear",
                post_edge_normalize_args=[1.0, 1.0],
                edge_thres=0.0,
                use_consistent_processing=True,
                save=True,
                load=False,
            )

    elif NAME != "TEST" and MODE == "CROSS":
        dts: SubgraphDataset = eval(NAME)(
            root=PATH,
            name=NAME,
            embedding_type=E_TYPE,
            debug=DEBUG,
        )
        _subgraph_data_list = dts.get_data_list_with_split_attr()
        _global_data = dts.global_data

        for cr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for method in SubgraphToNodePlusCoarsening.METHOD_NAMES_SELECTED:  # NOTE: METHOD_NAMES_SELECTED
                _s2n = SubgraphToNodePlusCoarsening(
                    _global_data, _subgraph_data_list,
                    coarsening_ratio=cr,
                    coarsening_method=method,
                    min_num_node_for_coarsening=2,  # NOTE: important
                    name=NAME,
                    path=f"{PATH}/{NAME.upper()}/sub2node_coarsening/",
                    undirected=True,
                    splits=dts.splits,
                    target_matrix=TARGET_MATRIX,
                )
    elif MODE == "node_task_data_splits":
        _cls = eval(NAME)
        dts = _cls(root=PATH, name=NAME, debug=DEBUG, embedding_type=E_TYPE)
        _subgraph_data_list = dts.get_data_list_with_split_attr()
        _global_data = dts.global_data

        for cr in [0.2, 0.4, 0.6, 0.8]:
            _s2n = SubgraphToNodePlusCoarsening(
                _global_data, _subgraph_data_list,
                coarsening_ratio=cr,
                coarsening_method="variation_neighborhoods",  # NOTE: important
                min_num_node_for_coarsening=2,
                name=NAME,
                path=f"{PATH}/{NAME.upper()}/sub2node_coarsening/",
                splits=dts.splits,
                num_start=dts.num_start,
                target_matrix=TARGET_MATRIX,
                edge_aggr=None,
                undirected=True,
            )
            print(_s2n)
            for ssxw in [None, "original_sqrt_d_node_div_d_sub"]:
                for usei in [True, False]:
                    # unnormalized, sqrt_d_node_div_d_sub, original_sqrt_d_node_div_d_sub
                    # standardize_then_trunc_thres_max_linear
                    for i in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]:
                        for j in [0.5, 1.0, 1.5, 2.0]:
                            ntds = _s2n.node_task_data_splits(
                                mapping_matrix_type="unnormalized",
                                set_sub_x_weight=ssxw,
                                use_sub_edge_index=usei,
                                post_edge_normalize="standardize_then_trunc_thres_max_linear",
                                post_edge_normalize_args=[i, j],
                                edge_thres=0.0,
                                use_consistent_processing=True,
                                save=True,
                            )
                            for _d in ntds:
                                print(_d)
                                print(f"\t- density: {_d.edge_index.size(1) / (_d.num_nodes ** 2)}")
                                if hasattr(_d, "sub_x_weight"):
                                    _sub_x_weight_stats = repr_kvs(
                                        min=torch.min(_d.sub_x_weight), max=torch.max(_d.sub_x_weight),
                                        avg=torch.mean(_d.sub_x_weight), std=torch.std(_d.sub_x_weight), sep=", ")
                                    print(f"\t- sub_x_weight: {_sub_x_weight_stats}")
                            _s2n._node_task_data_list = []  # flush
    else:
        # 0           5
        # | > 2 - 3 < |
        # 1           4
        E1 = to_undirected(torch.tensor([[0, 1, 2, 3, 4, 5, 2],
                                         [1, 2, 0, 4, 5, 3, 3]]))

        #             7
        # 6 - 2 - 3 < |
        #             4
        E3 = to_undirected(torch.tensor([[6, 3, 4, 7, 2],
                                         [2, 4, 7, 3, 3]]))

        #             7
        # 6 - 3 - 2 < |
        #             4
        E4 = to_undirected(torch.tensor([[6, 3, 2, 2, 7],
                                         [3, 2, 7, 4, 4]]))

        # 0           5                 13
        # | > 2 - 3 < |    12 - 9 - 8 < |
        # 1           4                 10
        E5 = torch.cat([E1, E4 + 6], dim=1)

        # 0           5                 13                  27
        # | > 2 - 3 < |    12 - 9 - 8 < |    26 - 22 - 23 < |
        # 1           4                 10                  24
        E6 = torch.cat([E1, E4 + 6, E3 + 20], dim=1)
        TEST_DATA = Data(edge_index=E6.long())

        _data_batch, _meat_info = SubgraphToNodePlusCoarsening.generate_and_cache_subgraphs_by_coarsening(
            path="../fscaches",
            data=TEST_DATA,
            coarsening_ratio=0.7,
            coarsening_method="variation_neighborhoods",
            min_num_node_for_coarsening=1,
        )
        print(_data_batch)
        print(
            Batch.to_data_list(_data_batch)
        )
