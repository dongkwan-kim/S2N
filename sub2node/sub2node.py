import os
import multiprocessing as mp
from collections import Counter
from pathlib import Path
from pprint import pprint
from typing import List, Callable, Union, Tuple, Dict

import numpy as np
from termcolor import cprint
import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.utils import (to_networkx, from_networkx, is_undirected, dense_to_sparse,
                                   add_remaining_self_loops, remove_self_loops, to_dense_adj)
from tqdm import tqdm

from utils import to_symmetric_matrix, try_getattr, spspmm_quad


class SubgraphToNode:
    _global_nxg = None
    _node_spl_mat = None
    _node_task_data_precursor = None

    def __init__(self,
                 global_data: Data,
                 subgraph_data_list: List[Data],
                 name: str, path: str,
                 splits: List[int],
                 target_matrix: str = "adjacent_no_self_loops",
                 edge_aggr: Union[Callable[[Tensor], Tensor], str] = None,
                 num_workers: int = None,
                 undirected: bool = None,
                 node_spl_cutoff=None):
        """
        :param global_data: Single Data(edge_index=[2, *], x=[*, F])
        :param subgraph_data_list: List of Data(x=[*, 1], edge_index=[2, *], y=[1])
        :param node_spl_cutoff:
        """
        self.global_data: Data = global_data
        self.subgraph_data_list: List[Data] = subgraph_data_list
        self.name: str = name
        self.path: Path = Path(path)
        self.splits = splits + [len(self.subgraph_data_list)]

        self.target_matrix = target_matrix
        self.edge_aggr = self.parse_edge_aggr(edge_aggr)

        self.num_workers = num_workers
        self.undirected = undirected or is_undirected(global_data.edge_index)
        self.node_spl_cutoff = node_spl_cutoff

        assert self.target_matrix in [
            "adjacent", "adjacent_with_self_loops", "adjacent_no_self_loops", "shortest_path"
        ]
        assert self.undirected, "Now only support undirected graphs"
        assert len(self.splits) == 3
        self.path.mkdir(exist_ok=True)
        self._node_task_data_list: List[Data] = []

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.node_task_name}', path='{self.path}')"

    def parse_edge_aggr(self, edge_aggr):
        if isinstance(edge_aggr, str):
            return eval(edge_aggr)
        else:
            return edge_aggr or torch.min

    @property
    def node_task_name(self):
        if self.target_matrix.startswith("adjacent"):
            return f"{self.name}-ADJ-{self.target_matrix}"
        else:
            return f"{self.name}-SP-EA-{self.edge_aggr.__name__}"

    @property
    def S(self):
        return len(self.subgraph_data_list)

    @property
    def N(self):
        return self.global_data.num_nodes

    @property
    def global_nxg(self) -> nx.Graph:
        if self._global_nxg is None:
            self._global_nxg = to_networkx(self.global_data)
        return self._global_nxg

    def single_source_shortest_path_length_for_global_data(self, n):
        spl_dict = nx.single_source_shortest_path_length(
            self.global_nxg, n, cutoff=self.node_spl_cutoff)
        spl_list = [val for node, val in sorted(spl_dict.items(), key=lambda t: t[0])]
        return spl_list

    def all_pairs_shortest_path_length_for_global_data(self):
        if self.num_workers is not None:
            with mp.Pool(processes=self.num_workers) as pool:
                shortest_paths = pool.map(self.single_source_shortest_path_length_for_global_data,
                                          self.global_nxg.nodes)
        else:
            shortest_paths = [self.single_source_shortest_path_length_for_global_data(n)
                              for n in tqdm(self.global_nxg.nodes)]
        return torch.tensor(shortest_paths, dtype=torch.long)

    def node_spl_mat(self, save=True):
        path = self.path / f"{self.name}_spl_mat.pth"
        try:
            self._node_spl_mat = torch.load(path)
            cprint(f"Load: tensor of {self._node_spl_mat.size()} at {path}", "green")
            return self._node_spl_mat
        except FileNotFoundError:
            pass
        self._node_spl_mat = self.all_pairs_shortest_path_length_for_global_data()
        if save:
            torch.save(self._node_spl_mat, path)
            cprint(f"Save: tensor of {self._node_spl_mat.size()} at {path}", "blue")
        return self._node_spl_mat

    def node_task_data_precursor(self, save=True):
        path = self.path / f"{self.node_task_name}_node_task_data_precursor.pth"
        try:
            self._node_task_data_precursor = torch.load(path)
            cprint(f"Load: {self._node_task_data_precursor} at {path}", "green")
            return self._node_task_data_precursor
        except FileNotFoundError:
            pass

        # Node aggregation: x, y, batch, ...
        # DataBatch(x=[16236, 1], y=[1591], split=[1591], batch=[16236], ptr=[1592])
        self._node_task_data_precursor = Batch.from_data_list(self.subgraph_data_list)
        del self._node_task_data_precursor.edge_index

        # Edge aggregation
        if self.target_matrix.startswith("adjacent"):
            self._node_task_data_precursor.edge_weight_matrix = self.get_ewmat_by_multiplying_adj()
        elif self.target_matrix == "shortest_path":
            self._node_task_data_precursor.edge_weight_matrix = self.get_ewmat_by_aggregating_sub_spl_mat(save)

        if save:
            torch.save(self._node_task_data_precursor, path)
            cprint(f"Save: {self._node_task_data_precursor} at {path}", "blue")

        return self._node_task_data_precursor

    def get_ewmat_by_multiplying_adj(self):
        # Mapping matrix M (sxn) construction
        # batch = subgraph ids, x = node ids
        m_index = torch.stack([self._node_task_data_precursor.batch,
                               self._node_task_data_precursor.x.squeeze(-1)]).long()
        # Adjacent matrix A (nxn)
        if self.target_matrix == "adjacent_with_self_loops":
            a_index, _ = add_remaining_self_loops(self.global_data.edge_index)
        elif self.target_matrix == "adjacent_no_self_loops":
            a_index, _ = remove_self_loops(self.global_data.edge_index)
        else:
            a_index = self.global_data.edge_index

        m_value = torch.ones(m_index.size(1))
        a_value = torch.ones(a_index.size(1))

        # unnormalized_ewmat = M * A * M^T
        unnorm_ewmat_index, unnorm_ewmat_value = spspmm_quad(
            m_index, m_value, a_index, a_value, self.S, self.N, coalesced=True)
        dense_unnorm_ewmat = to_dense_adj(
            unnorm_ewmat_index, edge_attr=unnorm_ewmat_value).squeeze()
        return dense_unnorm_ewmat

    def get_ewmat_by_aggregating_sub_spl_mat(self, save):
        # sub_spl_ij = min { d_uv | u \in S_i, v in S_j }
        node_spl_mat = self.node_spl_mat(save).float()
        sub_spl_mat = torch.full((self.S, self.S), fill_value=-1)
        for i, sub_data_i in enumerate(tqdm(self.subgraph_data_list,
                                            desc="get_ewmat_by_aggregating_sub_spl_mat")):
            for j, sub_data_j in enumerate(self.subgraph_data_list):
                if self.undirected and i <= j:
                    x_i = sub_data_i.x.squeeze(-1)
                    x_j = sub_data_j.x.squeeze(-1)
                    sub_spl = self.edge_aggr(node_spl_mat[x_i, :][:, x_j])
                    sub_spl_mat[i, j] = sub_spl
                    sub_spl_mat[j, i] = sub_spl

        # edge = 1 / (spl + 1) where 0 <= spl, then 0 < edge <= 1
        return 1 / (sub_spl_mat + 1)

    def node_task_data_splits(self,
                              edge_normalize: Union[str, Callable, None] = None,
                              edge_normalize_args: Union[List, None] = None,
                              edge_thres: Union[float, Callable, List[float]] = 1.0,
                              use_consistent_processing=False,
                              save=True) -> Tuple[Data, Data, Data]:
        """
        :return: Data(x=[N, 1], edge_index=[2, E], edge_attr=[E], y=[C], batch=[N])
            - N is the number of subgraphs = batch.sum()
            - edge_attr >= edge_thres
        """
        edge_normalize_args = edge_normalize_args or []
        if isinstance(edge_normalize, str):
            edge_normalize = func_normalize(edge_normalize, *edge_normalize_args)
        str_et = edge_thres.__name__ if isinstance(edge_thres, Callable) else edge_thres
        str_en = '-'.join([edge_normalize.__name__ if isinstance(edge_normalize, Callable) else edge_normalize] +
                          [str(round(a, 3)) for a in edge_normalize_args])  # todo: general repr for args
        path = self.path / (f"{self.node_task_name}_node_task_data"
                            f"_et={str_et}_en={str_en}_ucp={use_consistent_processing}.pth")
        try:
            self._node_task_data_list = torch.load(path)
            cprint(f"Load: {self._node_task_data_list} at {path}", "green")
            return self._node_task_data_list
        except FileNotFoundError:
            pass

        if not isinstance(edge_thres, list):
            edge_thres = [edge_thres, edge_thres, edge_thres]
        assert len(edge_thres) == len(self.splits)

        node_task_data_precursor = self.node_task_data_precursor(save)
        ew_mat = node_task_data_precursor.edge_weight_matrix

        edge_norm_kws = {}
        for i, (s, et) in enumerate(zip(self.splits, edge_thres)):
            x, y, batch, ptr = try_getattr(node_task_data_precursor,
                                           ["x", "y", "batch", "ptr"], as_dict=False)
            sub_x = x[:ptr[s], :]
            sub_batch = batch[:ptr[s]]
            y = y[:s]

            num_nodes = y.size(0)
            eval_mask = None
            if i > 0:
                eval_mask = torch.zeros(num_nodes, dtype=torch.bool)
                eval_mask[self.splits[i - 1]:] = True
            else:
                self.print_mat_stat(ew_mat, "Summarizing edge_weight_matrix")

            ew_mat_s_by_s = ew_mat.clone()[:s, :s]
            if edge_normalize is not None:
                if use_consistent_processing:
                    ew_mat_s_by_s, edge_norm_kws = edge_normalize(ew_mat_s_by_s, **edge_norm_kws)
                else:
                    ew_mat_s_by_s, edge_norm_kws = edge_normalize(ew_mat_s_by_s)
            # Remove ew_mat below than edge_thres
            et = et(ew_mat_s_by_s) if isinstance(et, Callable) else et
            ew_mat_s_by_s[ew_mat_s_by_s < et] = 0
            self.print_mat_stat(ew_mat_s_by_s, f"Summarizing processed edge_weight_matrix ({i})")

            edge_index, edge_attr = dense_to_sparse(ew_mat_s_by_s)
            self._node_task_data_list.append(Data(
                sub_x=sub_x, sub_batch=sub_batch, y=y, eval_mask=eval_mask,
                edge_index=edge_index, edge_attr=edge_attr.view(-1, 1),
                num_nodes=num_nodes,
            ))

        if save:
            torch.save(self._node_task_data_list, path)
            cprint(f"Save: {self._node_task_data_list} at {path}", "blue")

        return tuple(self._node_task_data_list)

    @staticmethod
    def print_mat_stat(matrix, start=None, print_counter=False):
        _decimal = 5
        _mean = lambda t: round(torch.mean(t).item(), _decimal)
        _std = lambda t: round(torch.std(t).item(), _decimal)
        _min = lambda t: round(torch.min(t).item(), _decimal)
        _median = lambda t: round(torch.median(t).item(), _decimal)
        _1q = lambda t: round(torch.quantile(t, 0.25).item(), _decimal)
        _3q = lambda t: round(torch.quantile(t, 0.75).item(), _decimal)
        _max = lambda t: round(torch.max(t).item(), _decimal)
        if start:
            cprint(start, "green")
        matrix_pos = matrix[matrix > 0]
        print(
            f"\tmean / std = {_mean(matrix)} / {_std(matrix)} \n"
            f"\tmin / 1q / median / 3q / max = {_min(matrix)} / {_1q(matrix)} / {_median(matrix)}"
            f" / {_3q(matrix)} / {_max(matrix)} \n"
            f"\tmean+ / std+ = {_mean(matrix_pos)} / {_std(matrix_pos)} \n"
            f"\tmin+ / 1q+ / median+ / 3q+ / max+ = {_min(matrix_pos)} / {_1q(matrix_pos)} / {_median(matrix_pos)}"
            f" / {_3q(matrix_pos)} / {_max(matrix_pos)} \n"
            f"\tN = {matrix.numel()}, N+ = {(matrix > 0).sum().item()}, "
            f"d = {(matrix > 0).sum().item() / matrix.numel()}"
        )
        if print_counter:
            print("\tCounters: ", Counter(matrix.flatten().tolist()))


def func_topk_thres(thres):
    def _func(x):
        k = int(x.numel() * thres)
        topk = torch.topk(x.flatten(), k, sorted=False).values
        return torch.min(topk).item()

    _func.__name__ = f"topk_{thres}"

    return _func


def dist_by_shared_nodes(node_spl_mat):
    non_shared_nodes = torch.count_nonzero(node_spl_mat)
    shared_nodes = node_spl_mat.numel() - non_shared_nodes
    # edge_weight = 1 / (1 + d) = 1 / (1 + -1 + (1 / shared_nodes)) = shared_nodes
    return -1 + (1 / shared_nodes)


def func_normalize(normalize_type: str, *args):
    def _func(matrix: Tensor, **kws) -> (Tensor, Dict):
        if len(kws) == 0:
            kws = {"mean": torch.mean(matrix),
                   "std": torch.std(matrix),
                   "max": torch.max(matrix)}
        if normalize_type == "standardize_then_thres_max_linear":
            assert len(args) == 1, f"Wrong args: {args}"
            thres = args[0]
            matrix = (matrix - kws["mean"]) / kws["std"]
            matrix = (matrix - thres) / (kws["max"] - thres)
        elif normalize_type == "standardize_then_trunc_thres_max_linear":
            assert len(args) == 2, f"Wrong args: {args}"
            assert args[1] > 0
            thres, trunc_diff = args[0], args[1]
            trunc_val = thres + trunc_diff
            matrix = (matrix - kws["mean"]) / kws["std"]
            matrix[matrix >= trunc_val] = trunc_val
            matrix = (matrix - thres) / (trunc_val - thres)
        elif normalize_type == "standardize_then_thres_max_power":
            assert len(args) == 2, f"Wrong args: {args}"
            thres, p = args[0], args[1]
            matrix = (matrix - kws["mean"]) / kws["std"]
            matrix = (matrix.relu_() ** p - thres ** p) / (kws["max"] ** p - thres ** p)
        else:
            raise ValueError(f"Wrong type: {normalize_type}")
        return matrix, kws

    _func.__name__ = f"normalize_{normalize_type}"

    return _func


if __name__ == '__main__':

    from data_sub import HPOMetab, HPONeuro, PPIBP, EMUser, Density, CC, Coreness, CutRatio

    MODE = "EMUser"
    # PPIBP, HPOMetab, HPONeuro, EMUser
    # Density, CC, Coreness, CutRatio
    PURPOSE = "MANY_2"
    # MANY, ONCE
    TARGET_MATRIX = "adjacent_with_self_loops"
    # adjacent_with_self_loops, adjacent_no_self_loops

    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    E_TYPE = "graphsaint_gcn"
    DEBUG = False

    if MODE in ["HPOMetab", "PPIBP", "HPONeuro", "EMUser",
                "Density", "CC", "Coreness", "CutRatio"]:
        _cls = eval(MODE)
        dts = _cls(root=PATH, name=MODE, debug=DEBUG, embedding_type=E_TYPE)
        _subgraph_data_list = dts.get_data_list_with_split_attr()
        _global_data = dts.global_data

        s2n = SubgraphToNode(
            _global_data, _subgraph_data_list,
            name=MODE,
            path=f"{PATH}/{MODE.upper()}/sub2node/",
            undirected=True,
            splits=dts.splits,
            target_matrix=TARGET_MATRIX,
            edge_aggr=dist_by_shared_nodes,
        )
        print(s2n)
        """ Inverse sigmoid table 0.5 -- 0.95,
        inv_sig = [0.0, 0.201, 0.405, 0.619, 0.847, 1.099, 1.386, 1.735, 2.197, 2.944]
        """
        if PURPOSE == "MANY_1":
            # standardize_then_thres_max_linear
            for i in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                      2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]:
                ntds = s2n.node_task_data_splits(
                    edge_normalize="standardize_then_thres_max_linear",
                    edge_normalize_args=[i],
                    edge_thres=0.0,
                    use_consistent_processing=True,
                    save=True,
                )
                for _d in ntds:
                    print(_d, "density", _d.edge_index.size(1) / (_d.num_nodes ** 2))
                s2n._node_task_data_list = []  # flush
        elif PURPOSE == "MANY_2":
            # standardize_then_trunc_thres_max_linear, standardize_then_thres_max_power
            for i in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                      2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]:
                for j in [0.5, 1.0, 1.5, 2.0]:
                    ntds = s2n.node_task_data_splits(
                        edge_normalize="standardize_then_trunc_thres_max_linear",
                        edge_normalize_args=[i, j],
                        edge_thres=0.0,
                        use_consistent_processing=True,
                        save=True,
                    )
                    for _d in ntds:
                        print(_d, "density", _d.edge_index.size(1) / (_d.num_nodes ** 2))
                    s2n._node_task_data_list = []  # flush
        elif PURPOSE == "ONCE":
            ntds = s2n.node_task_data_splits(
                edge_normalize="standardize_then_thres_max_linear",
                edge_normalize_args=[0.314],
                edge_thres=0.0,
                use_consistent_processing=False,
                save=True,
            )
            for _d in ntds:
                print(_d, "density", _d.edge_index.size(1) / (_d.num_nodes ** 2))
        else:
            raise ValueError(f"Wrong purpose: {PURPOSE}")

    elif MODE == "TEST":
        # Probably deprecated snippets
        _global_data = from_networkx(nx.path_graph(10))
        _subgraph_data_list = [
            Data(x=torch.Tensor([0, 1]).long().view(-1, 1),
                 edge_index=torch.Tensor([[0, 1],
                                          [1, 0]]).long(),
                 y=torch.Tensor([0])),
            Data(x=torch.Tensor([1, 2, 4, 5]).long().view(-1, 1),
                 edge_index=torch.Tensor([[1, 2, 4, 5],
                                          [2, 1, 5, 4]]).long(),
                 y=torch.Tensor([1])),
            Data(x=torch.Tensor([8, 9]).long().view(-1, 1),
                 edge_index=torch.Tensor([[8, 9], [9, 8]]).long(),
                 y=torch.Tensor([2])),
        ]

        s2n = SubgraphToNode(
            _global_data, _subgraph_data_list,
            name="test",
            path="./",
            undirected=True,
            splits=[1, 2],
            edge_aggr=torch.mean,
        )
        ntds = s2n.node_task_data_splits(
            # 0.25,
            func_topk_thres(0.25),
            save=False,
        )
        pprint(ntds)
        print(ntds[-2].eval_mask)
        print(ntds[-1].eval_mask)
        for _d in ntds:
            print(_d, "density", _d.edge_index.size(1) / (_d.num_nodes ** 2))
