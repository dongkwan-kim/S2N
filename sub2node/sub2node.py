import os
import multiprocessing as mp
from pathlib import Path
from typing import List, Callable

from termcolor import cprint
import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx, from_networkx, is_undirected, dense_to_sparse

from utils import to_symmetric_matrix


class SubgraphToNode:

    def __init__(self,
                 global_data, subgraph_data_list,
                 name, path,
                 edge_aggr: Callable[[Tensor], float] = None,
                 undirected=None,
                 node_spl_cutoff=None):
        """
        :param global_data: Single Data(edge_index=[2, *], x=[*, F])
        :param subgraph_data_list: List of Data(x=[*, 1], edge_index=[2, *], y=[1])
        :param node_spl_cutoff:
        """
        self.global_data: Data = global_data
        self.global_nxg: nx.Graph = to_networkx(global_data)
        self.subgraph_data_list: List[Data] = subgraph_data_list
        self.name: str = name
        self.path: Path = Path(path)

        self.edge_aggr = edge_aggr or torch.min

        self.undirected = undirected or is_undirected(global_data.edge_index)
        self.node_spl_cutoff = node_spl_cutoff

        self._node_spl_mat = None
        self._node_task_data_precursor = None
        self._node_task_data = None
        assert self.undirected, "Now only support undirected graphs"

    @property
    def S(self):
        return len(self.subgraph_data_list)

    def single_source_shortest_path_length_for_global_data(self, n):
        spl_dict = nx.single_source_shortest_path_length(
            self.global_nxg, n, cutoff=self.node_spl_cutoff)
        spl_list = [val for node, val in sorted(spl_dict.items(), key=lambda t: t[0])]
        return spl_list

    def all_pairs_shortest_path_length_for_global_data(self, processes=None):
        processes = processes or os.cpu_count()
        with mp.Pool(processes=processes) as pool:
            shortest_paths = pool.map(self.single_source_shortest_path_length_for_global_data,
                                      self.global_nxg.nodes)
        return torch.tensor(shortest_paths, dtype=torch.long)

    def node_spl_mat(self, save=True, **kwargs):
        path = self.path / f"{self.name}_spl_mat.pth"
        try:
            self._node_spl_mat = torch.load(path)
            cprint(f"Load: tensor of {self._node_spl_mat.size()} at {path}", "green")
            return self._node_spl_mat
        except FileNotFoundError:
            pass
        self._node_spl_mat = self.all_pairs_shortest_path_length_for_global_data(**kwargs)
        if save:
            torch.save(self._node_spl_mat, path)
            cprint(f"Save: tensor of {self._node_spl_mat.size()} at {path}", "blue")
        return self._node_spl_mat

    def node_task_data_precursor(self, save=True):
        path = self.path / f"{self.name}_node_task_data_precursor.pth"
        try:
            self._node_task_data_precursor = torch.load(path)
            cprint(f"Load: {self._node_task_data_precursor} at {path}", "green")
            return self._node_task_data_precursor
        except FileNotFoundError:
            pass

        # Node aggregation: x, y, batch, ...
        node_task_data_precursor = Batch.from_data_list(self.subgraph_data_list)
        del node_task_data_precursor.edge_index
        del node_task_data_precursor.ptr
        self._node_task_data_precursor = node_task_data_precursor

        # Edge aggregation
        # sub_spl_ij = min { d_uv | u \in S_i, v in S_j }
        node_spl_mat = self.node_spl_mat(save)
        sub_spl_mat = torch.full((self.S, self.S), fill_value=-1)
        for i, sub_data_i in enumerate(self.subgraph_data_list):
            for j, sub_data_j in enumerate(self.subgraph_data_list):
                if self.undirected and i <= j:
                    x_i = sub_data_i.x.squeeze()
                    x_j = sub_data_j.x.squeeze()
                    sub_spl = self.edge_aggr(node_spl_mat[x_i, :][:, x_j])
                    sub_spl_mat[i, j] = sub_spl
                    sub_spl_mat[j, i] = sub_spl
        if self.undirected:
            sub_spl_mat = to_symmetric_matrix(sub_spl_mat)

        # edge = 1 / (spl + 1) where 0 <= spl, then 0 < edge <= 1
        self._node_task_data_precursor.edge_weight_matrix = 1 / (sub_spl_mat + 1)

        if save:
            torch.save(self._node_task_data_precursor, path)
            cprint(f"Save: {self._node_task_data_precursor} at {path}", "blue")

        return self._node_task_data_precursor

    def node_task_data(self, edge_thres: float, save=True):
        """
        :return: Data(x=[N, 1], edge_index=[2, E], edge_attr=[E], y=[C], batch=[N])
            - N is the number of subgraphs = batch.sum()
            - edge_attr >= edge_thres
        """
        path = self.path / f"{self.name}_node_task_data_e{edge_thres}.pth"
        try:
            self._node_task_data = torch.load(path)
            cprint(f"Load: {self._node_task_data} at {path}", "green")
            return self._node_task_data
        except FileNotFoundError:
            pass

        node_task_data_precursor = self.node_task_data_precursor(save)
        ew_mat = node_task_data_precursor.edge_weight_matrix
        ew_mat[ew_mat < edge_thres] = 0
        edge_index, edge_attr = dense_to_sparse(ew_mat)
        self._node_task_data = Data(
            edge_index=edge_index, edge_attr=edge_attr,
            **{k: getattr(node_task_data_precursor, k)
               for k in node_task_data_precursor.keys
               if k != "edge_weight_matrix"},
        )
        if save:
            torch.save(self._node_task_data, path)
            cprint(f"Save: {self._node_task_data} at {path}", "blue")

        return self._node_task_data


if __name__ == '__main__':

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
    )
    print(s2n.node_task_data(0.25))
