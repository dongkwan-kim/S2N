import os
import multiprocessing as mp
from pathlib import Path
from pprint import pprint
from typing import List, Callable, Union, Tuple

from termcolor import cprint
import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx, from_networkx, is_undirected, dense_to_sparse
from tqdm import tqdm

from utils import to_symmetric_matrix, try_getattr


class SubgraphToNode:
    _global_nxg = None
    _node_spl_mat = None
    _node_task_data_precursor = None
    _node_task_data_list: List[Data] = []

    def __init__(self,
                 global_data: Data,
                 subgraph_data_list: List[Data],
                 name: str, path: str,
                 splits: List[int],
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

        self.edge_aggr = self.parse_edge_aggr(edge_aggr)

        self.num_workers = num_workers
        self.undirected = undirected or is_undirected(global_data.edge_index)
        self.node_spl_cutoff = node_spl_cutoff

        assert self.undirected, "Now only support undirected graphs"
        assert len(self.splits) == 3
        self.path.mkdir(exist_ok=True)

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.node_task_name}', path='{self.path}')"

    def parse_edge_aggr(self, edge_aggr):
        if isinstance(edge_aggr, str):
            return eval(edge_aggr)
        else:
            return edge_aggr or torch.min

    @property
    def node_task_name(self):
        return f"{self.name}-EA-{self.edge_aggr.__name__}"

    @property
    def S(self):
        return len(self.subgraph_data_list)

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
        node_task_data_precursor = Batch.from_data_list(self.subgraph_data_list)
        del node_task_data_precursor.edge_index
        self._node_task_data_precursor = node_task_data_precursor

        # Edge aggregation
        # sub_spl_ij = min { d_uv | u \in S_i, v in S_j }
        node_spl_mat = self.node_spl_mat(save).float()
        sub_spl_mat = torch.full((self.S, self.S), fill_value=-1)
        for i, sub_data_i in enumerate(tqdm(self.subgraph_data_list, desc="sub_spl_mat")):
            for j, sub_data_j in enumerate(self.subgraph_data_list):
                if self.undirected and i <= j:
                    x_i = sub_data_i.x.squeeze(-1)
                    x_j = sub_data_j.x.squeeze(-1)
                    sub_spl = self.edge_aggr(node_spl_mat[x_i, :][:, x_j])
                    sub_spl_mat[i, j] = sub_spl
                    sub_spl_mat[j, i] = sub_spl

        # edge = 1 / (spl + 1) where 0 <= spl, then 0 < edge <= 1
        self._node_task_data_precursor.edge_weight_matrix = 1 / (sub_spl_mat + 1)

        if save:
            torch.save(self._node_task_data_precursor, path)
            cprint(f"Save: {self._node_task_data_precursor} at {path}", "blue")

        return self._node_task_data_precursor

    def node_task_data_splits(self,
                              edge_thres: Union[float, Callable, List[float]],
                              save=True) -> Tuple[Data, Data, Data]:
        """
        :return: Data(x=[N, 1], edge_index=[2, E], edge_attr=[E], y=[C], batch=[N])
            - N is the number of subgraphs = batch.sum()
            - edge_attr >= edge_thres
        """
        str_et = edge_thres.__name__ if isinstance(edge_thres, Callable) else edge_thres
        path = self.path / f"{self.node_task_name}_node_task_data_e{str_et}.pth"
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
                print(
                    "Performing node_task_data_splits \n"
                    f"mean +- std = {torch.mean(ew_mat)} +- {torch.std(ew_mat)} \n"
                    f"min / median / max = {torch.min(ew_mat)} / {torch.median(ew_mat)} / {torch.max(ew_mat)} \n"
                )

            ew_mat_s_by_s = ew_mat.clone()[:s, :s]
            et = et(ew_mat_s_by_s) if isinstance(et, Callable) else et
            ew_mat_s_by_s[ew_mat_s_by_s < et] = 0
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


if __name__ == '__main__':

    from data_sub import HPOMetab, HPONeuro, PPIBP, EMUser, Density, CC, Coreness, CutRatio

    MODE = "Density"
    # PPIBP, HPOMetab, HPONeuro, EMUser
    # Density, CC, Coreness, CutRatio

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
            edge_aggr=torch.min,
        )
        print(s2n)
        ntds = s2n.node_task_data_splits(
            # 0.5, 1.0
            edge_thres=0.5,
            save=True,
        )
        for _d in ntds:
            print(_d, _d.edge_index.size(1) / (_d.num_nodes ** 2))

    elif MODE == "TEST":
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
        )
        ntds = s2n.node_task_data_splits(0.25, save=True)
        pprint(ntds)
        print(ntds[-2].eval_mask)
        print(ntds[-1].eval_mask)

