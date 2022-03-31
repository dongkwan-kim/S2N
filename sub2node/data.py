from argparse import Namespace
from typing import Type, Any, Optional, Union, Dict, Tuple, List, Callable
from pprint import pprint

import torch
from termcolor import cprint

import torch_geometric.transforms as T
from torch import Tensor
from torch_geometric.data import Data
from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor

from data_sub import HPONeuro, PPIBP, HPOMetab, EMUser, SubgraphDataset
from data_sub import Density, CC, Coreness, CutRatio
from data_utils import AddSelfLoopsV2
from sub2node import SubgraphToNode
from utils import get_log_func, EternalIter


class SubgraphDataModule(LightningDataModule):

    @property
    def h(self):
        return self.hparams

    def __init__(self,
                 dataset_name: str,
                 dataset_path: str,
                 embedding_type: str,
                 use_s2n: bool,
                 edge_thres: Union[float, Callable, List[float]],
                 edge_normalize: Union[str, Callable, None],
                 s2n_edge_aggr: Union[Callable[[Tensor], Tensor], str] = None,
                 batch_size: int = None,
                 eval_batch_size=None,
                 use_sparse_tensor=False,
                 pre_add_self_loops=False,
                 num_workers=0,
                 verbose=2,
                 prepare_data=False,
                 log_func=None,
                 *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["prepare_data", "logger"])
        self.dataset: Optional[SubgraphDataset] = None
        self.train_data, self.val_data, self.test_data = None, None, None
        self.split_idx: Union[Dict, None] = None

        if prepare_data:
            self.prepare_data()
        self.setup()

        self.log_func = log_func or get_log_func(cprint, color="green")
        if self.h.verbose >= 1:
            self.log_func(f"{self.__class__.__name__}/{self.h.dataset_name}: prepared and set up!")

    @property
    def num_nodes_global(self):
        return self.dataset.num_nodes_global

    @property
    def num_channels_global(self):
        return self.dataset.global_data.x.size(1)

    @property
    def num_classes(self):
        return self.dataset.num_classes

    @property
    def num_nodes(self) -> int:
        return self.dataset.num_nodes

    @property
    def embedding(self):
        return self.dataset.global_data.x

    @property
    def dataset_class(self):
        assert self.h.dataset_name in ["HPOMetab", "PPIBP", "HPONeuro", "EMUser",
                                       "Density", "CC", "Coreness", "CutRatio"]
        return eval(self.h.dataset_name)

    def prepare_data(self) -> None:
        self.dataset_class(root=self.h.dataset_path, name=self.h.dataset_name,
                           embedding_type=self.h.embedding_type)

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset: SubgraphDataset = self.dataset_class(
            root=self.h.dataset_path, name=self.h.dataset_name,
            embedding_type=self.h.embedding_type,
        )
        if self.h.use_s2n:
            s2n = SubgraphToNode(
                global_data=self.dataset.global_data,
                subgraph_data_list=self.dataset.get_data_list_with_split_attr(),
                name=self.h.dataset_name,
                path=f"{self.h.dataset_path}/{self.h.dataset_name.upper()}/sub2node/",
                splits=self.dataset.splits,
                edge_aggr=self.h.s2n_edge_aggr,
                undirected=True,
            )
            data_list = s2n.node_task_data_splits(
                edge_normalize=self.h.edge_normalize,
                edge_thres=self.h.edge_thres,
            )
            transform_list = []
            if self.h.pre_add_self_loops:
                transform_list.append(AddSelfLoopsV2("edge_attr"))
            if self.h.use_sparse_tensor:
                transform_list.append(T.ToSparseTensor("edge_attr"))
            transform = T.Compose(transform_list) if len(transform_list) > 0 else None
            if transform is not None:
                data_list = [transform(d) for d in data_list]
            self.train_data, self.val_data, self.test_data = data_list
        else:
            self.train_data, self.val_data, self.test_data = self.dataset.get_train_val_test_with_relabeling()

    def train_dataloader(self):
        if self.h.use_s2n:
            return EternalIter([self.train_data])
        else:
            return DataLoader(
                self.train_data, batch_size=self.h.batch_size,
                shuffle=True, num_workers=self.h.num_workers,
            )

    def _eval_loader(self, eval_data: Union[Data, List[Data]], stage=None):
        if self.h.use_s2n:
            return EternalIter([eval_data])
        else:
            return DataLoader(
                eval_data, batch_size=(self.h.eval_batch_size or self.h.batch_size),
                shuffle=False, num_workers=self.h.num_workers,
            )

    def val_dataloader(self):
        return self._eval_loader(self.val_data, stage="valid")

    def test_dataloader(self):
        return self._eval_loader(self.test_data, stage="test")

    def __repr__(self):
        return "{}(dataset={})".format(self.__class__.__name__, self.h.dataset_name)


def _print_data(data):
    pprint(_b)
    if data.edge_index is not None:
        print("\t- edge (Tensor)", f"{data.edge_index.min()} -- {data.edge_index.max()}")
    else:
        data.adj_t: SparseTensor
        row, col, _ = data.adj_t.coo()
        e = torch.cat([row, col])
        print("\t- edge (SparseTensor)", f"{e.min()} -- {e.max()}")
        print("\t- adj_t", data.adj_t)


if __name__ == '__main__':

    NAME = "Density"
    # PPIBP, HPOMetab, HPONeuro, EMUser
    # Density, CC, Coreness, CutRatio

    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    E_TYPE = "graphsaint_gcn"  # gin, graphsaint_gcn

    USE_S2N = True
    USE_SPARSE_TENSOR = True

    _sdm = SubgraphDataModule(
        dataset_name=NAME,
        dataset_path=PATH,
        embedding_type=E_TYPE,
        use_s2n=USE_S2N,
        edge_thres=0.75,
        edge_normalize="sig_standardize_incl_diag",
        batch_size=32,
        eval_batch_size=5,
        use_sparse_tensor=USE_SPARSE_TENSOR,
        pre_add_self_loops=True,
    )
    print(_sdm)
    cprint("Train ----", "green")
    for _i, _b in enumerate(_sdm.train_dataloader()):
        _print_data(_b)
        if _i == 2:
            break
    cprint("Valid ----", "green")
    for _i, _b in enumerate(_sdm.val_dataloader()):
        _print_data(_b)
        if _i == 2:
            break
    cprint("Test ----", "green")
    for _i, _b in enumerate(_sdm.test_dataloader()):
        _print_data(_b)
        if _i == 2:
            break
