import inspect
from argparse import Namespace
from typing import Type, Any, Optional, Union, Dict, Tuple, List, Callable
from pprint import pprint

import os
import torch
from termcolor import cprint

import torch_geometric.transforms as T
from torch import Tensor
from torch_geometric.data import Data, Batch
from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor

from data_sub import HPONeuro, PPIBP, HPOMetab, EMUser, SubgraphDataset
from data_sub import WLHistSubgraphBA, WLHistSubgraphER
from data_sub import Density, CC, Coreness, CutRatio
from data_utils import AddSelfLoopsV2, RemoveAttrs
from dataset_wl import SliceYByIndex
from sub2node import SubgraphToNode
from utils import get_log_func, EternalIter, merge_dict_by_keys


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
                 s2n_target_matrix: str,
                 s2n_edge_aggr: Union[Callable[[Tensor], Tensor], str] = None,
                 s2n_is_weighted: bool = True,
                 s2n_transform=None,
                 subgraph_batching: str = None,
                 batch_size: int = None,
                 eval_batch_size=None,
                 use_sparse_tensor=False,
                 pre_add_self_loops=False,
                 num_channels_global: int = None,
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
        try:
            return self.dataset.global_data.x.size(1)
        except AttributeError:  # If x is not given.
            return self.h.num_channels_global

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
                                       "Density", "CC", "Coreness", "CutRatio",
                                       "WLHistSubgraphER", "WLHistSubgraphBA"]
        return eval(self.h.dataset_name)

    def load_dataset(self):
        init_kwargs = merge_dict_by_keys(
            {}, dict(self.h.items()),
            inspect.getfullargspec(self.dataset_class.__init__).args
        )
        return self.dataset_class(root=self.h.dataset_path, name=self.h.dataset_name,
                                  **init_kwargs)

    def prepare_data(self) -> None:
        self.load_dataset()

    @property
    def s2n_path(self) -> str:
        if self.h.dataset_name in ["WLHistSubgraphER", "WLHistSubgraphBA"]:
            return os.path.join(self.dataset.key_dir, "sub2node")
        else:  # backward compatibility
            return f"{self.h.dataset_path}/{self.h.dataset_name.upper()}/sub2node/"

    def setup(self, stage: Optional[str] = None) -> None:

        self.dataset: SubgraphDataset = self.load_dataset()

        if self.h.use_s2n:
            s2n = SubgraphToNode(
                global_data=self.dataset.global_data,
                subgraph_data_list=self.dataset.get_data_list_with_split_attr(),
                name=self.h.dataset_name,
                path=self.s2n_path,
                splits=self.dataset.splits,
                target_matrix=self.h.s2n_target_matrix,
                edge_aggr=self.h.s2n_edge_aggr,
                undirected=True,
            )
            data_list = s2n.node_task_data_splits(
                edge_normalize=self.h.edge_normalize,
                edge_normalize_args=[getattr(self.h, f"edge_normalize_arg_{i}") for i in range(1, 3)
                                     if getattr(self.h, f"edge_normalize_arg_{i}", None) is not None],
                edge_thres=self.h.edge_thres,
                use_consistent_processing=self.h.use_consistent_processing,
            )
            transform_list = []
            if not self.h.s2n_is_weighted:
                transform_list.append(RemoveAttrs(["edge_attr"]))
            if self.h.pre_add_self_loops:
                transform_list.append(AddSelfLoopsV2("edge_attr"))
            if self.h.use_sparse_tensor:
                transform_list.append(T.ToSparseTensor("edge_attr"))
            if self.h.s2n_transform is not None:
                s2n_transform = self.h.s2n_transform
                if isinstance(self.h.s2n_transform, str):
                    s2n_transform = eval(s2n_transform)(*self.h.s2n_transform_args)
                transform_list.append(s2n_transform)
            transform = T.Compose(transform_list) if len(transform_list) > 0 else None
            if transform is not None:
                data_list = [transform(d) for d in data_list]
            self.train_data, self.val_data, self.test_data = data_list
        else:
            if self.h.subgraph_batching == "separated":
                self.train_data, self.val_data, self.test_data \
                    = self.dataset.get_train_val_test_with_individual_relabeling()
            elif self.h.subgraph_batching == "connected":
                self.train_data, self.val_data, self.test_data \
                    = self.dataset.get_train_val_test_connected_on_global_data()
            else:
                raise ValueError(f"Wrong subgraph_batching: {self.h.subgraph_batching}")

    def train_dataloader(self):
        if isinstance(self.train_data, (Data, Batch)):  # s2n, connected
            return EternalIter([self.train_data])
        else:
            assert isinstance(self.train_data, list)
            return DataLoader(
                self.train_data, batch_size=(self.h.batch_size or len(self.train_data)),
                shuffle=True, num_workers=self.h.num_workers,
            )

    def _eval_loader(self, eval_data: Union[Data, List[Data]], stage=None):
        if isinstance(eval_data, (Data, Batch)):  # s2n, connected
            return EternalIter([eval_data])
        else:
            assert isinstance(eval_data, list)
            return DataLoader(
                eval_data, batch_size=(self.h.eval_batch_size or self.h.batch_size or len(eval_data)),
                shuffle=False, num_workers=self.h.num_workers,
            )

    def val_dataloader(self):
        return self._eval_loader(self.val_data, stage="valid")

    def test_dataloader(self):
        return self._eval_loader(self.test_data, stage="test")

    def __repr__(self):
        return "{}(dataset={})".format(self.__class__.__name__, self.h.dataset_name)


def _print_data(data):
    pprint(data)
    if data.x is not None:
        print("\t- x (Tensor)", f"{data.x.min()} -- {data.x.max()}")
    if hasattr(data, "sub_x") and data.sub_x is not None:
        print("\t- sub_x (Tensor)", f"{data.sub_x.min()} -- {data.sub_x.max()}")
    if data.edge_index is not None:
        print("\t- edge (Tensor)", f"{data.edge_index.min()} -- {data.edge_index.max()}")
    else:
        data.adj_t: SparseTensor
        row, col, _ = data.adj_t.coo()
        e = torch.cat([row, col])
        print("\t- edge (SparseTensor)", f"{e.min()} -- {e.max()}")
        print("\t- adj_t", data.adj_t)


if __name__ == '__main__':

    NAME = "WLHistSubgraphBA"
    # WLHistSubgraphBA, WLHistSubgraphER
    # PPIBP, HPOMetab, HPONeuro, EMUser
    # Density, CC, Coreness, CutRatio

    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    E_TYPE = "graphsaint_gcn"  # gin, graphsaint_gcn

    USE_S2N = False
    USE_SPARSE_TENSOR = False
    SUBGRAPH_BATCHING = None if USE_S2N else "connected"  # separated, connected

    if not NAME.startswith("WL"):
        _sdm = SubgraphDataModule(
            dataset_name=NAME,
            dataset_path=PATH,
            embedding_type=E_TYPE,
            use_s2n=USE_S2N,
            edge_thres=0.0,
            use_consistent_processing=True,
            edge_normalize="standardize_then_trunc_thres_max_linear",
            edge_normalize_arg_1=0.0,
            edge_normalize_arg_2=2.0,
            s2n_target_matrix="adjacent_no_self_loops",
            s2n_is_weighted=False,
            subgraph_batching=SUBGRAPH_BATCHING,
            batch_size=32,
            eval_batch_size=5,
            use_sparse_tensor=USE_SPARSE_TENSOR,
            pre_add_self_loops=False,
        )
    else:
        E_TYPE = "no_embedding"  # override
        if NAME == "WLHistSubgraphBA":
            _more_kwargs = {"ba_n": 10000, "ba_m": 10, "ba_seed": 42}
        elif NAME == "WLHistSubgraphER":
            _more_kwargs = {"er_n": 10000, "er_p": 0.002, "er_seed": 42}
        else:
            _more_kwargs = {}

        _sdm = SubgraphDataModule(
            dataset_name=NAME,
            dataset_path=PATH,
            embedding_type=E_TYPE,
            use_s2n=USE_S2N,
            edge_thres=0.0,
            use_consistent_processing=True,
            edge_normalize="standardize_then_trunc_thres_max_linear",
            edge_normalize_arg_1=0.0,
            edge_normalize_arg_2=2.0,
            s2n_target_matrix="adjacent_no_self_loops",
            s2n_is_weighted=False,
            subgraph_batching=SUBGRAPH_BATCHING,
            batch_size=32,
            eval_batch_size=5,
            use_sparse_tensor=USE_SPARSE_TENSOR,
            pre_add_self_loops=False,
            **{
                "num_subgraphs": 1500,
                "subgraph_size": 10,
                "wl_hop_to_use": None,
                "wl_max_hop": 4,
                "wl_x_type_for_hists": "cluster",  # color, cluster
                "wl_num_color_clusters": None,
                "wl_num_hist_clusters": 2,
                **_more_kwargs,
            }
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
