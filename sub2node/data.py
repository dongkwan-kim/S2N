from argparse import Namespace
from typing import Type, Any, Optional, Union, Dict, Tuple, List, Callable
from pprint import pprint

import torch
from termcolor import cprint

import torch_geometric.transforms as T
from torch_geometric.data import Data
from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader

from data_sub import HPONeuro, PPIBP, HPOMetab, EMUser, SubgraphDataset
from sub2node import SubgraphToNode
from utils import get_log_func, EternalIter


class SubgraphDataModule(LightningDataModule):

    @property
    def h(self):
        return self.hparams

    def __init__(self,
                 dataset_name: str,
                 dataset_path: str,
                 use_s2n: bool,
                 edge_thres: Union[float, Callable, List[float]],
                 batch_size: int,
                 eval_batch_size=None,
                 use_sparse_tensor=False,
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
        self.model_kwargs = dict()

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
        return {"HPOMetab": HPOMetab,
                "PPIBP": PPIBP,
                "HPONeuro": HPONeuro,
                "EMUser": EMUser,
                }[self.h.dataset_name]

    def prepare_data(self) -> None:
        self.dataset_class(root=self.h.dataset_path, name=self.h.dataset_name)

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset: SubgraphDataset = self.dataset_class(root=self.h.dataset_path, name=self.h.dataset_name)
        if self.h.use_s2n:
            s2n = SubgraphToNode(
                global_data=self.dataset.global_data,
                subgraph_data_list=self.dataset.get_data_list_with_split_attr(),
                name=self.h.dataset_name,
                path=f"{self.h.dataset_path}/{self.h.dataset_name.upper()}/sub2node/",
                undirected=True,
                splits=self.dataset.splits,
            )
            data_list = s2n.node_task_data_splits(edge_thres=self.h.edge_thres)
            transform = T.ToSparseTensor("edge_attr") if self.h.use_sparse_tensor else None
            if transform is not None:
                data_list = [transform(d) for d in data_list]
            self.train_data, self.val_data, self.test_data = data_list
        else:
            self.train_data, self.val_data, self.test_data = self.dataset.get_train_val_test()

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


if __name__ == '__main__':

    NAME = "PPIBP"  # "HPOMetab", "PPIBP", "HPONeuro", "EMUser"
    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    USE_S2N = True

    _sdm = SubgraphDataModule(
        dataset_name=NAME,
        dataset_path=PATH,
        use_s2n=USE_S2N,
        edge_thres=0.5,
        batch_size=32,
        eval_batch_size=5,
        use_sparse_tensor=True,
    )
    print(_sdm)
    cprint("Train ----", "green")
    for _i, _b in enumerate(_sdm.train_dataloader()):
        pprint(_b)
        if _i == 2:
            break
    cprint("Valid ----", "green")
    for _i, _b in enumerate(_sdm.val_dataloader()):
        pprint(_b)
        if _i == 2:
            break
    cprint("Test ----", "green")
    for _i, _b in enumerate(_sdm.test_dataloader()):
        pprint(_b)
        if _i == 2:
            break
