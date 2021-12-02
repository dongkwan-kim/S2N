from collections import OrderedDict, Counter
from itertools import chain
from pprint import pprint
from typing import List, Dict, Tuple, Union
import os.path as osp

import torch
from termcolor import cprint
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import numpy_indexed as npi
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split


class DatasetBase(InMemoryDataset):
    """Dataset base class"""

    def __init__(self, root, name,
                 val_ratio=0.15, test_ratio=0.15, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):

        self.name = name
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.debug = debug
        self.seed = seed

        self.num_train = -1
        self.num_val = -1
        self.global_data = None
        self._num_nodes_global = None
        super(DatasetBase, self).__init__(root, transform, pre_transform)

        self.load()
        self.cprint()

    def load(self):
        raise NotImplementedError

    def cprint(self):
        cprint(
            "Initialized: {} (debug={}) \n"
            "/ num_nodes: {}, num_edges: {} \n"
            "/ num_train: {}, num_val: {}, num_test: {} \n".format(
                self.__class__.__name__, self.debug,
                self.global_data.edge_index.max() + 1, self.global_data.edge_index.size(),
                self.num_train, self.num_val, len(self) - self.num_train - self.num_val)
            + "Loaded from: {} \n".format(self.processed_dir),
            "blue",
        )

    @property
    def num_nodes_global(self):
        if self._num_nodes_global is None:
            self._num_nodes_global = self.global_data.edge_index.max().item() + 1
        return self._num_nodes_global

    def _get_important_elements(self):
        ie = {
            "name": self.name,
            "seed": self.seed,
            "debug": self.debug,
        }
        if self.pre_transform is not None:
            # Remove all blanks.
            ie["pre_transform"] = "".join(str(self.pre_transform).split())
        return ie

    def _logging_args(self):
        with open(osp.join(self.processed_dir, "args.txt"), "w") as f:
            f.writelines(["{}: {}\n".format(k, v) for k, v in self._get_important_elements().items()])
        cprint("Args logged: ")
        pprint(self._get_important_elements())

    def _get_stats(self, stat_names=None, stat_functions=None):
        if stat_names is None:
            stat_names = ['x', 'edge_index']
        if stat_functions is None:
            stat_functions = [
                torch.mean, torch.std,
                torch.min, torch.max, torch.median,
            ]
        stat_dict = OrderedDict()
        for name in stat_names:
            if name in self.slices:
                s_vec = (self.slices[name][1:] - self.slices[name][:-1])
                s_vec = s_vec.float()
                for func in stat_functions:
                    printing_name = "{}/#{}".format(func.__name__, name)
                    printing_value = func(s_vec)
                    stat_dict[printing_name] = printing_value
        s = {
            "num_graphs": len(self),
            "num_train": self.num_train, "num_val": self.num_val,
            "num_test": len(self) - self.num_train - self.num_val,
            "num_classes": self.num_classes,
            "num_global_nodes": self.global_data.edge_index.max() + 1,
            "num_global_edges": self.global_data.edge_index.size(1),
            **stat_dict,
        }
        return s

    @property
    def raw_dir(self):
        return osp.join(self.root, self.__class__.__name__.upper(), 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.__class__.__name__.upper(),
                        'processed_{}'.format("_".join([str(e) for e in self._get_important_elements().values()])))

    @property
    def raw_file_names(self):
        raise NotImplementedError

    @property
    def processed_file_names(self):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    def train_val_test_split(self, data_list):
        num_total = len(data_list)
        num_val = int(num_total * self.val_ratio)
        num_test = int(num_total * self.test_ratio)
        y = np.asarray([int(d.y) for d in data_list])
        data_train_and_val, data_test = train_test_split(
            data_list,
            test_size=num_test, random_state=self.seed, stratify=y,
        )
        y_train_and_val = np.asarray([int(d.y) for d in data_train_and_val])
        data_train, data_val = train_test_split(
            data_train_and_val,
            test_size=num_val, random_state=self.seed, stratify=y_train_and_val,
        )
        return data_train, data_val, data_test

    def tolist(self):
        return list(self)

    def get_train_val_test(self) -> Tuple[List[Data], List[Data], List[Data]]:
        data_list = self.tolist()
        num_train_and_val = self.num_train + self.num_val
        data_train = data_list[:self.num_train]
        data_val = data_list[self.num_train:num_train_and_val]
        data_test = data_list[num_train_and_val:]
        return data_train, data_val, data_test

    def get_data_list_with_split_attr(self) -> List[Data]:
        data_train, data_val, data_test = self.get_train_val_test()
        for i, d_set in enumerate([data_train, data_val, data_test]):
            for d in d_set:
                setattr(d, "split", torch.Tensor([i]).long())
        return data_train + data_val + data_test

    def print_summary(self):

        def out(v):
            return str(float(v)) if isinstance(v, torch.Tensor) else str(v)

        print("---------------------------------------------------------------")
        for k, v in chain(self._get_important_elements().items(),
                          self._get_stats().items()):
            print("{:>20}{:>43}".format(k, out(v)))
        print("---------------------------------------------------------------")

    def __repr__(self):
        return '{}(\n{}\n)'.format(
            self.__class__.__name__,
            "\n".join("\t{}={},".format(k, v) for k, v in self._get_important_elements().items()),
        )
