from collections import OrderedDict, Counter
from itertools import chain
from pprint import pprint
from typing import List, Dict, Tuple, Union, Optional
import os.path as osp
import random

import torch
from termcolor import cprint
from torch_geometric.data import InMemoryDataset, Data, Batch
import numpy as np
import numpy_indexed as npi
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

from data_utils import RelabelNodes


class DatasetBase(InMemoryDataset):
    """Dataset base class"""

    def __init__(self, root, name,
                 val_ratio=0.15, test_ratio=0.15, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0,
                 transform=None, pre_transform=None, **kwargs):

        self.name = name
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.debug = debug
        self.seed = seed

        # These will be True for data-scarce experiments
        self.num_training_tails_to_tile_per_class = num_training_tails_to_tile_per_class
        self.num_train_original = -1
        self.num_val_original = -1

        self.num_start = 0
        self.num_train = -1
        self.num_val = -1
        self.global_data: Optional[Data] = None
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
            "green",
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
        # Data example: Data(x=[10, 1], edge_index=[2, 18], y=[1, 4])
        data_list = self.tolist()
        num_until_train = self.num_start + self.num_train
        num_until_val = num_until_train + self.num_val

        # If there should be the same number of samples per class in the tail (after self.num_start),
        ntt2tpc = self.num_training_tails_to_tile_per_class
        if ntt2tpc is not None and ntt2tpc > 0:
            data_train_original = data_list[:self.num_train_original]
            random.Random(42).shuffle(data_train_original)
            cprint("Ran random.Random(42).shuffle(data_train_original)", "blue")

            data_tails = [data_train_original.pop()]
            assert data_tails[-1].y.dim() == 1, "only for single-label"
            while len(data_tails) < self.num_classes * self.num_training_tails_to_tile_per_class:
                d = data_train_original.pop()
                if (data_tails[-1].y.item() + 1) % self.num_classes == d.y.item():
                    # append a sample of the next class
                    data_tails.append(d)
                else:
                    data_train_original.insert(0, d)
            data_train = (data_train_original + data_tails)[self.num_start:num_until_train]
        else:
            data_train = data_list[self.num_start:num_until_train]
        data_val = data_list[num_until_train:num_until_val]
        data_test = data_list[num_until_val:]
        return data_train, data_val, data_test

    def get_data_list_with_split_attr(self) -> List[Data]:
        # For S2N graphs
        data_train, data_val, data_test = self.get_train_val_test()
        for i, d_set in enumerate([data_train, data_val, data_test]):
            for d in d_set:
                setattr(d, "split", torch.Tensor([i]).long())
        return data_train + data_val + data_test

    def get_train_val_test_with_individual_relabeling(self) -> Tuple[List[Data], List[Data], List[Data]]:
        # For individual, and separated subgraphs
        rn_transform = RelabelNodes()
        data_train, data_val, data_test = self.get_train_val_test()
        data_train = [rn_transform(d) for d in data_train]
        data_val = [rn_transform(d) for d in data_val]
        data_test = [rn_transform(d) for d in data_test]
        return data_train, data_val, data_test

    def get_train_val_test_connected_on_global_data(self) -> Tuple[Data, Data, Data]:
        # For individual, but all connected in the global data
        data_train, data_val, data_test = self.get_train_val_test()
        global_edge_index = self.global_data.edge_index

        rets: List[Data] = []
        for data_list in [data_train, data_val, data_test]:
            aggr_data = Batch.from_data_list(data_list)

            aggr_data.x_to_xs = aggr_data.x.squeeze()
            aggr_data.x = torch.arange(self.global_data.num_nodes, dtype=torch.long).view(-1, 1)
            aggr_data.edge_index = global_edge_index

            rets.append(aggr_data)

        return rets[0], rets[1], rets[2]

    def print_summary(self, **kwargs):

        def out(v):
            return str(float(v)) if isinstance(v, torch.Tensor) else str(v)

        print("-" * 69)
        for k, v in chain(self._get_important_elements().items(),
                          self._get_stats().items(),
                          kwargs.items()):
            print("{:>25}{:>43}".format(k, out(v)))
        print("-" * 69)

    def __repr__(self):
        return '{}(\n{}\n)'.format(
            self.__class__.__name__,
            "\n".join("\t{}={},".format(k, v) for k, v in self._get_important_elements().items()),
        )
