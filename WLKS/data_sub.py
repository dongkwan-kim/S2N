import os
import re
from typing import List, Union

import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from termcolor import cprint
from torch_geometric.data import Data
from torch_geometric.transforms import LocalDegreeProfile
from torch_geometric.utils import subgraph, sort_edge_index, to_undirected
from tqdm import tqdm

from data_base import DatasetBase
from data_transform import AddRandomWalkPE, AddLaplacianEigenvectorPE
from utils import from_networkx_customized_ordering, to_directed
from utils_fscache import fscaches


def read_subgnn_data(edge_list_path, subgraph_path,
                     embedding_path=None, save_directed_edges=False, debug=False):
    """
    Read in the subgraphs & their associated labels
    Reference: https://github.com/mims-harvard/SubGNN/blob/main/SubGNN/SubGNN.py#L519
    """
    # read list of node ids for each subgraph & their labels
    train_nodes, _train_ys, val_nodes, _val_ys, test_nodes, _test_ys, y_dtype = read_subgraphs(subgraph_path)
    cprint("Loaded subgraphs at {}".format(subgraph_path), "green")

    # check if the dataset is multilabel (e.g. HPO-NEURO)
    if type(_train_ys) == list:
        all_labels = _train_ys + _val_ys + _test_ys
        mlb = MultiLabelBinarizer()
        mlb.fit(all_labels)
        train_sub_ys = torch.Tensor(mlb.transform(_train_ys))
        val_sub_ys = torch.Tensor(mlb.transform(_val_ys))
        test_sub_ys = torch.Tensor(mlb.transform(_test_ys))
    else:
        train_sub_ys, val_sub_ys, test_sub_ys = _train_ys, _val_ys, _test_ys

    # Initialize pretrained node embeddings
    try:
        xs = torch.load(embedding_path).cpu()  # feature matrix should be initialized to the node embeddings
        # xs_with_zp = torch.cat([torch.zeros(1, xs.shape[1]), xs], 0)  # there's a zeros in the first index for padding
        cprint("Loaded embeddings at {}".format(embedding_path), "green")
    except (FileNotFoundError, AttributeError):
        xs = None
        cprint("No embeddings at {}".format(embedding_path), "red")

    # read networkx graph from edge list
    global_nxg: nx.Graph = nx.read_edgelist(edge_list_path)
    cprint("Loaded global_graph at {}".format(edge_list_path), "green")
    global_data = from_networkx_customized_ordering(global_nxg, ordering="keep")
    cprint("Converted global_graph to PyG format", "green")
    global_data.edge_index = sort_edge_index(global_data.edge_index)
    global_data.x = xs

    train_data_list = get_data_list_from_subgraphs(
        global_data.edge_index, train_nodes, train_sub_ys,
        save_directed_edges=save_directed_edges, y_dtype=y_dtype, debug=debug)
    cprint("Converted train_subgraph to PyG format", "green")
    val_data_list = get_data_list_from_subgraphs(
        global_data.edge_index, val_nodes, val_sub_ys,
        save_directed_edges=save_directed_edges, y_dtype=y_dtype, debug=debug)
    cprint("Converted val_subgraph to PyG format", "green")
    test_data_list = get_data_list_from_subgraphs(
        global_data.edge_index, test_nodes, test_sub_ys,
        save_directed_edges=save_directed_edges, y_dtype=y_dtype, debug=debug)
    cprint("Converted test_subgraph to PyG format", "green")
    return global_data, train_data_list, val_data_list, test_data_list


def get_data_list_from_subgraphs(global_edge_index, sub_nodes: List[List[int]], sub_ys,
                                 save_directed_edges, y_dtype="long", debug=False):
    data_list = []
    for idx, (x_index, y) in enumerate(zip(sub_nodes, tqdm(sub_ys))):
        x_index = torch.Tensor(x_index).long().view(-1, 1)
        if len(y.size()) == 0:  # single-label
            y = torch.Tensor([y])
        else:  # multi-label, or many-label
            y = y.view(1, -1)
        y = y.long() if y_dtype == "long" else y.float()
        edge_index, _ = subgraph(x_index, global_edge_index, relabel_nodes=False)
        if edge_index.size(1) <= 0:
            cprint("No edge graph: size of X is {}".format(x_index.size()), "red")
        if x_index.size(0) <= 1:
            cprint("Single node graph: size of E is {}".format(edge_index.size()), "yellow")
        if save_directed_edges and edge_index.size(1) >= 2:
            edge_index = to_directed(edge_index)
        data = Data(x=x_index, edge_index=edge_index, y=y)
        data_list.append(data)

        if debug and idx >= 5:
            break

    return data_list


def read_subgraphs(subgraph_path):
    """
    Read subgraphs from file
    Args
       - sub_f (str): filename where subgraphs are stored
    Return for each train, val, test split:
       - sub_G (list): list of nodes belonging to each subgraph
       - sub_G_label (list): labels for each subgraph
    """

    # Enumerate/track labels
    label_idx = 0
    labels = {}

    # Train/Val/Test subgraphs
    train_sub_g, val_sub_g, test_sub_g = [], [], []

    # Train/Val/Test subgraph labels
    train_sub_y, val_sub_y, test_sub_y = [], [], []

    # Train/Val/Test masks
    train_mask, val_mask, test_mask = [], [], []

    multilabel = False
    manylabel = False

    # Parse data
    with open(subgraph_path) as fin:
        subgraph_idx = 0
        for line in fin:
            nodes = [int(n) for n in line.split("\t")[0].split("-") if n != ""]
            if len(nodes) != 0:
                if len(nodes) == 1:
                    print("G with one node: ", nodes)

                label_cell = line.split("\t")[1]
                if "+" in label_cell:  # just many labels, not multi-labels
                    manylabel = True
                    l = label_cell.split("+")
                    for lab in l:  # use original 'integer' labels
                        labels[lab] = int(lab)
                else:
                    l = label_cell.split("-")
                    if len(l) > 1:
                        multilabel = True
                    for lab in l:
                        if lab not in labels.keys():
                            labels[lab] = label_idx
                            label_idx += 1

                if line.split("\t")[2].strip() == "train":
                    train_sub_g.append(nodes)
                    train_sub_y.append([labels[lab] for lab in l])
                    train_mask.append(subgraph_idx)
                elif line.split("\t")[2].strip() == "val":
                    val_sub_g.append(nodes)
                    val_sub_y.append([labels[lab] for lab in l])
                    val_mask.append(subgraph_idx)
                elif line.split("\t")[2].strip() == "test":
                    test_sub_g.append(nodes)
                    test_sub_y.append([labels[lab] for lab in l])
                    test_mask.append(subgraph_idx)
                subgraph_idx += 1

    if not multilabel:
        train_sub_y = torch.tensor(train_sub_y).long().squeeze()
        val_sub_y = torch.tensor(val_sub_y).long().squeeze()
        test_sub_y = torch.tensor(test_sub_y).long().squeeze()
    if manylabel:
        train_sub_y, val_sub_y, test_sub_y = train_sub_y.long(), val_sub_y.long(), test_sub_y.long()

    if len(val_mask) < len(test_mask):
        return train_sub_g, train_sub_y, test_sub_g, test_sub_y, val_sub_g, val_sub_y

    y_dtype = "float" if multilabel else "long"
    return train_sub_g, train_sub_y, val_sub_g, val_sub_y, test_sub_g, test_sub_y, y_dtype


def read_glass_syn_data(path):
    # copied from https://github.com/mims-harvard/SubGNN/blob/main/SubGNN/subgraph_utils.py
    obj: dict = np.load(path, allow_pickle=True).item()
    # dict of ['G', 'subG', 'subGLabel', 'mask']
    edge = torch.from_numpy(np.array([[i[0] for i in obj['G'].edges],
                                      [i[1] for i in obj['G'].edges]]))
    edge = sort_edge_index(to_undirected(edge))

    node = [int(n) for n in obj['G'].nodes]
    subG = obj["subG"]
    # subG_pad = pad_sequence([torch.tensor(i) for i in subG],
    #                         batch_first=True,
    #                         padding_value=-1)
    subGLabel = torch.tensor([ord(i) - ord('A') for i in obj["subGLabel"]])
    mask = torch.tensor(obj['mask'])

    train_nodes, val_nodes, test_nodes = [], [], []
    for m, sg in zip(obj["mask"], subG):
        if m == 0:
            train_nodes.append(sg)
        elif m == 1:
            val_nodes.append(sg)
        elif m == 2:
            test_nodes.append(sg)

    train_sub_ys = subGLabel[mask == 0]
    val_sub_ys = subGLabel[mask == 1]
    test_sub_ys = subGLabel[mask == 2]

    # Generate data classes
    global_data = Data(edge_index=edge, x=torch.ones((len(node), 64)).float() / 64, num_nodes=len(node))
    train_data_list = get_data_list_from_subgraphs(
        global_data.edge_index, train_nodes, train_sub_ys, False)
    cprint("Converted train_subgraph to PyG format", "green")
    val_data_list = get_data_list_from_subgraphs(
        global_data.edge_index, val_nodes, val_sub_ys, False)
    cprint("Converted val_subgraph to PyG format", "green")
    test_data_list = get_data_list_from_subgraphs(
        global_data.edge_index, test_nodes, test_sub_ys, False)
    cprint("Converted test_subgraph to PyG format", "green")

    return global_data, train_data_list, val_data_list, test_data_list


class SubgraphDataset(DatasetBase):
    url = "https://github.com/mims-harvard/SubGNN"

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False,
                 transform=None, pre_transform=None, **kwargs):
        # assert embedding_type in ["gin", "graphsaint_gcn", "no_embedding", "glass", "one"]
        self.embedding_type = embedding_type
        self.save_directed_edges = save_directed_edges
        super().__init__(
            root, name, val_ratio, test_ratio, debug, seed, num_training_tails_to_tile_per_class,
            transform, pre_transform, **kwargs,
        )
        self.load_rwpe = load_rwpe
        self.load_lepe = load_lepe
        if self.load_rwpe:
            self.global_data.pe = self.get_cached_rwpe_data(
                path=os.path.join(root, self.__class__.__name__.upper(), "rwpe"),
                data=self.global_data, walk_length=self.global_data.x.size(1),
            )
        elif self.load_lepe:
            self.global_data.pe = self.get_cached_lepe_data(
                path=os.path.join(root, self.__class__.__name__.upper(), "lepe"),
                data=self.global_data, k=self.global_data.x.size(1),
            )

    def _get_important_elements(self):
        ie = super()._get_important_elements()
        ie["save_directed_edges"] = "directed" if self.save_directed_edges else "undirected"
        return ie

    def load(self):
        """
        DatasetSubGNN attributes example
            - data: Data(edge_index=[2, 435110], obs_x=[11754], x=[34646, 1], y=[2400])
            - global_data: Data(edge_index=[2, 6476348], x=[14587, 64])
        """
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.global_data = torch.load(self.processed_paths[1], map_location=torch.device("cpu"))
        meta = torch.load(self.processed_paths[2])
        self.num_start = 0
        self.num_train = self.num_train_original = int(meta[0])
        self.num_val = self.num_val_original = int(meta[1])

    @property
    def splits(self):
        return [self.num_train, self.num_train + self.num_val]

    def set_num_start_train_val(self,
                                num_or_ratio_start: Union[int, float],
                                num_or_ratio_train: Union[int, float],
                                num_or_ratio_val: Union[int, float]):
        num_all = len(self)
        num_start = int(num_all * num_or_ratio_start) if isinstance(num_or_ratio_start, float) else num_or_ratio_start
        num_train = int(num_all * num_or_ratio_train) if isinstance(num_or_ratio_train, float) else num_or_ratio_train
        num_val = int(num_all * num_or_ratio_val) if isinstance(num_or_ratio_val, float) else num_or_ratio_val
        cprint(f"Set num_start, num_train and num_val to [{num_start}, {num_train}, {num_val}] "
               f"(Defaults: [{self.num_start}, {self.num_train}, {self.num_val}])", "green")
        self.num_start = num_start
        self.num_train = num_train
        self.num_val = num_val

    def set_num_start_train_by_num_train_per_class(self, num_train_per_class: int):
        new_num_train = self.num_classes * num_train_per_class
        self.num_start = self.num_train - new_num_train
        self.num_train = new_num_train

    @property
    def raw_file_names(self):
        return ["edge_list.txt", "subgraphs.pth", f"{self.embedding_type}_embeddings.pth"]

    @property
    def processed_file_names(self):
        return ["data.pt", f"global_{self.embedding_type}.pt", "meta.pt"]

    def download(self):
        raise FileNotFoundError("Please download: {} \n\t at {} \n\t from {}".format(
            self.raw_file_names, self.raw_dir, self.url,
        ))

    def process(self):
        global_data, data_train, data_val, data_test = read_subgnn_data(
            *self.raw_paths, save_directed_edges=self.save_directed_edges, debug=self.debug,
        )
        self.process_common(global_data, data_train, data_val, data_test)

    def process_common(self, global_data, data_train, data_val, data_test):
        data_total = data_train + data_val + data_test
        if self.pre_transform is not None:
            data_total = [self.pre_transform(d) for d in tqdm(data_total)]
            cprint("Pre-transformed: {}".format(self.pre_transform), "green")

        torch.save(self.collate(data_total), self.processed_paths[0])
        cprint("Saved data at {}".format(self.processed_paths[0]), "blue")
        torch.save(global_data, self.processed_paths[1])
        cprint("Saved global_data at {}".format(self.processed_paths[1]), "blue")

        self.num_train = len(data_train)
        self.num_val = len(data_val)
        torch.save(torch.as_tensor([self.num_train, self.num_val]).long(), self.processed_paths[2])

        self._logging_args()

    @staticmethod
    @fscaches(path_attrname_in_kwargs="path", verbose=True)
    def get_random_walk_pe(path, data: Data, walk_length: int, key=None):
        return AddRandomWalkPE(walk_length=walk_length, attr_name=None)(data).x

    @staticmethod
    @fscaches(path_attrname_in_kwargs="path", verbose=True)
    def get_cached_rwpe_data(path, data: Data, walk_length: int, key=None) -> Data:
        return AddRandomWalkPE(walk_length=walk_length, attr_name="pe")(data).pe

    @staticmethod
    @fscaches(path_attrname_in_kwargs="path", verbose=True)
    def get_le_pe(path, data: Data, k: int, key=None) -> Data:
        return AddLaplacianEigenvectorPE(k=k, attr_name=None, is_undirected=True)(data).x

    @staticmethod
    @fscaches(path_attrname_in_kwargs="path", verbose=True)
    def get_cached_lepe_data(path, data: Data, k: int, key=None) -> Data:
        return AddLaplacianEigenvectorPE(k=k, attr_name="pe", is_undirected=True)(data).pe


class SynSubgraphGLASSDataset(SubgraphDataset):

    def __init__(self, root, name, embedding_type="one",
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False,
                 transform=None, pre_transform=None, **kwargs):

        __embedding_type__ = "one"

        super().__init__(root, name, __embedding_type__, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

        if "RWPE" in embedding_type or "LEPE" in embedding_type:
            PE = "RWPE" if "RWPE" in embedding_type else "LEPE"
            try:
                # e.g., RWPE_K_4
                K = int(re.search("K_([0-9]+)", embedding_type).group(1))
            except:
                K = 32  # default
            D = 64 - K

            if embedding_type.startswith(f"ones_1/64/{PE}"):
                self.global_data.x = torch.ones((self.global_data.x.size(0), D)).float() / D
            elif embedding_type.startswith(f"ones_64/{PE}"):
                self.global_data.x = torch.ones((self.global_data.x.size(0), D)).float()

            syn_path = os.path.join(root, self.__class__.__name__.upper(), "synthetic")
            if PE == "RWPE":
                self.global_data.x = self.get_random_walk_pe(
                    path=syn_path, data=self.global_data, walk_length=K,
                    key=f"x={self.global_data.x.min().item()}-{self.global_data.x.max().item()}",
                )
            else:
                self.global_data.x = self.get_le_pe(
                    path=syn_path, data=self.global_data, k=K,
                    key=f"x={self.global_data.x.min().item()}-{self.global_data.x.max().item()}",
                )

        elif "LDP" in embedding_type:
            D = 64 - 5
            if embedding_type == "ones_1/64/LDP":
                self.global_data.x = torch.ones((self.global_data.x.size(0), D)).float() / D
            elif embedding_type == "ones_64/LDP":
                self.global_data.x = torch.ones((self.global_data.x.size(0), D)).float()

            self.global_data = LocalDegreeProfile()(self.global_data)

        else:
            D = 64
            if embedding_type == "one_d=1":
                self.global_data.x = torch.ones((self.global_data.x.size(0), 1)).float()
            elif embedding_type == "ones_64":
                self.global_data.x = torch.ones((self.global_data.x.size(0), D)).float()
            elif embedding_type == "ones_1/64":
                self.global_data.x = torch.ones((self.global_data.x.size(0), D)).float() / D

    @property
    def raw_file_names(self):
        return ["tmp.npy"]

    @property
    def processed_file_names(self):
        return ["data.pt", f"global_{self.embedding_type}.pt", "meta.pt"]

    def download(self):
        super().download()

    def process(self):
        global_data, data_train, data_val, data_test = read_glass_syn_data(self.raw_paths[0])
        self.process_common(global_data, data_train, data_val, data_test)


class HPONeuro(SubgraphDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class HPOMetab(SubgraphDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class EMUser(SubgraphDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class PPIBP(SubgraphDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class Density(SynSubgraphGLASSDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class Coreness(SynSubgraphGLASSDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class CutRatio(SynSubgraphGLASSDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class Component(SynSubgraphGLASSDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 num_training_tails_to_tile_per_class=0, load_rwpe=False, load_lepe=False, transform=None,
                 pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, num_training_tails_to_tile_per_class, load_rwpe, load_lepe,
                         transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


def largest_component_diameter(data):
    # Convert edge_index to a NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Find connected components
    components = list(nx.connected_components(G))

    # Find the largest connected component
    largest_component = max(components, key=len)
    print("lc size: ", len(largest_component))

    # Create subgraph for the largest component
    subgraph = G.subgraph(largest_component)
    print("lc nx: ", subgraph)

    # Calculate the diameter of the largest component
    diameter = nx.diameter(subgraph)
    return diameter


if __name__ == '__main__':

    FIND_SEED = False  # NOTE: If True, find_seed_that_makes_balanced_datasets will be performed

    NAME = "HPOMetab"
    # WLKSRandomTree
    # PPIBP, HPOMetab, HPONeuro, EMUser
    # Density, Component, Coreness, CutRatio

    USE_RWPE = False
    USE_LEPE = False

    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    if NAME.startswith("WL"):
        E_TYPE = "no_embedding"
    elif NAME in ["Density", "Component", "Coreness", "CutRatio"]:
        E_TYPE = "ones_1/64/LEPE"
    else:
        E_TYPE = "glass"  # gin, graphsaint_gcn, glass

    DEBUG = False

    MORE_KWARGS = {}
    if USE_RWPE and NAME not in ["Density", "Component", "Coreness", "CutRatio"]:
        MORE_KWARGS["load_rwpe"] = True
    elif USE_LEPE and NAME not in ["Density", "Component", "Coreness", "CutRatio"]:
        MORE_KWARGS["load_lepe"] = True

    dts: SubgraphDataset = eval(NAME)(
        root=PATH,
        name=NAME,
        embedding_type=E_TYPE,
        debug=DEBUG,
        **MORE_KWARGS,
    )

    train_dts, val_dts, test_dts = dts.get_train_val_test()

    dts.print_summary()

    cprint("Train samples", "yellow")
    for i, b in enumerate(train_dts):
        print(b)
        if i >= 5:
            break

    cprint("Validation samples", "yellow")
    for i, b in enumerate(val_dts):
        print(b)
        if i >= 5:
            break

    cprint("global_data samples", "yellow")
    print(dts.global_data)
    print("Avg. degree: ", dts.global_data.edge_index.size(1) / dts.global_data.num_nodes)
    if hasattr(dts.global_data, "pe"):
        print("PE", dts.global_data.pe)

    cprint("All subgraph samples", "magenta")
    print(dts.data)
    try:
        for k, vs in dts.y_stat_dict().items():
            print(k, [round(v, 3) for v in vs])
            for v in vs:
                print(round(v, 3))
    except AttributeError:
        pass

    print("LCD of global_data: ", largest_component_diameter(dts.global_data))
