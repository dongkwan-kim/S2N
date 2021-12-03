from pprint import pprint
from typing import List, Dict, Tuple

import torch
from sklearn.preprocessing import MultiLabelBinarizer
from termcolor import cprint
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, sort_edge_index
import networkx as nx

from tqdm import tqdm

from data_base import DatasetBase
from utils import from_networkx_customized_ordering, to_directed


def read_subgnn_data(edge_list_path, subgraph_path, embedding_path, save_directed_edges, debug=False):
    """
    Read in the subgraphs & their associated labels
    Reference: https://github.com/mims-harvard/SubGNN/blob/main/SubGNN/SubGNN.py#L519
    """
    # read list of node ids for each subgraph & their labels
    train_nodes, _train_ys, val_nodes, _val_ys, test_nodes, _test_ys = read_subgraphs(subgraph_path)
    cprint("Loaded subgraphs at {}".format(subgraph_path), "green")

    # check if the dataset is multilabel (e.g. HPO-NEURO)
    if type(_train_ys) == list:
        all_labels = _train_ys + _val_ys + _test_ys
        mlb = MultiLabelBinarizer()
        mlb.fit(all_labels)
        train_sub_ys = torch.Tensor(mlb.transform(_train_ys)).long()
        val_sub_ys = torch.Tensor(mlb.transform(_val_ys)).long()
        test_sub_ys = torch.Tensor(mlb.transform(_test_ys)).long()
    else:
        train_sub_ys, val_sub_ys, test_sub_ys = _train_ys, _val_ys, _test_ys

    # Initialize pretrained node embeddings
    xs = torch.load(embedding_path)  # feature matrix should be initialized to the node embeddings
    # xs_with_zp = torch.cat([torch.zeros(1, xs.shape[1]), xs], 0)  # there's a zeros in the first index for padding
    cprint("Loaded embeddings at {}".format(embedding_path), "green")

    # read networkx graph from edge list
    global_nxg: nx.Graph = nx.read_edgelist(edge_list_path)
    cprint("Loaded global_graph at {}".format(edge_list_path), "green")
    global_data = from_networkx_customized_ordering(global_nxg, ordering="keep")
    cprint("Converted global_graph to PyG format", "green")
    global_data.edge_index = sort_edge_index(global_data.edge_index)
    global_data.x = xs

    train_data_list = get_data_list_from_subgraphs(global_data.edge_index, train_nodes, train_sub_ys,
                                                   save_directed_edges=save_directed_edges, debug=debug)
    cprint("Converted train_subgraph to PyG format", "green")
    val_data_list = get_data_list_from_subgraphs(global_data.edge_index, val_nodes, val_sub_ys,
                                                 save_directed_edges=save_directed_edges, debug=debug)
    cprint("Converted val_subgraph to PyG format", "green")
    test_data_list = get_data_list_from_subgraphs(global_data.edge_index, test_nodes, test_sub_ys,
                                                  save_directed_edges=save_directed_edges, debug=debug)
    cprint("Converted test_subgraph to PyG format", "green")
    return global_data, train_data_list, val_data_list, test_data_list


def get_data_list_from_subgraphs(global_edge_index, sub_nodes: List[List[int]], sub_ys,
                                 save_directed_edges, debug=False):
    data_list = []
    for idx, (x_index, y) in enumerate(zip(sub_nodes, tqdm(sub_ys))):
        x_index = torch.Tensor(x_index).long().view(-1, 1)
        if len(y.size()) == 0:
            y = torch.Tensor([y]).long()
        else:
            y = y.view(1, -1).float()
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

    # Parse data
    with open(subgraph_path) as fin:
        subgraph_idx = 0
        for line in fin:
            nodes = [int(n) for n in line.split("\t")[0].split("-") if n != ""]
            if len(nodes) != 0:
                if len(nodes) == 1:
                    print("G with one node: ", nodes)
                l = line.split("\t")[1].split("-")
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

    if len(val_mask) < len(test_mask):
        return train_sub_g, train_sub_y, test_sub_g, test_sub_y, val_sub_g, val_sub_y

    return train_sub_g, train_sub_y, val_sub_g, val_sub_y, test_sub_g, test_sub_y


class SubgraphDataset(DatasetBase):

    url = "https://github.com/mims-harvard/SubGNN"

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):
        assert embedding_type in ["gin", "graphsaint_gcn"]
        self.embedding_type = embedding_type
        self.save_directed_edges = save_directed_edges
        super().__init__(
            root, name, val_ratio, test_ratio, debug, seed,
            transform, pre_transform, **kwargs,
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
        self.global_data = torch.load(self.processed_paths[1])
        meta = torch.load(self.processed_paths[2])
        self.num_train = int(meta[0])
        self.num_val = int(meta[1])

    @property
    def splits(self):
        return [self.num_train, self.num_train + self.num_val]

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

        data_total = data_train + data_val + data_test
        if self.pre_transform is not None:
            data_total = [self.pre_transform(d) for d in tqdm(data_total)]
            cprint("Pre-transformed: {}".format(self.pre_transform), "green")

        torch.save(self.collate(data_total), self.processed_paths[0])
        cprint("Saved data at {}".format(self.processed_paths[0]), "green")
        torch.save(global_data, self.processed_paths[1])
        cprint("Saved global_data at {}".format(self.processed_paths[1]), "green")

        self.num_train = len(data_train)
        self.num_val = len(data_val)
        torch.save(torch.as_tensor([self.num_train, self.num_val]).long(), self.processed_paths[2])

        self._logging_args()


class HPONeuro(SubgraphDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class HPOMetab(SubgraphDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class EMUser(SubgraphDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class PPIBP(SubgraphDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


if __name__ == '__main__':

    TYPE = "EMUser"  # PPIBP, HPOMetab, HPONeuro, EMUser

    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    E_TYPE = "gin"  # gin, graphsaint_gcn
    DEBUG = False

    if TYPE == "HPONeuro":  # multi-label
        dts = HPONeuro(
            root=PATH,
            name="HPONeuro",
            embedding_type=E_TYPE,
            debug=DEBUG,
        )
    elif TYPE == "HPOMetab":
        dts = HPOMetab(
            root=PATH,
            name="HPOMetab",
            embedding_type=E_TYPE,
            debug=DEBUG,
        )
    elif TYPE == "EMUser":
        dts = EMUser(
            root=PATH,
            name="EMUser",
            embedding_type=E_TYPE,
            debug=DEBUG,
        )
    elif TYPE == "PPIBP":
        dts = PPIBP(
            root=PATH,
            name="PPIBP",
            embedding_type=E_TYPE,
            debug=DEBUG,
        )
    else:
        raise ValueError

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

    print(dts.global_data)

