import os
import shutil
from collections import Counter
from typing import List, Union

import networkx as nx
import numpy as np
import torch
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MultiLabelBinarizer
from termcolor import cprint
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, sort_edge_index, to_undirected, from_networkx
from torch_geometric.utils.num_nodes import maybe_num_nodes
from tqdm import tqdm

from data_base import DatasetBase
from data_sub_utils import save_subgraphs
from dataset_wl import generate_random_subgraph_by_walk, WL4PatternNet, WL4PatternConv, \
    generate_random_k_hop_subgraph, generate_random_subgraph_batch_by_sampling_0_to_l_to_d, \
    nx_rewired_balanced_tree
from utils import from_networkx_customized_ordering, to_directed, unbatch


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
    global_data = Data(edge_index=edge, x=torch.ones((len(node), 1)).float(), num_nodes=len(node))
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
                 transform=None, pre_transform=None, **kwargs):
        assert embedding_type in ["gin", "graphsaint_gcn", "no_embedding", "glass", "one"]
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
        self.global_data = torch.load(self.processed_paths[1], map_location=torch.device("cpu"))
        meta = torch.load(self.processed_paths[2])
        self.num_start = 0
        self.num_train = int(meta[0])
        self.num_val = int(meta[1])

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


class SynSubgraphGLASSDataset(SubgraphDataset):

    def __init__(self, root, name, embedding_type="one",
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, transform, pre_transform, **kwargs)

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


class Density(SynSubgraphGLASSDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class Coreness(SynSubgraphGLASSDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class CutRatio(SynSubgraphGLASSDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class Component(SynSubgraphGLASSDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class WLKSubgraph(SubgraphDataset):

    def __init__(self, root, name, embedding_type,
                 network_generator: str, network_args: list,
                 num_subgraphs: int, subgraph_size: int,
                 wl_hop_to_use: int, wl_max_hop: int, wl_x_type_for_hists: str = "color",
                 wl_num_color_clusters: int = None, wl_num_hist_clusters: int = 2,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):
        """Params specific for WLHistSubgraph
        :param network_generator: String form of graph generators in networkx.
            e.g., nx.barabasi_albert_graph, nx.grid_2d_graph,
                See https://networkx.org/documentation/stable/reference/generators.html
        :param network_args: Arguments for network_generator.
        :param num_subgraphs: Number of subgraphs to generate.
        :param subgraph_size: Size of subgraphs to generate.
        :param wl_hop_to_use: The hop to use in training and testing.
        :param wl_max_hop: Number of hops to run WL algorithm.
        :param wl_x_type_for_hists: Whether to use clustering for colors: cluster or color
        :param wl_num_color_clusters: If wl_x_type_for_hists == cluster, number of clusters to use.
        :param wl_num_hist_clusters: Number of clusters by histograms, i.e., number of classes.
        """
        self.network_generator = network_generator
        self.network_args = network_args

        self.num_subgraphs = num_subgraphs
        self.subgraph_size = subgraph_size

        self.wl_hop_to_use = wl_hop_to_use
        self.wl_max_hop = wl_max_hop
        self.wl_x_type_for_hists = wl_x_type_for_hists
        self.wl_num_color_clusters = wl_num_color_clusters or self.wl_max_hop
        self.wl_num_hist_clusters = wl_num_hist_clusters
        assert self.wl_x_type_for_hists in ["cluster", "color"]
        assert network_generator.startswith("nx")
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, transform, pre_transform, **kwargs)

        if wl_hop_to_use is not None:
            # Use one wl step of y & cache the original ys in _y.
            assert 0 < wl_hop_to_use <= wl_max_hop
            self.data._y = self.data.y.clone()
            self.data.y = self.data._y[:, wl_hop_to_use - 1]

    @property
    def num_classes(self) -> int:
        # NOTE: If we use original num_classes, it will be self.wl_hop_to_use.
        return self.wl_num_hist_clusters

    def y_stat_dict(self):
        try:
            y_2d = self.data._y
        except:
            y_2d = self.data.y
        assert y_2d.dim() == 2

        from torch_geometric.data import Batch
        _, _, test_data = self.get_train_val_test()
        test_y_2d = Batch.from_data_list(test_data).y

        y_before = None
        major_class_ratio_all, major_class_ratio_test, prev_diff_ratio = [], [], []
        for y_idx in range(y_2d.size(-1)):
            y_curr = y_2d[:, y_idx].long()
            class_counter_all = Counter(y_curr.tolist())
            class_counter_test = Counter(test_y_2d[:, y_idx].long().tolist())
            major_class_ratio_all.append(max(class_counter_all.values()) / sum(class_counter_all.values()))
            major_class_ratio_test.append(max(class_counter_test.values()) / sum(class_counter_test.values()))
            if y_before is not None:
                pdr = (y_before != y_curr).sum().item() / y_curr.size(0)
                prev_diff_ratio.append(pdr if pdr <= 0.5 else pdr - 0.5)
            y_before = y_curr
        return {
            "major_class_ratio_all": major_class_ratio_all,
            "major_class_ratio_test": major_class_ratio_test,
            "prev_diff_ratio": prev_diff_ratio,
        }

    @property
    def key_dir(self):
        return os.path.join(self.root, self.__class__.__name__.upper(),
                            "_".join([str(e) for e in self._get_important_elements().values()]))

    @property
    def raw_dir(self):
        return os.path.join(self.key_dir, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.key_dir, "processed")

    @property
    def raw_file_names(self):
        return ["edge_list.txt", "subgraphs.pth"]

    def _get_important_elements(self):
        ie = super()._get_important_elements()
        ie.update({
            "network_generator": self.network_generator,
            "network_args": self.network_args,
            "num_subgraphs": self.num_subgraphs,
            "subgraph_size": self.subgraph_size,
            "wl_max_hop": self.wl_max_hop,
            "wl_x_type_for_hists": self.wl_x_type_for_hists,
            "wl_num_hist_clusters": self.wl_num_hist_clusters,
        })
        if self.wl_x_type_for_hists == "cluster":
            ie["wl_num_color_clusters"] = self.wl_num_color_clusters
        return ie

    def download(self):  # generate
        g: nx.Graph = eval(self.network_generator)(*self.network_args)
        edge_index = to_undirected(from_networkx(g).edge_index.long())
        data = Data(x=torch.ones(maybe_num_nodes(edge_index)).long(),
                    edge_index=edge_index)

        # dump to edge_list_path = raw_paths[0]
        nx.write_edgelist(g, self.raw_paths[0], data=False)

        L = self.wl_max_hop
        init_sub_x, batch_list = generate_random_subgraph_batch_by_sampling_0_to_l_to_d(
            data, num_subgraphs=self.num_subgraphs, subgraph_size=self.subgraph_size,
            k=1, l=L,
            subgraph_generation_method="generate_random_k_hop_subgraph",
        )
        assert len(batch_list) == (1 + L + 1)

        last_hist_cluster_list = []
        for ith, i_hop_batch in enumerate(tqdm(batch_list, desc="WL-coloring-to-labels")):
            wl = WL4PatternNet(
                num_layers=self.wl_max_hop * 2 + 1,
                x_type_for_hists=self.wl_x_type_for_hists,
                clustering_name="MiniBatchKMeans",
                n_clusters=self.wl_num_color_clusters,  # clustering & kwargs
            )
            sub_x = unbatch(i_hop_batch.initial_node_index, i_hop_batch.initial_node_index_batch)
            x_as_colors = torch.ones(i_hop_batch.num_nodes).long()
            wl_rets = wl(
                sub_x, x_as_colors, i_hop_batch.edge_index, hist_norm=True, use_tqdm=False,
            )
            hists, colors, clusters = wl_rets["hists"], wl_rets["colors"], wl_rets["clusters"]
            print("\n", ith, hists[-1].size(), colors[-1].size())

            # Remove not used colors in the histogram. (S, #colors)
            last_hist = torch.from_numpy(VarianceThreshold().fit_transform(
                hists[-1].numpy()
            ))
            last_hist_cluster = WL4PatternConv.to_cluster(
                last_hist, clustering_name="KMeans", n_clusters=self.wl_num_hist_clusters)
            last_hist_cluster_list.append(last_hist_cluster.view(-1, 1))  # (S, 1)

        hist_cluster = torch.cat(last_hist_cluster_list, dim=-1)  # (S, C)

        if isinstance(init_sub_x, list):
            nodes_in_subgraphs = [nodes.tolist() for nodes in init_sub_x]
        else:
            nodes_in_subgraphs = init_sub_x.tolist()

        save_subgraphs(
            path=self.raw_paths[1],
            nodes_in_subgraphs=nodes_in_subgraphs,
            labels=["+".join(str(v) for v in hc)
                    for hc in hist_cluster.tolist()],
        )

    def process(self):
        super().process()


class WLKSRandomTree(WLKSubgraph):

    def __init__(self, root, name, embedding_type,
                 num_nodes: int, num_branch: int, height: int, rewiring_ratio: float, wl_seed: int,
                 num_subgraphs: int, subgraph_size: int, wl_hop_to_use: int, wl_max_hop: int,
                 wl_x_type_for_hists: str = "color", wl_num_color_clusters: int = None,
                 wl_num_hist_clusters: int = 2,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):
        # num_nodes, num_branch, height, rewiring_ratio, seed
        network_generator = "nx_rewired_balanced_tree"
        network_args = [num_nodes, num_branch, height, rewiring_ratio]
        network_args.append(self.seed_that_makes_balanced_datasets(wl_seed, num_subgraphs, *network_args))
        super().__init__(root, name, embedding_type, network_generator, network_args,
                         num_subgraphs, subgraph_size, wl_hop_to_use, wl_max_hop, wl_x_type_for_hists,
                         wl_num_color_clusters, wl_num_hist_clusters,
                         val_ratio, test_ratio, save_directed_edges, debug, seed,
                         transform, pre_transform, **kwargs)

    def seed_that_makes_balanced_datasets(self, wl_seed, *args):
        if wl_seed is not None:
            return wl_seed
        else:
            return {
                (2000, 10000, 4, 8, 0.05): 8,
                (2000, 10000, 4, 8, 0.025): 3,
            }[args]

    def download(self):
        from utils import make_deterministic_everything
        make_deterministic_everything(self.network_args[-1])
        super().download()

    def process(self):
        super().process()


def find_seed_that_makes_balanced_datasets(seed_name="wl_seed", class_ratio_thres=0.8, **kwargs):
    min_of_max_vs, seed_at_min_of_max_vs = 999, None
    good_seeds = []
    for seed in range(15):
        assert seed_name in kwargs
        kwargs[seed_name] = seed
        trial_dataset: WLKSubgraph = eval(NAME)(
            root=PATH,
            name=NAME,
            embedding_type=E_TYPE,
            debug=DEBUG,
            **kwargs,
        )
        mcrt_list = trial_dataset.y_stat_dict()["major_class_ratio_test"]
        mcrt_list.pop(-1)  # todo: remove
        if max(mcrt_list) < class_ratio_thres:
            cprint(f"Good seed found: {seed} ({[round(v, 3) for v in mcrt_list]})",
                   "green")
            good_seeds.append(seed)
        else:
            cprint(f"Bad seed: {seed} ({[round(v, 3) for v in mcrt_list]}), "
                   f"removing: {trial_dataset.key_dir}", "red")
            shutil.rmtree(trial_dataset.key_dir)

        if min_of_max_vs > max(mcrt_list):
            min_of_max_vs = max(mcrt_list)
            seed_at_min_of_max_vs = seed
        print(f"\t- Current min_of_max_vs is {min_of_max_vs} at seed {seed_at_min_of_max_vs}")
        print(f"\t- Good seeds are {good_seeds}")


if __name__ == '__main__':

    FIND_SEED = False  # NOTE: If True, find_seed_that_makes_balanced_datasets will be performed

    NAME = "Density"
    # WLKSRandomTree
    # PPIBP, HPOMetab, HPONeuro, EMUser
    # Density, Component, Coreness, CutRatio

    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    if NAME.startswith("WL"):
        E_TYPE = "no_embedding"
    elif NAME in ["Density", "Component", "Coreness", "CutRatio"]:
        E_TYPE = "one"
    else:
        E_TYPE = "gin"  # gin, graphsaint_gcn, glass
    DEBUG = False
    MORE_KWARGS = {
        "num_subgraphs": 2000,
        "subgraph_size": None,  # NOTE: Using None will use ego-graphs
        "wl_hop_to_use": None,
        "wl_max_hop": 2,
        "wl_x_type_for_hists": "color",  # color, cluster
        "wl_num_color_clusters": None,
        "wl_num_hist_clusters": 2,
    }
    if NAME == "WLKSRandomTree":
        MORE_KWARGS = {
            "num_nodes": 10000,
            "num_branch": 4,
            "height": 8,
            "rewiring_ratio": 0.05,  # 0.05, 0.025
            "wl_seed": None,  # NOTE: Using None will use wl_seed_that_makes_balanced_datasets
            **MORE_KWARGS,
        }
    else:
        MORE_KWARGS = {}

    if FIND_SEED:
        find_seed_that_makes_balanced_datasets(**MORE_KWARGS)
        exit("Exit with success")

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

    cprint("All subgraph samples", "magenta")
    print(dts.data)
    try:
        for k, vs in dts.y_stat_dict().items():
            print(k, [round(v, 3) for v in vs])
            for v in vs:
                print(round(v, 3))
    except AttributeError:
        pass
