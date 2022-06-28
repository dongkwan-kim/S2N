from collections import Counter
from pprint import pprint
from typing import List, Dict, Tuple
import os

import torch
from sklearn.preprocessing import MultiLabelBinarizer
from termcolor import cprint
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, sort_edge_index, to_undirected, from_networkx
import networkx as nx
from torch_geometric.utils.num_nodes import maybe_num_nodes

from tqdm import tqdm

from data_base import DatasetBase
from data_sub_utils import save_subgraphs
from dataset_wl import generate_random_subgraph_by_walk, WL4PatternNet, WL4PatternConv, \
    generate_random_subgraph_by_k_hop
from utils import from_networkx_customized_ordering, to_directed


def read_subgnn_data(edge_list_path, subgraph_path,
                     embedding_path=None, save_directed_edges=False, debug=False):
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
    try:
        xs = torch.load(embedding_path)  # feature matrix should be initialized to the node embeddings
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
            y = y.view(1, -1).long()
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

    return train_sub_g, train_sub_y, val_sub_g, val_sub_y, test_sub_g, test_sub_y


class SubgraphDataset(DatasetBase):
    url = "https://github.com/mims-harvard/SubGNN"

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):
        assert embedding_type in ["gin", "graphsaint_gcn", "no_embedding"]
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


class Density(SubgraphDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):
        assert embedding_type == "graphsaint_gcn"
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class Coreness(SubgraphDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):
        assert embedding_type == "graphsaint_gcn"
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class CutRatio(SubgraphDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):
        assert embedding_type == "graphsaint_gcn"
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class CC(SubgraphDataset):

    def __init__(self, root, name, embedding_type,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):
        assert embedding_type == "graphsaint_gcn"
        super().__init__(root, name, embedding_type, val_ratio, test_ratio,
                         save_directed_edges, debug, seed, transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class WLHistSubgraph(SubgraphDataset):

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
        assert network_generator.startswith("nx.")
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
                prev_diff_ratio.append((y_before != y_curr).sum().item() / y_curr.size(0))
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

        # dump to subgraph_path = raw_paths[1]
        sub_x = generate_random_subgraph_by_k_hop(
            data, num_subgraphs=self.num_subgraphs, subgraph_size=self.subgraph_size, k=1)
        wl = WL4PatternNet(
            num_layers=self.wl_max_hop, x_type_for_hists=self.wl_x_type_for_hists,
            clustering_name="KMeans", n_clusters=self.wl_num_color_clusters,  # clustering & kwargs
        )
        colors, hists, clusters = wl(sub_x, data.x, data.edge_index)
        hist_cluster_list = []
        for wl_step, (co, hi, cl) in enumerate(zip(colors, hists, clusters)):
            _hist_cluster = WL4PatternConv.to_cluster(
                hi, clustering_name="KMeans", n_clusters=self.wl_num_hist_clusters)
            hist_cluster_list.append(_hist_cluster.view(-1, 1))
        hist_cluster = torch.cat(hist_cluster_list, dim=-1)

        save_subgraphs(
            path=self.raw_paths[1],
            nodes_in_subgraphs=sub_x.tolist(),
            labels=["+".join(str(v) for v in hc)
                    for hc in hist_cluster.tolist()],
        )

    def process(self):
        super().process()


class WLHistSubgraphBA(WLHistSubgraph):

    def __init__(self, root, name, embedding_type, ba_n, ba_m, ba_seed,
                 num_subgraphs: int, subgraph_size: int, wl_hop_to_use: int, wl_max_hop: int,
                 wl_x_type_for_hists: str = "color", wl_num_color_clusters: int = None, wl_num_hist_clusters: int = 2,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):

        network_generator = "nx.barabasi_albert_graph"
        network_args = [ba_n, ba_m, ba_seed]
        super().__init__(root, name, embedding_type, network_generator, network_args, num_subgraphs, subgraph_size,
                         wl_hop_to_use, wl_max_hop, wl_x_type_for_hists, wl_num_color_clusters, wl_num_hist_clusters,
                         val_ratio, test_ratio, save_directed_edges, debug, seed, transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


class WLHistSubgraphER(WLHistSubgraph):

    def __init__(self, root, name, embedding_type, er_n, er_p, er_seed,
                 num_subgraphs: int, subgraph_size: int, wl_hop_to_use: int, wl_max_hop: int,
                 wl_x_type_for_hists: str = "color", wl_num_color_clusters: int = None, wl_num_hist_clusters: int = 2,
                 val_ratio=None, test_ratio=None, save_directed_edges=False, debug=False, seed=42,
                 transform=None, pre_transform=None, **kwargs):

        network_generator = "nx.erdos_renyi_graph"
        network_args = [er_n, er_p, er_seed]
        super().__init__(root, name, embedding_type, network_generator, network_args, num_subgraphs, subgraph_size,
                         wl_hop_to_use, wl_max_hop, wl_x_type_for_hists, wl_num_color_clusters, wl_num_hist_clusters,
                         val_ratio, test_ratio, save_directed_edges, debug, seed, transform, pre_transform, **kwargs)

    def download(self):
        super().download()

    def process(self):
        super().process()


if __name__ == '__main__':

    TYPE = "WLHistSubgraphBA"
    # WLHistSubgraphBA, WLHistSubgraphER
    # PPIBP, HPOMetab, HPONeuro, EMUser
    # Density, CC, Coreness, CutRatio

    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    if TYPE.startswith("WL"):
        E_TYPE = "no_embedding"
    else:
        E_TYPE = "gin"  # gin, graphsaint_gcn
    DEBUG = False
    WL_HIST_KWARGS = {
        "num_subgraphs": 1500,
        "subgraph_size": 10,
        "wl_hop_to_use": None,
        "wl_max_hop": 4,
        "wl_x_type_for_hists": "cluster",  # color, cluster
        "wl_num_color_clusters": None,
        "wl_num_hist_clusters": 2,
    }
    if TYPE == "WLHistSubgraphBA":
        KWARGS = {
            "ba_n": 10000, "ba_m": 10,  # 10, 15, 20
            "ba_seed": 42,
            **WL_HIST_KWARGS,
        }
    elif TYPE == "WLHistSubgraphER":
        KWARGS = {
            "er_n": 10000, "er_p": 0.003,  # 0.002, 0.003, 0.004
            "er_seed": 42,
            **WL_HIST_KWARGS,
        }
    else:
        KWARGS = {}

    dts = eval(TYPE)(
        root=PATH,
        name=TYPE,
        embedding_type=E_TYPE,
        debug=DEBUG,
        **KWARGS,
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
    except AttributeError:
        pass
