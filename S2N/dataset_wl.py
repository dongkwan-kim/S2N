from collections import defaultdict, Counter
from itertools import zip_longest, accumulate
from pprint import pprint
from typing import Optional, List, Dict, Any, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from termcolor import cprint
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data, Batch
from torch_geometric.nn import WLConv
from torch_geometric.transforms import BaseTransform
from torch_geometric.typing import Adj
from torch_geometric.utils import to_undirected, to_networkx, from_networkx, k_hop_subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor
from torch_scatter import scatter_add
from tqdm import tqdm

from utils import torch_choice

torch.manual_seed(42)


class SliceYByIndex(BaseTransform):

    def __init__(self, y_idx):
        self.y_idx = y_idx

    def __call__(self, data: Data):
        data._y = data.y.clone()
        data.y = data.y[:, self.y_idx]
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}({self.y_idx})'


class ReplaceXWithWL4Pattern(BaseTransform):

    def __init__(self, num_layers, wl_step_to_use, wl_type_to_use,
                 num_color_clusters=None, clustering_name="KMeans",
                 cache_path=None, cumcat=False, num_pca_features=None,
                 **kwargs):
        self.wl = WL4PatternNet(
            num_layers=num_layers,
            x_type_for_hists="all",  # todo: replace by wl_type_to_use
            clustering_name=clustering_name,
            n_clusters=num_color_clusters or num_layers,  # clustering & kwargs
            use_clustering_validation=False,
            compute_last_only=False,
            **kwargs,
        )
        self.wl_step_to_use = wl_step_to_use
        self.wl_type_to_use = wl_type_to_use
        self.cache_path = cache_path
        self.cumcat = cumcat  # cumulative cat for separated
        self.num_pca_features = num_pca_features

    def __call__(self, data: Union[Data, List[Data], List[List[Data]]]):
        # TODO: Functionally, it is working, but refactoring is needed, but some other day...
        """
        Data --> List[Data(x=[S, F], y=[S])] (one)
        List[Data] (connected) --> List[Data(x=[S, F], y=[S])] (same length)
        List[List[Data]] (separated) --> List[Data(x=[S, F], y=[S])] (same length)
        """
        num_split_list = None
        if isinstance(data, Data):
            data_list = [data]
        elif isinstance(data, list) and isinstance(data[0], list):  # separated: List[List[Data]]
            num_split_list = list(accumulate([0] + [len(d) for d in data]))
            data_list = [Batch.from_data_list(sum(data, []))]
            data_list[0].x_to_xs = torch.arange(data_list[0].batch.size(0))
        else:  # connected: List[Data]
            data_list = data
        # Here, data_list will be list of
        #   DataBatch(x=[Ng, 1], edge_index=[2, E], y=[S, 4], batch=[\sum Ns], ptr=[S+1], x_to_xs=[\sum Ns])

        try:
            data_ptr, hists_colors, hists_clusters = torch.load(self.cache_path)
            cprint(f"Load (data_ptr, hists_colors, hists_clusters) from {self.cache_path}", "green")

        except (FileNotFoundError, AttributeError):
            sub_x = []
            data_ptr, count = [0], 0
            for data in data_list:
                ptr_list = data.ptr.tolist()
                # x_to_xs, batch (or ptr) --> sub_x: List[Tensor]
                for prev, curr in zip(ptr_list, ptr_list[1:]):
                    sub_x.append(data.x_to_xs[prev:curr])
                    count += 1
                data_ptr.append(count)

            assert len(set(data.num_nodes for data in data_list)) == 1  # same num_nodes across data_list
            x_as_colors = torch.ones(data_list[0].num_nodes).long()  # as initial colors.

            assert len(set(data.edge_index.size(0) for data in data_list)) == 1  # same num_edges across data_list
            wl_rets = self.wl(sub_x, x_as_colors, data_list[0].edge_index)
            hists_colors, hists_clusters = wl_rets["hists_colors"], wl_rets["hists_clusters"]

            if self.cache_path is not None:
                torch.save((data_ptr, hists_colors, hists_clusters), self.cache_path)
                cprint(f"Saved: (data_ptr, hists_colors, hists_clusters) at {self.cache_path}", "blue")

        for data_idx, data in enumerate(data_list):
            p1, p2 = data_ptr[data_idx], data_ptr[data_idx + 1]
            if self.wl_type_to_use == "color":
                data.x = hists_colors[self.wl_step_to_use][p1:p2, :]
            else:  # cluster
                data.x = hists_clusters[self.wl_step_to_use][p1:p2, :]
            data.edge_index, data.batch, data.ptr, data.x_to_xs = None, None, None, None

        if num_split_list is not None:  # separated
            assert len(data_list) == 1
            data = data_list[0]
            if not self.cumcat:
                data_list = [Data(x=data.x[prev:curr, :], y=data.y[prev:curr])
                             for prev, curr in zip(num_split_list, num_split_list[1:])]
            else:
                data_list = [Data(x=data.x[:curr, :], y=data.y[:curr])
                             for _, curr in zip(num_split_list, num_split_list[1:])]
        # The output will be a List of Data(x=[S, F], y=[S])
        return data_list


class WL4PatternConv(WLConv):

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        if x.dim() > 1:
            assert (x.sum(dim=-1) == 1).sum() == x.size(0)
            x = x.argmax(dim=-1)  # one-hot -> integer.
        assert x.dtype == torch.long

        adj_t = edge_index
        if not isinstance(adj_t, SparseTensor):
            adj_t = SparseTensor(row=edge_index[1], col=edge_index[0],
                                 sparse_sizes=(x.size(0), x.size(0)))

        out = []
        _, col, _ = adj_t.coo()
        deg = adj_t.storage.rowcount().tolist()
        for node, neighbors in zip(x.tolist(), x[col].split(deg)):
            # Use idx without hash(.)
            # idx = hash(tuple([node] + neighbors.sort()[0].tolist()))
            idx = tuple([node] + neighbors.sort()[0].tolist())
            if idx not in self.hashmap:
                self.hashmap[idx] = len(self.hashmap)

            out.append(self.hashmap[idx])

        return torch.tensor(out, device=x.device)

    def color_pattern(self, color: Tensor, outtype="bow", preprocessor=None) -> Tensor:
        color_to_pattern = {v: k for k, v in self.hashmap.items()}
        assert len(self.hashmap) == len(color_to_pattern)

        neighbor_patterns = []
        for c in color.tolist():
            sp, *nep = color_to_pattern[c]
            neighbor_patterns.append(nep)

        if outtype == "bow":
            vectorizer = CountVectorizer(
                preprocessor=lambda _: _, tokenizer=lambda _: _,
                # min_df=5e-5,  # Should be given when the number of colors is large.
            )
            pattern_transformed = vectorizer.fit_transform(neighbor_patterns)
            pattern_vec = pattern_transformed.toarray()
            if preprocessor is not None:
                pattern_vec = eval(preprocessor)().fit_transform(pattern_vec)
            return torch.from_numpy(pattern_vec)

        else:
            raise NotImplementedError

    def color_pattern_cluster(self, color: Tensor,
                              pattern_outtype="bow",
                              pattern_preprocessor=None,
                              clustering_name="KMeans", **kwargs) -> Tensor:
        pattern_x = self.color_pattern(color, pattern_outtype, pattern_preprocessor)
        return self.to_cluster(pattern_x, clustering_name, **kwargs)

    def histogram(self, x: Tensor, batch: Optional[Tensor] = None,
                  norm: bool = False, num_colors=None) -> Tensor:
        r"""Given a node coloring :obj:`x`, computes the color histograms of
        the respective graphs (separated by :obj:`batch`)."""

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_colors = num_colors or len(self.hashmap)  # this is the only difference
        batch_size = int(batch.max()) + 1

        index = batch * num_colors + x
        out = scatter_add(torch.ones_like(index), index, dim=0,
                          dim_size=num_colors * batch_size)
        out = out.view(batch_size, num_colors)

        if norm:
            out = out.to(torch.float)
            out /= out.norm(dim=-1, keepdim=True)

        return out

    def subgraph_histogram(self,
                           subgraph_nodes: List[Tensor],
                           x_as_color: Tensor,
                           norm: bool = False, num_colors=None) -> Tensor:
        S = len(subgraph_nodes)
        sizes = torch.tensor([s.size(0) for s in subgraph_nodes], dtype=torch.long)
        batch = torch.arange(S).repeat_interleave(sizes)
        sub_x_as_color = x_as_color[torch.cat(subgraph_nodes)]
        sub_hist = self.histogram(sub_x_as_color, batch, norm=norm, num_colors=num_colors)
        return sub_hist

    @staticmethod
    def to_cluster(x: Tensor, clustering_name="KMeans", **kwargs) -> Tensor:
        assert clustering_name in ["KMeans", "MiniBatchKMeans"]
        clustering = eval(clustering_name)(**kwargs).fit(x.numpy())
        return torch.from_numpy(clustering.labels_).long()


class WL4PatternNet(torch.nn.Module):

    def __init__(self, num_layers, clustering_name="KMeans", x_type_for_hists="color",
                 use_clustering_validation=False, compute_last_only=True, **kwargs):
        super().__init__()
        self.convs = torch.nn.ModuleList([WL4PatternConv() for _ in range(num_layers)])

        self.clustering_name = clustering_name
        self.cluster_kwargs: Dict[str, Any] = {
            "KMeans": {"n_clusters": -1, "pattern_preprocessor": None},
            "MiniBatchKMeans": {"n_clusters": -1, "pattern_preprocessor": None,
                                "batch_size": 256 * 40, "max_iter": 100},
        }[clustering_name]
        self.cluster_kwargs.update(kwargs)

        self.x_type_for_hists = x_type_for_hists
        assert x_type_for_hists in ["color", "cluster", "all"]
        self.use_clustering_validation = use_clustering_validation
        self.compute_last_only = compute_last_only

    def validate_clustering(self, color, cluster, memo=""):
        cprint(f"Validating clustering: ({memo})", "green")
        co_to_cl = defaultdict(list)
        for co, cl in zip(color.tolist(), cluster.tolist()):
            co_to_cl[co].append(cl)
        co_to_cl_set = {}
        for co, cl_list in co_to_cl.items():
            co_to_cl_set[co] = set(cl_list)
            if len(set(cl_list)) != 1:
                cprint(f"Clustering validation failed:", "red")
                print(co, "->", set(cl_list), Counter(cl_list))
        print(f"{memo} co_to_cl passed.")

    def forward(self, sub_x: Union[Tensor, List[Tensor]], x, edge_index, hist_norm=True, use_tqdm=False) -> Dict:
        colors, clusters = [], []
        hists_colors, hists_clusters = [], []

        convs = tqdm(self.convs, desc="WL4PatternNet.forward") if use_tqdm else self.convs
        for i, conv in enumerate(convs):
            conv: WL4PatternConv
            x = conv(x, edge_index)

            colors.append(x)
            if self.use_clustering_validation:
                self.validate_clustering(colors[-1], clusters[-1], memo=f"{i + 1}-step")

            if self.x_type_for_hists in ["color", "all"]:
                if (self.compute_last_only and (i == len(convs) - 1)) or not self.compute_last_only:
                    hists_colors.append(conv.subgraph_histogram(list(sub_x), colors[-1], norm=hist_norm,
                                                                num_colors=len(conv.hashmap)))
            if self.x_type_for_hists in ["cluster", "all"]:
                if (self.compute_last_only and (i == len(convs) - 1)) or not self.compute_last_only:
                    clusters.append(conv.color_pattern_cluster(x, clustering_name=self.clustering_name,
                                                               **self.cluster_kwargs))
                    hists_clusters.append(conv.subgraph_histogram(list(sub_x), clusters[-1], norm=hist_norm,
                                                                  num_colors=self.cluster_kwargs["n_clusters"]))

        if self.x_type_for_hists == "all":
            hists_rets = {"hists_colors": hists_colors, "hists_clusters": hists_clusters}
        else:
            hists_rets = {"hists": hists_colors or hists_clusters}
        return {
            "colors": colors,
            "clusters": clusters,
            **hists_rets,
        }


def generate_random_subgraph_batch_by_sampling_0_to_l_to_d(
        global_data: Data, num_subgraphs, subgraph_size=None, k=1, l=2,
        subgraph_generation_method="generate_random_k_hop_subgraph",
        only_nodes_in_subgraphs=False, max_subgraph_size=None,
) -> (Union[Tensor, List[Tensor]], List[Data]):
    if subgraph_generation_method == "generate_random_k_hop_subgraph":
        nodes_in_subgraphs: Union[Tensor, List[Tensor]] = generate_random_k_hop_subgraph(
            global_data, num_subgraphs=num_subgraphs, subgraph_size=subgraph_size,
            k=k, max_subgraph_size=max_subgraph_size)
    elif subgraph_generation_method == "generate_random_subgraph_by_walk":
        assert subgraph_size is not None
        nodes_in_subgraphs: Tensor = generate_random_subgraph_by_walk(
            global_data, num_subgraphs=num_subgraphs, subgraph_size=subgraph_size)
    elif subgraph_generation_method == "hybrid":
        assert subgraph_size is not None
        nodes_in_subgraphs_1: Tensor = generate_random_k_hop_subgraph(
            global_data, num_subgraphs=num_subgraphs // 2, subgraph_size=subgraph_size, k=k)
        nodes_in_subgraphs_2: Tensor = generate_random_subgraph_by_walk(
            global_data, num_subgraphs=num_subgraphs // 2, subgraph_size=subgraph_size)
        nodes_in_subgraphs = torch.cat([nodes_in_subgraphs_1, nodes_in_subgraphs_2], dim=0)
    else:
        raise ValueError(f"Wrong subgraph_generation_method {subgraph_generation_method}")

    if only_nodes_in_subgraphs:
        return nodes_in_subgraphs

    batch_list = []
    hop_range = list(range(l + 1))
    # hop_range = []
    for ith_hop in tqdm(hop_range, desc="sampling_0_to_l_to_d"):
        ith_data_list = []
        for nodes in list(nodes_in_subgraphs):
            ith_nodes, ith_edge_index, inv, _ = k_hop_subgraph(
                nodes, ith_hop, global_data.edge_index,
                relabel_nodes=True)
            ith_data_list.append(Data(
                edge_index=ith_edge_index,
                num_nodes=ith_nodes.size(0),
                initial_node_index=inv,  # index of 'nodes' in 'ith_nodes'.
            ))
        ith_batch = Batch.from_data_list(ith_data_list, follow_batch=["initial_node_index"])
        # e.g., DataBatch(edge_index=[2, 1548380], num_nodes=419191, initial_node_index=[17014],
        #                 initial_node_index_batch=[17014], batch=[419191], ptr=[1501])
        batch_list.append(ith_batch)

    # diameter
    _diameter_batch = Batch.from_data_list([Data(x=nodes) for nodes in nodes_in_subgraphs])
    batch_list.append(Data(
        edge_index=global_data.edge_index,
        num_nodes=global_data.num_nodes,
        initial_node_index=_diameter_batch.x,
        initial_node_index_batch=_diameter_batch.batch,
    ))
    return nodes_in_subgraphs, batch_list


def generate_random_subgraph_by_walk(global_data: Data, num_subgraphs, subgraph_size):
    N, E = global_data.num_nodes, global_data.num_edges
    adj = SparseTensor(
        row=global_data.edge_index[0], col=global_data.edge_index[1],
        value=torch.arange(E, device=global_data.edge_index.device),
        sparse_sizes=(N, N))

    nodes_in_subgraphs = []
    start = torch.randint(0, N, (num_subgraphs * 2,), dtype=torch.long).flatten()
    for nodes in adj.random_walk(start, walk_length=(2 * subgraph_size - 1)):
        for size in range(subgraph_size, 2 * subgraph_size):
            unique_nodes = torch.unique(nodes[:size])
            if unique_nodes.size(0) == subgraph_size:
                nodes_in_subgraphs.append(unique_nodes)
                break
        if len(nodes_in_subgraphs) == num_subgraphs:
            break

    nodes_in_subgraphs = torch.stack(nodes_in_subgraphs)
    assert list(nodes_in_subgraphs.size()) == [num_subgraphs, subgraph_size]
    return nodes_in_subgraphs


def generate_random_k_hop_subgraph(global_data: Data, num_subgraphs,
                                   subgraph_size=None, k=1,
                                   max_subgraph_size=None) -> Union[Tensor, List[Tensor]]:
    assert not (subgraph_size is not None and max_subgraph_size is not None)
    N, E = global_data.num_nodes, global_data.num_edges
    nodes_in_subgraphs = []
    start = torch.unique(torch.randint(0, N, (num_subgraphs * 3,), dtype=torch.long).flatten(),
                         sorted=False)
    start = start[torch.randperm(start.size(0))]
    for n_idx in tqdm(start, desc=f"generate_random_subgraph_by_k_hop (#={num_subgraphs})", total=num_subgraphs * 2):
        subset, _, idx_of_start_in_subset, _ = k_hop_subgraph([n_idx], k, global_data.edge_index, num_nodes=N)
        _S = subset.size(0)

        need_sampling = False
        if subgraph_size is None and max_subgraph_size is None:
            nodes_in_subgraphs.append(subset)

        elif subgraph_size is None and max_subgraph_size is not None:
            if _S <= max_subgraph_size:
                nodes_in_subgraphs.append(subset)
            else:
                need_sampling = True

        elif subgraph_size is not None:
            if _S == subgraph_size:
                nodes_in_subgraphs.append(subset)
            elif _S < subgraph_size:
                continue
            elif _S > subgraph_size:
                need_sampling = True

        if need_sampling:
            _size_to_sample = subgraph_size or max_subgraph_size
            # Sample nodes of subgraph_size from subset:
            mask = torch.ones(subset.size(0), dtype=torch.bool)
            mask[idx_of_start_in_subset] = False
            sub_subset = torch_choice(subset[mask], _size_to_sample - 1)  # -1 for start
            sub_subset = torch.cat([sub_subset, torch.tensor([n_idx], dtype=torch.long)])
            nodes_in_subgraphs.append(sub_subset)

        if len(nodes_in_subgraphs) == num_subgraphs:
            break

    if subgraph_size is not None:
        nodes_in_subgraphs = torch.stack(nodes_in_subgraphs)
        assert list(nodes_in_subgraphs.size()) == [num_subgraphs, subgraph_size]

    return nodes_in_subgraphs


def nx_rewired_balanced_tree(num_nodes, num_branch, height, rewiring_ratio, seed):
    # from https://frhyme.github.io/python-lib/random-tree-in-nx/
    bg = nx.balanced_tree(num_branch, height - 1)
    bg.remove_nodes_from(list(bg.nodes())[num_nodes:])
    diameter = nx.diameter(bg)
    print(f"depth: {1 + diameter / 2}, diameter: {diameter}")

    level_node = [[0], ]
    for i in tqdm(range(0, height - 1), desc="rbt.level_node_construction"):
        left = sum([num_branch ** j for j in range(0, i + 1)])
        right = sum([num_branch ** (j + 1) for j in range(0, i + 1)])
        level_node.append([k for k in range(left, right + 1)])

    for i in tqdm(range(1, len(level_node) - 1), desc="rbt.rewiring"):
        num_edges_level = len([e for e in bg.edges()
                               if e[0] in level_node[i] and e[1] in level_node[i + 1]])
        for _ in range(int(num_edges_level * rewiring_ratio)):
            edges = [e for e in bg.edges()
                     if e[0] in level_node[i] and e[1] in level_node[i + 1]]
            if len(edges) > 0:
                r_e = edges[np.random.randint(0, len(edges))]
                bg.remove_edge(r_e[0], r_e[1])
                bg.add_edge(np.random.choice(level_node[i]), r_e[1])
    return bg


def draw_graph_with_coloring(data: Data,
                             colors: torch.Tensor,
                             title: str):
    g = to_networkx(data, to_undirected=True)

    nodes = g.nodes()
    colors = colors.tolist()

    pos = nx.spring_layout(g, seed=0)  # or nx.shell_layout(g)
    nx.draw_networkx_edges(g, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=colors,
                           node_size=150, cmap=plt.cm.Set3)
    nx.draw_networkx_labels(g, pos, labels=dict(zip(nodes, colors)),
                            font_size=15)

    ax = plt.gca()
    ax.margins(0.20)
    plt.title(title)
    plt.axis("off")
    plt.show()


def run_and_draw_examples(edge_index, num_layers):
    edge_index = to_undirected(edge_index.long())
    data = Data(x=torch.ones(maybe_num_nodes(edge_index)).long(),
                edge_index=edge_index)

    sub_x = generate_random_k_hop_subgraph(data, num_subgraphs=20, subgraph_size=5)

    wl = WL4PatternNet(
        num_layers=num_layers, x_type_for_hists="color",
        clustering_name="KMeans", n_clusters=3,  # clustering & kwargs
        use_clustering_validation=True,
    )
    wl_rets = wl(sub_x, data.x, data.edge_index)
    colors, hists, clusters = wl_rets["colors"], wl_rets["hists"], wl_rets["clusters"]

    for i, (co, cl, hi) in enumerate(zip(colors, clusters, hists)):
        hist_cluster = WL4PatternConv.to_cluster(
            hi, clustering_name="KMeans", n_clusters=3)
        hist_cluster, indices = torch.sort(hist_cluster)

        print(f"{i + 1} steps", "-" * 10)
        if wl.x_type_for_hists == "color":
            print(co[sub_x[indices, :]])
        else:
            print(cl[sub_x[indices, :]])
        print(hist_cluster)

    for i, (co, cl) in enumerate(zip_longest(colors, clusters)):

        if cl is not None:
            draw_graph_with_coloring(data, cl, title=f"WL pattern-cluster: {i + 1} steps")

        if co is not None:
            draw_graph_with_coloring(data, co, title=f"WL color: {i + 1} steps")


def draw_subgraph_embeddings(edge_index, num_layers,
                             num_subgraphs=1500,
                             **kwargs):
    edge_index = to_undirected(edge_index.long())
    data = Data(x=torch.ones(maybe_num_nodes(edge_index)).long(),
                edge_index=edge_index)

    sub_x = generate_random_k_hop_subgraph(
        data, num_subgraphs=num_subgraphs, subgraph_size=None)

    wl = WL4PatternNet(
        num_layers=num_layers, x_type_for_hists="all",
        clustering_name="KMeans", n_clusters=num_layers,  # clustering & kwargs
    )
    wl_rets = wl(sub_x, data.x, data.edge_index)
    colors, clusters = wl_rets["colors"], wl_rets["clusters"]
    hists_colors, hists_clusters = wl_rets["hists_colors"], wl_rets["hists_clusters"]

    from visualize import plot_data_points_by_tsne
    hist_co_label_list, hist_cl_label_list = [], []

    for i, (hi_co, hi_cl) in enumerate(zip(tqdm(hists_colors, desc="WL4PatternConv.to_cluster"),
                                           hists_clusters)):
        hist_co_label_list.append(WL4PatternConv.to_cluster(hi_co, clustering_name="KMeans", n_clusters=2).view(-1, 1))
        hist_cl_label_list.append(WL4PatternConv.to_cluster(hi_cl, clustering_name="KMeans", n_clusters=2).view(-1, 1))
    hist_co_labels = torch.cat(hist_co_label_list, dim=1)  # [S, C]
    hist_cl_labels = torch.cat(hist_cl_label_list, dim=1)  # [S, C]

    for i, (hi_co, hi_cl) in enumerate(zip(tqdm(hists_colors, desc="plot_data_points_by_tsne"),
                                           hists_clusters)):
        plot_data_points_by_tsne(
            xs=hi_co.numpy(),
            ys=hist_cl_labels.numpy(),
            key=f"WL4S-{num_layers} / x from color-{i} / y from cluster",
            **kwargs,
        )
        plot_data_points_by_tsne(
            xs=hi_cl.numpy(),
            ys=hist_cl_labels.numpy(),
            key=f"WL4S-{num_layers} / x from cluster-{i} / y from cluster",
            **kwargs,
        )


if __name__ == '__main__':

    MODE = "draw_subgraph_embeddings"

    # _g = nx.dorogovtsev_goltsev_mendes_graph(3)
    # _g = nx.random_partition_graph([7, 7, 7, 7], 0.64, 0.1, seed=10)
    # _g = nx.lollipop_graph(3, 5)
    if MODE == "draw_examples":
        _g = nx.barabasi_albert_graph(50, 3)
        run_and_draw_examples(
            edge_index=from_networkx(_g).edge_index,
            num_layers=4,
        )
    elif MODE == "draw_subgraph_embeddings":
        SEED = 399
        from utils import make_deterministic_everything

        make_deterministic_everything(SEED)
        _g = nx.barabasi_albert_graph(10000, 5, SEED)
        draw_subgraph_embeddings(
            edge_index=from_networkx(_g).edge_index,
            num_layers=4,
            alpha=0.5, s=5,
        )
