from itertools import zip_longest
from typing import Optional, List, Dict, Any

import matplotlib.pyplot as plt
import networkx as nx
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cluster import KMeans
from torch import Tensor
from torch_geometric.data import Data
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
        data.y = data.y[:, self.y_idx]
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}({self.y_idx})'


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

        self_patterns, neighbor_patterns = [], []
        for c in color.tolist():
            sp, *np = color_to_pattern[c]
            self_patterns.append([sp])
            neighbor_patterns.append(np)

        if outtype == "bow":
            vectorizer = CountVectorizer(preprocessor=lambda _: _, tokenizer=lambda _: _)
            pattern_transformed = vectorizer.fit_transform(self_patterns + neighbor_patterns)
            # print(vectorizer.get_feature_names_out())
            N = color.size(0)
            pattern_vec = pattern_transformed.toarray()
            sp_vec, np_vec = pattern_vec[:N, :], pattern_vec[N:, :]
            if preprocessor is not None:
                sp_vec = eval(preprocessor)().fit_transform(sp_vec)
                np_vec = eval(preprocessor)().fit_transform(np_vec)

            return torch.cat([torch.from_numpy(sp_vec),
                              torch.from_numpy(np_vec)], dim=-1)

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
                           x: Tensor,
                           norm: bool = False, num_colors=None) -> Tensor:
        S = len(subgraph_nodes)
        sizes = torch.tensor([s.size(0) for s in subgraph_nodes], dtype=torch.long)
        batch = torch.arange(S).repeat_interleave(sizes)
        sub_x = x[torch.cat(subgraph_nodes)]
        sub_hist = self.histogram(sub_x, batch, norm=norm, num_colors=num_colors)
        return sub_hist

    @staticmethod
    def to_cluster(x: Tensor, clustering_name="KMeans", **kwargs) -> Tensor:
        assert clustering_name in ["KMeans"]
        clustering = eval(clustering_name)(**kwargs).fit(x.numpy())
        return torch.from_numpy(clustering.labels_).long()


class WL4PatternNet(torch.nn.Module):

    def __init__(self, num_layers, clustering_name="KMeans", x_type_for_hists="color", **kwargs):
        super().__init__()
        self.convs = torch.nn.ModuleList([WL4PatternConv() for _ in range(num_layers)])

        self.clustering_name = clustering_name
        self.cluster_kwargs: Dict[str, Any] = {
            "KMeans": {"n_clusters": 3, "pattern_preprocessor": None},
        }[clustering_name]
        self.cluster_kwargs.update(kwargs)

        self.x_type_for_hists = x_type_for_hists
        assert x_type_for_hists in ["color", "cluster"]

    def forward(self, sub_x, x, edge_index, hist_norm=True):
        colors, hists, clusters = [], [], []

        for i, conv in enumerate(self.convs):
            conv: WL4PatternConv
            x = conv(x, edge_index)

            colors.append(x)
            clusters.append(conv.color_pattern_cluster(x, **self.cluster_kwargs))
            if self.x_type_for_hists == "color":
                hists.append(conv.subgraph_histogram(list(sub_x), colors[-1], norm=hist_norm,
                                                     num_colors=len(conv.hashmap)))
            else:  # cluster
                hists.append(conv.subgraph_histogram(list(sub_x), clusters[-1], norm=hist_norm,
                                                     num_colors=self.cluster_kwargs["n_clusters"]))

        return colors, hists, clusters


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


def generate_random_subgraph_by_k_hop(global_data: Data, num_subgraphs, subgraph_size, k=1):
    N, E = global_data.num_nodes, global_data.num_edges
    nodes_in_subgraphs = []
    start = torch.randint(0, N, (num_subgraphs * 2,), dtype=torch.long).flatten()
    for n_idx in tqdm(start, desc="generate_random_subgraph_by_k_hop", total=num_subgraphs * 2):
        subset, _, idx_of_start_in_subset, _ = k_hop_subgraph([n_idx], k, global_data.edge_index, num_nodes=N)
        _S = subset.size(0)
        if _S < subgraph_size:
            continue
        elif _S > subgraph_size:
            # Sample nodes of subgraph_size from subset:
            mask = torch.ones(subset.size(0), dtype=torch.bool)
            mask[idx_of_start_in_subset] = False
            sub_subset = torch_choice(subset[mask], subgraph_size - 1)  # -1 for start
            sub_subset = torch.cat([sub_subset, torch.tensor([n_idx], dtype=torch.long)])
            nodes_in_subgraphs.append(sub_subset)
        else:
            nodes_in_subgraphs.append(subset)

        if len(nodes_in_subgraphs) == num_subgraphs:
            break

    nodes_in_subgraphs = torch.stack(nodes_in_subgraphs)
    assert list(nodes_in_subgraphs.size()) == [num_subgraphs, subgraph_size]
    return nodes_in_subgraphs


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

    sub_x = generate_random_subgraph_by_k_hop(data, num_subgraphs=20, subgraph_size=5)

    wl = WL4PatternNet(
        num_layers=num_layers, x_type_for_hists="color",
        clustering_name="KMeans", n_clusters=3,  # clustering & kwargs
    )
    colors, hists, clusters = wl(sub_x, data.x, data.edge_index)

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


if __name__ == '__main__':

    MODE = "draw_examples"

    # _g = nx.dorogovtsev_goltsev_mendes_graph(3)
    # _g = nx.random_partition_graph([7, 7, 7, 7], 0.64, 0.1, seed=10)
    # _g = nx.lollipop_graph(3, 5)
    _g = nx.barabasi_albert_graph(50, 3)
    if MODE == "draw_examples":
        run_and_draw_examples(
            edge_index=from_networkx(_g).edge_index,
            num_layers=4,
        )
