from itertools import zip_longest

import matplotlib.pyplot as plt
import networkx as nx
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cluster import KMeans
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import WLConv
from torch_geometric.typing import Adj
from torch_geometric.utils import to_undirected, to_networkx, from_networkx
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor

torch.manual_seed(42)


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
        assert clustering_name in ["KMeans"]
        clustering = eval(clustering_name)(**kwargs).fit(pattern_x)
        return torch.from_numpy(clustering.labels_).long()


class WL4PatternNet(torch.nn.Module):

    def __init__(self, num_layers, clustering_name="KMeans"):
        super().__init__()
        self.convs = torch.nn.ModuleList([WL4PatternConv() for _ in range(num_layers)])

        self.clustering_name = clustering_name
        self.cluster_kwargs = {
            "KMeans": {"n_clusters": 3, "pattern_preprocessor": None},
        }[self.clustering_name]

    def forward(self, x, edge_index, batch=None, hist_norm=True):

        colors, hists, clusters = [], [], []

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            colors.append(x)
            hists.append(conv.histogram(x, batch, norm=hist_norm))

            clusters.append(conv.color_pattern_cluster(x, **self.cluster_kwargs))

        return colors, hists, clusters


def draw_graph_with_coloring(data: Data,
                             colors: torch.Tensor,
                             title: str):
    g = to_networkx(data, to_undirected=True)

    nodes = g.nodes()
    colors = colors.tolist()

    pos = nx.spring_layout(g, seed=0)  # or nx.shell_layout(g)
    nx.draw_networkx_edges(g, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=colors,
                           node_size=200, cmap=plt.cm.Set3)
    nx.draw_networkx_labels(g, pos, labels=dict(zip(nodes, colors)),
                            font_size=15)

    ax = plt.gca()
    ax.margins(0.20)
    plt.title(title)
    plt.axis("off")
    plt.show()


def draw_examples(edge_index, num_layers):
    edge_index = to_undirected(edge_index.long())
    data = Data(x=torch.ones(maybe_num_nodes(edge_index)).long(),
                edge_index=edge_index)

    wl = WL4PatternNet(num_layers=num_layers)
    colors, hists, clusters = wl(data.x, data.edge_index, data.batch)

    for i, (co, cl) in enumerate(zip_longest(colors, clusters)):

        if cl is not None:
            draw_graph_with_coloring(data, cl, title=f"WL pattern-cluster: {i + 1} steps")

        if co is not None:
            draw_graph_with_coloring(data, co, title=f"WL color: {i + 1} steps")


if __name__ == '__main__':
    # _g = nx.dorogovtsev_goltsev_mendes_graph(3)
    # _g = nx.random_partition_graph([7, 7, 7, 7], 0.64, 0.1, seed=10)
    # _g = nx.lollipop_graph(3, 5)
    _g = nx.barabasi_albert_graph(50, 3)
    draw_examples(
        edge_index=from_networkx(_g).edge_index,
        num_layers=4,
    )
