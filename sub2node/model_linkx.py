from torch_geometric.typing import OptTensor, Adj
from typing import List

import math

import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Parameter, Identity, Linear
import torch.nn.functional as F

from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import inits
# from torch_geometric.nn.models import MLP
from torch_geometric.nn.conv import MessagePassing


class MLP(torch.nn.Module):
    r"""A multi-layer perception (MLP) model.
    Args:
        channel_list (List[int]): List of input, intermediate and output
            channels. :obj:`len(channel_list) - 1` denotes the number of layers
            of the MLP.
        dropout (float, optional): Dropout probability of each hidden
            embedding. (default: :obj:`0.`)
        batch_norm (bool, optional): If set to :obj:`False`, will not make use
            of batch normalization. (default: :obj:`True`)
        relu_first (bool, optional): If set to :obj:`True`, ReLU activation is
            applied before batch normalization. (default: :obj:`False`)
    """
    def __init__(self, channel_list: List[int], dropout: float = 0.,
                 batch_norm: bool = True, relu_first: bool = False):
        super().__init__()
        assert len(channel_list) >= 2
        self.channel_list = channel_list
        self.dropout = dropout
        self.relu_first = relu_first

        self.lins = torch.nn.ModuleList()
        for dims in zip(channel_list[:-1], channel_list[1:]):
            self.lins.append(Linear(*dims))

        self.norms = torch.nn.ModuleList()
        for dim in zip(channel_list[1:-1]):
            self.norms.append(BatchNorm1d(dim) if batch_norm else Identity())

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """"""
        x = self.lins[0](x)
        for lin, norm in zip(self.lins[1:], self.norms):
            if self.relu_first:
                x = x.relu_()
            x = norm(x)
            if not self.relu_first:
                x = x.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin.forward(x)
        return x

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.channel_list)[1:-1]})'


class SparseLinear(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        inits.kaiming_uniform(self.weight, fan=self.in_channels,
                              a=math.sqrt(5))
        inits.uniform(self.in_channels, self.bias)

    def forward(self, edge_index: Adj,
                edge_weight: OptTensor = None,
                mask_idx: int = None) -> Tensor:
        # Mask untrained parameters.
        if mask_idx is not None:
            self.weight.data[mask_idx:, :] = 0.

        # propagate_type: (weight: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, weight=self.weight,
                             edge_weight=edge_weight, size=None)
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, weight_j: Tensor, edge_weight: OptTensor) -> Tensor:
        if edge_weight is None:
            return weight_j
        else:
            return edge_weight.view(-1, 1) * weight_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              weight: Tensor) -> Tensor:
        return matmul(adj_t, weight, reduce=self.aggr)


class InductiveLINKX(torch.nn.Module):
    r"""The inductive version of the LINKX model from the `"Large Scale
    Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple
    Methods" <https://arxiv.org/abs/2110.14446>`_ paper

    .. math::
        \mathbf{H}_{\mathbf{A}} &= \textrm{MLP}_{\mathbf{A}}(\mathbf{A})

        \mathbf{H}_{\mathbf{X}} &= \textrm{MLP}_{\mathbf{X}}(\mathbf{X})

        \mathbf{Y} &= \textrm{MLP}_{f} \left( \sigma \left( \mathbf{W}
        [\mathbf{H}_{\mathbf{A}}, \mathbf{H}_{\mathbf{X}}] +
        \mathbf{H}_{\mathbf{A}} + \mathbf{H}_{\mathbf{X}} \right) \right)

    .. note::

        For an example of using LINKX, see `examples/linkx.py <https://
        github.com/pyg-team/pytorch_geometric/blob/master/examples/linkx.py>`_.

    Args:
        num_nodes (int): The number of nodes in the graph.
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        num_layers (int): Number of layers of :math:`\textrm{MLP}_{f}`.
        num_edge_layers (int): Number of layers of
            :math:`\textrm{MLP}_{\mathbf{A}}`. (default: :obj:`1`)
        num_node_layers (int): Number of layers of
            :math:`\textrm{MLP}_{\mathbf{X}}`. (default: :obj:`1`)
        dropout (float, optional): Dropout probability of each hidden
            embedding. (default: :obj:`0.`)
    """
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, num_edge_layers: int = 1,
                 num_node_layers: int = 1, dropout: float = 0.,
                 num_train_nodes: int = None):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_train_nodes = num_train_nodes or num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_layers = num_edge_layers

        self.edge_lin = SparseLinear(num_nodes, hidden_channels)
        if self.num_edge_layers > 1:
            self.edge_norm = BatchNorm1d(hidden_channels)
            channels = [hidden_channels] * num_edge_layers
            self.edge_mlp = MLP(channels, dropout=dropout, relu_first=True)

        channels = [in_channels] + [hidden_channels] * num_node_layers
        self.node_mlp = MLP(channels, dropout=dropout, relu_first=True)

        self.cat_lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.cat_lin2 = torch.nn.Linear(hidden_channels, hidden_channels)

        channels = [hidden_channels] * num_layers + [out_channels]
        self.final_mlp = MLP(channels, dropout, relu_first=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.edge_lin.reset_parameters()
        if self.num_edge_layers > 1:
            self.edge_norm.reset_parameters()
            self.edge_mlp.reset_parameters()
        self.node_mlp.reset_parameters()
        self.cat_lin1.reset_parameters()
        self.cat_lin2.reset_parameters()
        self.final_mlp.reset_parameters()

    def forward(self, x: OptTensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        # If x is larger than trained nodes, mask adj-params beyond the trained idx.
        out = self.edge_lin(
            edge_index, edge_weight,
            mask_idx=self.num_train_nodes if x.size(0) > self.num_train_nodes else None,
        )

        if x.size(0) < out.size(0):
            out = out[:x.size(0), :]

        if self.num_edge_layers > 1:
            out = out.relu_()
            out = self.edge_norm(out)
            out = self.edge_mlp(out)

        out = out + self.cat_lin1(out)

        if x is not None:
            x = self.node_mlp(x)
            out += x
            out += self.cat_lin2(x)

        return self.final_mlp(out.relu_())

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_nodes={self.num_nodes}, '
                f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels})')


if __name__ == '__main__':
    from torch_geometric.utils import to_undirected, add_self_loops, sort_edge_index
    import torch_geometric

    torch_geometric.seed_everything(100)

    N = 8

    _ei_N2 = torch.arange(N + 2).view(2, (N + 2) // 2)
    _ei_N2 = to_undirected(_ei_N2)
    _ei_N2, _ = add_self_loops(_ei_N2)
    _ei_N2 = sort_edge_index(_ei_N2)

    _ei_N = _ei_N2[:, _ei_N2[0] < N]
    _ei_N = _ei_N[:, _ei_N[1] < N]

    # _x_N2 = torch.randn((N + 2) * 32).view(N + 2, -1)
    _x_N2 = torch.zeros((N + 2) * 32).view(N + 2, -1)
    _x_N = _x_N2[:N, :]

    _linkx = InductiveLINKX(N + 2, 32, 4, 3, 1, num_train_nodes=N)

    _out = _linkx(_x_N, _ei_N)
    print("Transductive X, E, _x_N, _ei_N, out", _out.size())
    print(_out)
    print("-" * 10)

    _out = _linkx(_x_N2, _ei_N)
    print("Transductive E, _x_N2, _ei_N, out", _out.size())
    print(_out)
    print("-" * 10)

    _out = _linkx(_x_N2, _ei_N2)
    print("Inductive, _x_N2, _ei_N2, out", _out.size())
    print(_out)
