from copy import deepcopy
from typing import Union, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from termcolor import cprint
from torch import Tensor
from torch_geometric.nn.glob import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GlobalAttention, GCNConv, SAGEConv, GATConv
from torch_geometric.utils import dropout_adj
from torch_scatter import scatter_add

from utils import softmax_half, act, merge_dict_by_keys


class EPSILON(object):

    def __add__(self, other):
        eps = 1e-7 if other.dtype == torch.float16 else 1e-15
        return other + eps

    def __radd__(self, other):
        return self.__add__(other)


class MyLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)

    def forward(self, x, *args, **kwargs):
        return super(MyLinear, self).forward(x)


class MyGATConv(GATConv):

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, edge_dim: Optional[int] = None,
                 fill_value: Union[float, Tensor, str] = 'mean',
                 bias: bool = True, **kwargs):
        if concat:
            assert out_channels % heads == 0
        out_channels = out_channels // heads
        super().__init__(in_channels, out_channels, heads, concat, negative_slope,
                         dropout, add_self_loops, edge_dim, fill_value, bias, **kwargs)

    def forward(self, x, edge_index, edge_attr, size=None, return_attention_weights=None):
        if self.edge_dim is None and edge_attr is not None:
            edge_attr = None
        return super().forward(x, edge_index, edge_attr, size, return_attention_weights)


class PositionalEncoding(nn.Module):

    def __init__(self, max_len, num_channels, dropout=0.0):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.num_channels = num_channels
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, num_channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_channels, 2).float() * (-math.log(10000.0) / num_channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1).squeeze()
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [N, F] or [B, N_max, F]
        x = x + self.pe[:x.size(-2), :]
        return self.dropout(x)

    def __repr__(self):
        return "{}(max_len={}, num_channels={}, dropout={})".format(
            self.__class__.__name__, self.max_len, self.num_channels, self.dropout.p,
        )


class BilinearWith1d(nn.Bilinear):

    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super().__init__(in1_features, in2_features, out_features, bias)
        # weight: [o, i1, i2], bias: [o,]

    def forward(self, x1, x2):

        x1_dim = x1.squeeze().dim()

        if x1_dim == 1:  # single-batch
            # x1 [F,] * weight [O, F, S] * x2 [N, S] -> [N, O]
            x1, x2 = x1.squeeze(), x2.squeeze()
            x = torch.einsum("f,ofs,ns->no", x1, self.weight, x2)

        elif x1_dim == 2:  # multi-batch
            # x1 [B, F] * weight [O, F, S] * x2 [B, N, S] -> [B, N, O]
            x = torch.einsum("bf,ofs,bns->bno", x1, self.weight, x2)

        else:
            raise ValueError("Wrong x1 shape: {}".format(x1.size()))

        if self.bias is not None:
            x += self.bias
        return x


class MLP(nn.Module):

    def __init__(self, num_layers, in_channels, hidden_channels, out_channels, activation,
                 use_bn=False, dropout=0.0, activate_last=False):
        super().__init__()
        self.num_layers = num_layers
        self.in_channels, self.hidden_channels, self.out_channels = in_channels, hidden_channels, out_channels
        self.activation, self.use_bn, self.dropout = activation, use_bn, dropout
        self.activate_last = activate_last
        layers = nn.ModuleList()

        for i in range(num_layers - 1):
            if i == 0:
                layers.append(nn.Linear(in_channels, hidden_channels))
            else:
                layers.append(nn.Linear(hidden_channels, hidden_channels))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(Activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        if num_layers != 1:
            layers.append(nn.Linear(hidden_channels, out_channels))
        else:  # single-layer
            layers.append(nn.Linear(in_channels, out_channels))

        if self.activate_last:
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_channels))
            layers.append(Activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        self.fc = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        return self.fc(x)

    def __repr__(self):
        if self.num_layers > 1:
            return "{}(L={}, I={}, H={}, O={}, act={}, act_last={}, bn={}, do={})".format(
                self.__class__.__name__, self.num_layers, self.in_channels, self.hidden_channels, self.out_channels,
                self.activation, self.activate_last, self.use_bn, self.dropout,
            )
        else:
            return "{}(L={}, I={}, O={}, act={}, act_last={}, bn={}, do={})".format(
                self.__class__.__name__, self.num_layers, self.in_channels, self.out_channels,
                self.activation, self.activate_last, self.use_bn, self.dropout,
            )

    def layer_repr(self):
        """
        :return: e.g., '64->64'
        """
        hidden_layers = [self.hidden_channels] * (self.num_layers - 1) if self.num_layers >= 2 else []
        layers = [self.in_channels] + hidden_layers + [self.out_channels]
        return "->".join(str(l) for l in layers)


class Activation(nn.Module):

    def __init__(self, activation_name):
        super().__init__()
        if activation_name is None:
            self.a = nn.Identity()
        elif activation_name == "relu":
            self.a = nn.ReLU()
        elif activation_name == "elu":
            self.a = nn.ELU()
        elif activation_name == "leaky_relu":
            self.a = nn.LeakyReLU()
        elif activation_name == "sigmoid":
            self.a = nn.Sigmoid()
        elif activation_name == "tanh":
            self.a = nn.Tanh()
        else:
            raise ValueError(f"Wrong activation name: {activation_name}")

    def forward(self, tensor):
        return self.a(tensor)

    def __repr__(self):
        return self.a.__repr__()


def get_gnn_conv_and_kwargs(gnn_name, **kwargs):
    gkw = {}
    if gnn_name == "GCNConv":
        gnn_cls = GCNConv
        gkw = merge_dict_by_keys(gkw, kwargs, ["add_self_loops"])
    elif gnn_name == "SAGEConv":
        gnn_cls = SAGEConv
    elif gnn_name == "GATConv":
        gnn_cls = MyGATConv
        gkw = merge_dict_by_keys(
            gkw, kwargs,
            ["add_self_loops", "heads", "dropout", "edge_dim"],
        )
    elif gnn_name == "Linear":
        gnn_cls = MyLinear
    else:
        raise ValueError(f"Wrong gnn conv name: {gnn_name}")
    return gnn_cls, gkw


class GraphEncoder(nn.Module):

    def __init__(self, layer_name, num_layers, in_channels, hidden_channels, out_channels,
                 activation="relu", use_bn=False, use_skip=False,
                 dropout_channels=0.0, dropout_edges=0.0,
                 activate_last=True, **kwargs):
        super().__init__()

        self.layer_name, self.num_layers = layer_name, num_layers
        self.in_channels, self.hidden_channels, self.out_channels = in_channels, hidden_channels, out_channels
        self.activation = activation
        self.use_bn, self.use_skip = use_bn, use_skip
        self.dropout_channels = dropout_channels
        self.dropout_edges = dropout_edges
        self.activate_last = activate_last

        self._gnn_kwargs = {}
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList() if self.use_bn else []
        self.skips = torch.nn.ModuleList() if self.use_skip else []
        self.build(**kwargs)

    def build(self, **kwargs):
        gnn, self._gnn_kwargs = get_gnn_conv_and_kwargs(self.layer_name, **kwargs)
        for conv_id in range(self.num_layers):
            _in_channels = self.in_channels if conv_id == 0 else self.hidden_channels
            _out_channels = self.hidden_channels if (conv_id != self.num_layers - 1) else self.out_channels
            self.convs.append(gnn(_in_channels, _out_channels, **self._gnn_kwargs))
            if conv_id != self.num_layers - 1 or self.activate_last:
                if self.use_bn:
                    self.bns.append(nn.BatchNorm1d(self.hidden_channels))
                if self.use_skip:
                    self.skips.append(nn.Linear(_in_channels, _out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        # Order references.
        #  https://d2l.ai/chapter_convolutional-modern/resnet.html#residual-blocks
        #  https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ppi.py#L30-L34
        #  https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/gnn.py#L69-L76
        for i, conv in enumerate(self.convs):
            x_before_layer = x
            _edge_index, _edge_attr = dropout_adj(
                edge_index, edge_attr,
                p=self.dropout_edges, training=self.training)
            x = conv(x, _edge_index, _edge_attr, **kwargs)
            if i != self.num_layers - 1 or self.activate_last:
                if self.use_bn:
                    x = self.bns[i](x)
                if self.use_skip:
                    x = x + self.skips[i](x_before_layer)
                x = act(x, self.activation)
                x = F.dropout(x, p=self.dropout_channels, training=self.training)
        return x

    def __gnn_kwargs_repr__(self):
        if len(self._gnn_kwargs) == 0:
            return ""
        else:
            return ", " + ", ".join([f"{k}={v}" for k, v in self._gnn_kwargs.items()])

    def __repr__(self):
        return "{}(conv={}, L={}, I={}, H={}, O={}, act={}, act_last={}, skip={}, bn={}{})".format(
            self.__class__.__name__, self.layer_name, self.num_layers,
            self.in_channels, self.hidden_channels, self.out_channels,
            self.activation, self.activate_last, self.use_skip, self.use_bn,
            self.__gnn_kwargs_repr__(),
        )


class EdgePredictor(nn.Module):

    def __init__(self, predictor_type, num_layers, hidden_channels, out_channels,
                 activation="relu", out_activation=None, use_bn=False, dropout_channels=0.0):
        super().__init__()
        assert predictor_type in ["DotProduct", "Concat", "HadamardProduct"]
        if predictor_type == "DotProduct":
            assert out_channels == 1
        self.predictor_type = predictor_type
        in_channels = 2 * hidden_channels if predictor_type == "Concat" else hidden_channels
        self.mlp = MLP(
            num_layers=num_layers,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            activation=activation,
            use_bn=use_bn,
            dropout=dropout_channels,
            activate_last=False,
        )
        self.out_activation = Activation(out_activation)

    def forward(self, x=None, edge_index=None, x_i=None, x_j=None):
        if x is not None and edge_index is not None:
            x_i, x_j = x[edge_index[0]], x[edge_index[1]]  # [E, F]
        assert x_i is not None and x_j is not None
        if self.predictor_type == "DotProduct":
            x_i, x_j = self.mlp(x_i), self.mlp(x_j)
            logits = torch.einsum("ef,ef->e", x_i, x_j)
        elif self.predictor_type == "Concat":
            e = torch.cat([x_i, x_j], dim=-1)  # [E, 2F]
            logits = self.mlp(e)
        elif self.predictor_type == "HadamardProduct":
            e = x_i * x_j  # [E, F]
            logits = self.mlp(e)
        else:
            raise ValueError
        return self.out_activation(logits)

    def __repr__(self):
        return "{}({}, mlp={}, out={})".format(
            self.__class__.__name__, self.predictor_type,
            self.mlp.layer_repr(), self.out_activation,
        )


class Readout(nn.Module):

    def __init__(self, readout_types,
                 use_in_mlp, use_out_linear=False,
                 num_in_layers=None, hidden_channels=None, out_channels=None,
                 activation="relu", use_bn=False, dropout_channels=0.0, **kwargs):
        super().__init__()

        self.readout_types = readout_types  # e.g., mean, max, sum, mean-max, ...,
        self.use_in_mlp = use_in_mlp
        self.use_out_linear = use_out_linear

        self.num_in_layers = num_in_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.activation = activation
        self.use_bn = use_bn
        self.dropout_channels = dropout_channels

        self.in_mlp, self.out_linear = None, None

        if self.use_in_mlp:
            assert num_in_layers is not None
            assert hidden_channels is not None
            self.in_mlp = self.build_in_mlp(**kwargs)  # [N, F] -> [N, F]

        if self.use_out_linear:
            assert hidden_channels is not None
            assert out_channels is not None
            num_readout_types = len(self.readout_types.split("-"))
            self.out_linear = nn.Linear(
                num_readout_types * hidden_channels,
                out_channels,
            )

    def build_in_mlp(self, **kwargs):
        kw = dict(
            num_layers=self.num_in_layers,
            in_channels=self.hidden_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            activation=self.activation,
            use_bn=self.use_bn,
            dropout=self.dropout_channels,
            activate_last=True,  # important
        )
        kw.update(**kwargs)
        return MLP(**kw)

    @staticmethod
    def aggregate(aggr_types, x, batch=None):
        B = int(batch.max().item() + 1) if batch is not None else 1
        o_list = []
        if "mean" in aggr_types:
            o_list.append(torch.mean(x, dim=0) if batch is None else
                          global_mean_pool(x, batch, B))
        if "max" in aggr_types:
            is_half = x.dtype == torch.half  # global_max_pool does not support half precision.
            x = x.float() if is_half else x
            o_list.append(torch.max(x, dim=0).values if batch is None else
                          global_max_pool(x, batch, B).half() if is_half else
                          global_max_pool(x, batch, B))
        if "sum" in aggr_types:
            o_list.append(torch.sum(x, dim=0) if batch is None else
                          global_add_pool(x, batch, B))
        return torch.cat(o_list, dim=-1)  # [F * #type] or [B, F * #type]

    def forward(self, x, batch=None, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        if self.use_in_mlp:
            x = self.in_mlp(x)
        z = self.aggregate(self.readout_types, x, batch)  # [F * #type] or [B, F * #type]

        B = int(batch.max().item() + 1) if batch is not None else 1
        if self.use_out_linear:
            out_logits = self.out_linear(z).view(B, -1)
        else:
            out_logits = None

        return z.view(B, -1), out_logits

    def __repr__(self):
        attr_reprs = [self.readout_types]
        if self.use_in_mlp:
            attr_reprs.append(f"in_mlp={self.in_mlp.layer_repr()}")
        if self.use_out_linear:
            attr_reprs.append(f"out_linear={self.out_linear.in_features}->{self.out_linear.out_features}")
        return "{}({})".format(self.__class__.__name__, ", ".join(attr_reprs))


class DeepSets(nn.Module):

    def __init__(self, encoder, decoder, aggr=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.aggr = aggr or "sum"  # e.g., mean, max, sum, mean-max, ...,
        self.att = None
        if aggr == "attention":
            self.att = GlobalAttention(nn.Linear(encoder.out_channels, 1))

    def aggregate(self, x, batch):
        if self.att is None:
            return Readout.aggregate(self.aggr, x, batch)
        else:
            return self.att(x, batch)

    def forward(self, x, batch=None, *args, **kwargs):
        x = self.encoder(x)
        x = self.aggregate(x, batch)
        x = self.decoder(x)
        return x

    def extra_repr(self) -> str:
        return f"(aggr): {self.aggr}"


class VersatileEmbedding(nn.Module):

    def __init__(self, embedding_type, num_entities, num_channels,
                 pretrained_embedding=None, freeze_pretrained=False):
        super().__init__()

        self.embedding_type = embedding_type
        self.num_entities = num_entities
        self.num_channels = num_channels
        if not isinstance(num_channels, int) or num_channels <= 0:
            assert embedding_type == "UseRawFeature"

        if self.embedding_type == "Embedding":
            self.embedding = nn.Embedding(self.num_entities, self.num_channels)
        elif self.embedding_type == "Random":
            self.embedding = nn.Embedding(self.num_entities, self.num_channels)
            self.embedding.weight.requires_grad = False
        elif self.embedding_type == "UseRawFeature":
            self.embedding = None
        elif self.embedding_type == "Pretrained":
            assert pretrained_embedding is not None
            N, C = pretrained_embedding.size()
            assert self.num_entities == N
            assert self.num_channels == C
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, freeze=freeze_pretrained)
        else:
            raise ValueError(f"Wrong global_channel_type: {self.embedding_type}")

    def forward(self, indices_or_features):
        if self.embedding is not None:
            return self.embedding(indices_or_features.squeeze())
        else:
            return indices_or_features

    def __repr__(self):
        return '{}({}, {}, type={})'.format(
            self.__class__.__name__,
            self.num_entities,
            self.num_channels,
            self.embedding_type,
        )


class BiConv(nn.Module):

    def __init__(self, base_conv, reset_at_init=True):
        super().__init__()
        self.conv = deepcopy(base_conv)
        self.rev_conv = base_conv
        if reset_at_init:
            self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.rev_conv.reset_parameters()

    def forward(self, x, edge_index, *args, **kwargs):
        rev_edge_index = edge_index[[1, 0]]
        fwd_x = self.conv(x, edge_index, *args, **kwargs)
        rev_x = self.rev_conv(x, rev_edge_index, *args, **kwargs)
        return torch.cat([fwd_x, rev_x], dim=1)

    def __repr__(self):
        return "Bi{}".format(self.conv.__repr__())


class GlobalAttentionHalf(GlobalAttention):
    r"""GlobalAttention that supports torch.half tensors.
        See torch_geometric.nn.GlobalAttention for more details."""

    def __init__(self, gate_nn, nn=None):
        super(GlobalAttentionHalf, self).__init__(gate_nn, nn)

    def forward(self, x, batch, size=None):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax_half(gate, batch, num_nodes=size)  # A substitute for softmax
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out


if __name__ == '__main__':

    MODE = "DeepSets"

    from pytorch_lightning import seed_everything

    seed_everything(42)

    if MODE == "BilinearWith1d":
        _bilinear = BilinearWith1d(in1_features=3, in2_features=6, out_features=7)
        _x1 = torch.randn((1, 3))
        _x2 = torch.randn((23, 6))
        print(_bilinear(_x1, _x2).size())  # [23, 7]
        _x1 = torch.randn((5, 3))
        _x2 = torch.randn((5, 23, 6))
        print(_bilinear(_x1, _x2).size())  # [5, 23, 7]

    elif MODE == "Readout":
        _ro = Readout(readout_types="sum", use_in_mlp=True, hidden_channels=64, num_in_layers=2)
        _x = torch.ones(10 * 64).view(10, 64)
        _batch = torch.zeros(10).long()
        _batch[:4] = 1
        cprint("-- sum w/ batch", "red")
        _z, _ = _ro(_x, _batch)
        print(_ro)
        print("_z", _z.size())  # [2, 64]

        _ro = Readout(readout_types="sum", use_in_mlp=True, use_out_linear=True,
                      num_in_layers=2, hidden_channels=64, out_channels=3)
        cprint("-- sum w/ batch", "red")
        _z, _logits = _ro(_x, _batch)
        print(_ro)
        print("_z", _z.size())  # [2, 64]
        print("_logits", _logits.size())  # [2, 3]

        _ro = Readout(readout_types="mean-sum", use_in_mlp=True, use_out_linear=True,
                      num_in_layers=2, hidden_channels=64, out_channels=3)
        cprint("-- mean-sum w/ batch", "red")
        _z, _logits = _ro(_x, _batch)
        print(_ro)
        print("_z", _z.size())  # [2, 128]
        print("_logits", _logits.size())  # [2, 3]

    elif MODE == "DeepSets":
        _x = torch.ones(10 * 64).view(10, 64)
        _batch = torch.zeros(10).long()
        _batch[4:] = 1
        _ds = DeepSets(
            encoder=MLP(2, 64, 17, 17, "relu"),
            decoder=MLP(2, 17, 17, 10, "relu"),
            aggr="attention",
        )
        print(_ds)
        print(_ds(_x, _batch).size())

    elif MODE == "GraphEncoder":
        enc = GraphEncoder(
            layer_name="SAGEConv", num_layers=3, in_channels=32, hidden_channels=64, out_channels=128,
            activation="relu", use_bn=False, use_skip=False, dropout_channels=0.0, dropout_edges=0.2,
            activate_last=True,
        )
        cprint(enc, "red")
        _x = torch.ones(10 * 32).view(10, -1)
        _ei = torch.randint(0, 10, [2, 10])
        print(enc(_x, _ei).size())

        enc = GraphEncoder(
            layer_name="GATConv", num_layers=3, in_channels=32, hidden_channels=64, out_channels=64,
            activation="elu", use_bn=True, use_skip=True, dropout_channels=0.0,
            activate_last=True, add_self_loops=False, heads=8,
        )
        cprint(enc, "red")
        print(enc(_x, _ei).size())

    elif MODE == "VersatileEmbedding":
        _pte = torch.arange(11 * 32).view(11, 32).float()
        de = VersatileEmbedding(embedding_type="Pretrained", num_entities=11, num_channels=32,
                                pretrained_embedding=_pte, freeze_pretrained=False)
        cprint(de, "red")
        print("Embedding: {} +- {}".format(
            de.embedding.weight.mean().item(),
            de.embedding.weight.std().item(),
        ))

        _x = torch.arange(11)
        print(de(_x).size())
