import inspect
from copy import deepcopy
from typing import Union, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from termcolor import cprint
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn.glob import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GlobalAttention, GCNConv, SAGEConv, GATConv, FAConv, GCN2Conv, WLConv, GraphNorm
from torch_geometric.typing import OptTensor, Adj
from torch_scatter import scatter_add
from torch_sparse import SparseTensor

from utils import softmax_half, act, merge_dict_by_keys, dropout_adj_st


class EPSILON(object):

    def __add__(self, other):
        eps = 1e-7 if other.dtype == torch.float16 else 1e-15
        return other + eps

    def __radd__(self, other):
        return self.__add__(other)


class MyLinear(Linear):

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)

    def forward(self, x, *args, **kwargs):
        return super(MyLinear, self).forward(x)


def clear_edge_attr(self: nn.Module, edge_index, edge_weight, condition_str):
    if edge_weight is not None or (isinstance(edge_index, SparseTensor) and edge_index.has_value()):
        cprint(f"Warning: {self.__class__.__base__.__name__} cannot use edge_weight with {condition_str}, "
               f"but {self.__class__.__name__} overwrites edge_weight to None.", "red")
    edge_weight = None
    if isinstance(edge_index, SparseTensor):
        edge_index: SparseTensor
        edge_index.set_value_(None)
    return edge_index, edge_weight


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
        if self.edge_dim is None:
            edge_index, edge_attr = clear_edge_attr(self, edge_index, edge_attr, "edge_dim=None")
        return super().forward(x, edge_index, edge_attr, size, return_attention_weights)


class MyFAConv(FAConv):

    def __init__(self, in_channels: int, out_channels: int,
                 eps: float = 0.1, dropout: float = 0.0,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):
        super().__init__(in_channels, eps, dropout, cached, add_self_loops, normalize, **kwargs)
        self.out = Linear(in_channels, out_channels, bias=False) if in_channels != out_channels else None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None,
                x_0: OptTensor = None, return_attention_weights=None):
        if self.normalize:
            edge_index, edge_weight = clear_edge_attr(self, edge_index, edge_weight, "normalize=True")
        x = super().forward(x=x, x_0=x_0, edge_index=edge_index, edge_weight=edge_weight,
                            return_attention_weights=return_attention_weights)
        return self.out(x) if self.out is not None else x

    def __repr__(self):
        if self.out is None:
            return super().__repr__()
        else:
            return "{}({}, {}, eps={})".format(self.__class__.__name__,
                                               self.channels, self.out.out_features, self.eps)


class MyGCN2Conv(GCN2Conv):
    """
    Examples are
        https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn2_cora.py
        https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn2_ppi.py
    Docs is
        https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCN2Conv
    """

    def __init__(self, in_channels: int, out_channels: int,
                 alpha: float, theta: float = None,
                 layer: int = None, shared_weights: bool = True,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):
        super().__init__(in_channels, alpha, theta, layer, shared_weights, cached,
                         add_self_loops, normalize, **kwargs)
        self.out = Linear(in_channels, out_channels, bias=False) if in_channels != out_channels else None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None,
                x_0: OptTensor = None) -> Tensor:
        x = super().forward(x=x, x_0=x_0, edge_index=edge_index, edge_weight=edge_weight)
        return self.out(x) if self.out is not None else x

    def __repr__(self):
        if self.out is None:
            return super().__repr__()
        else:
            return '{}({}, {}, alpha={}, beta={})'.format(
                self.__class__.__name__, self.channels, self.out.out_features, self.alpha, self.beta)

    @classmethod
    def __construct_init_kwargs__(cls, **kwargs):
        kwargs = merge_dict_by_keys({}, kwargs, inspect.getfullargspec(cls.__init__).args)
        if "theta" not in kwargs and "layer" in kwargs:
            del kwargs["layer"]
        return kwargs


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
                 use_bn=False, use_gn=False, dropout=0.0, activate_last=False):
        super().__init__()
        self.num_layers = num_layers
        self.in_channels, self.hidden_channels, self.out_channels = in_channels, hidden_channels, out_channels
        self.activation, self.use_bn, self.use_gn, self.dropout = activation, use_bn, use_gn, dropout
        self.activate_last = activate_last
        layers = nn.ModuleList()

        for i in range(num_layers - 1):
            if i == 0:
                layers.append(Linear(in_channels, hidden_channels))
            else:
                layers.append(Linear(hidden_channels, hidden_channels))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_channels))
            if use_gn:
                layers.append(GraphNorm(hidden_channels))
            layers.append(Activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        if num_layers != 1:
            layers.append(Linear(hidden_channels, out_channels))
        else:  # single-layer
            layers.append(Linear(in_channels, out_channels))

        if self.activate_last:
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_channels))
            if use_gn:
                layers.append(GraphNorm(hidden_channels))
            layers.append(Activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

        self.fc = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        return self.fc(x)

    def __repr__(self):
        if self.num_layers > 1:
            return "{}(L={}, I={}, H={}, O={}, act={}, act_last={}, bn={}, gn={}, do={})".format(
                self.__class__.__name__, self.num_layers, self.in_channels, self.hidden_channels, self.out_channels,
                self.activation, self.activate_last, self.use_bn, self.use_gn, self.dropout,
            )
        else:
            return "{}(L={}, I={}, O={}, act={}, act_last={}, bn={}, gn={}, do={})".format(
                self.__class__.__name__, self.num_layers, self.in_channels, self.out_channels,
                self.activation, self.activate_last, self.use_bn, self.use_gn, self.dropout,
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
    gnn_cls = {
        "GCNConv": GCNConv,
        "SAGEConv": SAGEConv,
        "GATConv": MyGATConv,
        "FAConv": MyFAConv,
        "GCN2Conv": MyGCN2Conv,
        "Linear": MyLinear,
    }[gnn_name]
    gkw = merge_dict_by_keys({}, kwargs, inspect.getfullargspec(gnn_cls.__init__).args)
    return gnn_cls, gkw


class GraphEncoderSequential(nn.Module):

    def __init__(self, layer_name_list: List[str], num_layers_list: List[int],
                 in_channels, hidden_channels, out_channels,
                 activation="relu", use_bn=False, use_gn=False, use_skip=False,
                 dropout_channels=0.0, dropout_edges=0.0,
                 activate_last=True, force_lin_x_0=False,
                 **kwargs):
        assert len(layer_name_list) == len(num_layers_list)
        self.num_encoders = len(layer_name_list)
        super().__init__()

        for i, (layer_name, num_layers) in enumerate(zip(layer_name_list, num_layers_list)):
            if i == 0:
                _in_channels, _out_channels, _activate_last = in_channels, hidden_channels, True
            else:  # i < self.num_encoders - 1:
                _in_channels, _out_channels, _activate_last = hidden_channels, hidden_channels, True

            if i == self.num_encoders - 1:
                _out_channels, _activate_last = out_channels, activate_last

            self.add_module(
                f"enc_{i}",
                GraphEncoder(layer_name, num_layers, _in_channels, hidden_channels, _out_channels,
                             activation, use_bn, use_gn, use_skip, dropout_channels, dropout_edges,
                             _activate_last, force_lin_x_0, **kwargs))

    def encoders(self):
        for i in range(self.num_encoders):
            yield getattr(self, f"enc_{i}")

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        for encoder in self.encoders():
            x = encoder(x, edge_index, edge_attr, batch, **kwargs)
        return x


class GraphEncoder(nn.Module):

    def __init__(self, layer_name, num_layers, in_channels, hidden_channels, out_channels,
                 activation="relu", use_bn=False, use_gn=False, use_skip=False,
                 dropout_channels=0.0, dropout_edges=0.0,
                 activate_last=True, force_lin_x_0=False,
                 **kwargs):
        super().__init__()

        self.layer_name, self.num_layers = layer_name, num_layers
        self.in_channels, self.hidden_channels, self.out_channels = in_channels, hidden_channels, out_channels
        self.activation = activation
        self.use_bn, self.use_gn, self.use_skip = use_bn, use_gn, use_skip
        self.dropout_channels = dropout_channels
        self.dropout_edges = dropout_edges
        self.activate_last = activate_last

        self.force_lin_x_0 = force_lin_x_0
        self.use_x_0, self.lin_x_0 = None, None

        self.gnn_kwargs = {}
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList() if self.use_bn else []
        self.gns = torch.nn.ModuleList() if self.use_gn else []
        self.skips = torch.nn.ModuleList() if self.use_skip else []
        self.build(**kwargs)
        self.reset_parameters()

    def build(self, **kwargs):
        gnn, self.gnn_kwargs = get_gnn_conv_and_kwargs(self.layer_name, **kwargs)

        self.use_x_0 = "x_0" in inspect.getfullargspec(gnn.forward).args
        if self.use_x_0:
            if (self.in_channels == self.hidden_channels and self.force_lin_x_0) or \
                    (self.in_channels != self.hidden_channels):
                self.lin_x_0 = Linear(self.in_channels, self.hidden_channels)
            in_conv_channels = self.hidden_channels
        else:
            in_conv_channels = self.in_channels

        for conv_id in range(self.num_layers):
            _in_channels = in_conv_channels if conv_id == 0 else self.hidden_channels
            _out_channels = self.hidden_channels if (conv_id != self.num_layers - 1) else self.out_channels

            layer_kwargs = self.gnn_kwargs
            if hasattr(gnn, "__construct_init_kwargs__"):
                layer_kwargs = gnn.__construct_init_kwargs__(**self.gnn_kwargs, layer=conv_id + 1)

            self.convs.append(gnn(_in_channels, _out_channels, **layer_kwargs))
            if conv_id != self.num_layers - 1 or self.activate_last:
                if self.use_bn:
                    self.bns.append(nn.BatchNorm1d(_out_channels))
                if self.use_gn:
                    self.gns.append(GraphNorm(_out_channels))
                if self.use_skip:
                    self.skips.append(Linear(_in_channels, _out_channels))

    def reset_parameters(self):
        if self.lin_x_0 is not None:
            self.lin_x_0.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for gn in self.gns:
            gn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        # Order references.
        #  https://d2l.ai/chapter_convolutional-modern/resnet.html#residual-blocks
        #  https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ppi.py#L30-L34
        #  https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/gnn.py#L69-L76
        if self.use_x_0:
            if self.lin_x_0 is not None:
                x = act(self.lin_x_0(x), self.activation)
                x = F.dropout(x, p=self.dropout_channels, training=self.training)
            kwargs["x_0"] = x

        for i, conv in enumerate(self.convs):
            x_before_layer = x
            _edge_index, _edge_attr = dropout_adj_st(
                edge_index, edge_attr,
                p=self.dropout_edges, num_nodes=x.size(0),
                training=self.training)
            x = conv(x, _edge_index, _edge_attr, **kwargs)
            if i != self.num_layers - 1 or self.activate_last:
                if self.use_bn:
                    x = self.bns[i](x)
                if self.use_gn:
                    x = self.gns[i](x, batch)
                if self.use_skip:
                    x = x + self.skips[i](x_before_layer)
                x = act(x, self.activation)
                x = F.dropout(x, p=self.dropout_channels, training=self.training)
        return x

    def __gnn_kwargs_repr__(self):
        if len(self.gnn_kwargs) == 0:
            return ""
        else:
            return ", " + ", ".join([f"{k}={v}" for k, v in self.gnn_kwargs.items()])

    def __repr__(self):
        return "{}(conv={}, L={}, I={}, H={}, O={}, act={}, act_last={}, skip={}, bn={}, gn={}{})".format(
            self.__class__.__name__, self.layer_name, self.num_layers,
            self.in_channels, self.hidden_channels, self.out_channels,
            self.activation, self.activate_last, self.use_skip, self.use_bn, self.use_gn,
            self.__gnn_kwargs_repr__(),
        )


class WL4Subgraph(nn.Module):

    def __init__(self, num_layers, out_channels, dropout=0.0, concat=False, cache=False):
        self.x_cache = {}
        super().__init__()
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.dropout = dropout
        self.concat = concat
        self.cache = cache
        self.wl_convs = torch.nn.ModuleList([WLConv() for _ in range(num_layers)])
        # NOTE: Lazy modules are a new feature under heavy development so changes to the API
        #  or functionality can happen at any moment.
        self.lin = nn.LazyLinear(out_channels)

    def forward_wl(self, x, edge_index, batch=None, x_to_xs=None):

        if x.dim() > 1 and x.size(-1) == 1:
            x = x.flatten()

        hists = []
        for wl_step, conv in enumerate(self.wl_convs):

            if not self.cache:
                x = conv(x, edge_index)
            else:
                cache_key = (wl_step, x.size(0), edge_index.size(1))
                if cache_key not in self.x_cache:
                    self.x_cache[cache_key] = conv(x, edge_index)
                x = self.x_cache[cache_key]

            if x_to_xs is not None:  # for connected subgraphs
                hists.append(conv.histogram(x[x_to_xs], batch, norm=True))
            else:
                hists.append(conv.histogram(x, batch, norm=True))
        return hists

    def forward(self, x=None, edge_index=None, batch=None, x_to_xs=None, **kwargs):

        hists = self.forward_wl(x, edge_index, batch, x_to_xs)

        if self.concat:
            hist_x = torch.cat(hists, dim=1)  # [num_subgraphs, \sum colors_i]
        else:
            hist_x = hists[-1]  # [num_subgraphs, colors_last]

        hist_x = F.dropout(hist_x, p=self.dropout, training=self.training)
        return self.lin(hist_x)  # [num_subgraphs, out_channels]

    def __repr__(self):
        return f"{self.__class__.__name__}(L={self.num_layers}, O={self.out_channels}, " \
               f"dropout={self.dropout}, concat={self.concat}, cache={self.cache})"


class EdgePredictor(nn.Module):

    def __init__(self, predictor_type, num_layers, hidden_channels, out_channels,
                 activation="relu", out_activation=None, use_bn=False, use_gn=False,
                 dropout_channels=0.0):
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
            use_gn=use_gn,
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
            self.out_linear = Linear(
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
            self.att = GlobalAttention(Linear(encoder.out_channels, 1))

    def aggregate(self, x, batch):
        if self.att is None:
            return Readout.aggregate(self.aggr, x, batch)
        else:
            return self.att(x, batch)

    def forward(self, x, batch=None, *args, **kwargs):
        x = self.encoder(x, batch)
        x = self.aggregate(x, batch)
        x = self.decoder(x)  # batch is not matched for decoder.
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
        if self.embedding is not None:
            return "{}({}, {}, type={})".format(
                self.__class__.__name__,
                self.num_entities,
                self.num_channels,
                self.embedding_type,
            )
        else:
            return "{}(type={})".format(self.__class__.__name__, self.embedding_type)


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

    MODE = "GraphEncoderSequential"

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
        cprint("-- sum w/ batch", "green")
        _z, _ = _ro(_x, _batch)
        print(_ro)
        print("_z", _z.size())  # [2, 64]

        _ro = Readout(readout_types="sum", use_in_mlp=True, use_out_linear=True,
                      num_in_layers=2, hidden_channels=64, out_channels=3)
        cprint("-- sum w/ batch", "green")
        _z, _logits = _ro(_x, _batch)
        print(_ro)
        print("_z", _z.size())  # [2, 64]
        print("_logits", _logits.size())  # [2, 3]

        _ro = Readout(readout_types="mean-sum", use_in_mlp=True, use_out_linear=True,
                      num_in_layers=2, hidden_channels=64, out_channels=3)
        cprint("-- mean-sum w/ batch", "green")
        _z, _logits = _ro(_x, _batch)
        print(_ro)
        print("_z", _z.size())  # [2, 128]
        print("_logits", _logits.size())  # [2, 3]

    elif MODE == "DeepSets":
        _x = torch.ones(10 * 64).view(10, 64)
        _batch = torch.zeros(10).long()
        _batch[4:] = 1
        _ds = DeepSets(
            encoder=MLP(2, 64, 17, 17, "relu", use_gn=True),
            decoder=MLP(2, 17, 17, 10, "relu"),
            aggr="attention",
        )
        print(_ds)
        print(_ds(_x, _batch).size())

    elif MODE == "GraphEncoder":
        enc = GraphEncoder(
            layer_name="GCN2Conv", num_layers=3, in_channels=32, hidden_channels=64, out_channels=128,
            activation="relu", use_bn=False, use_gn=True, use_skip=False, dropout_channels=0.0, dropout_edges=0.2,
            activate_last=True,
            alpha=0.5, theta=1.0, shared_weights=False,
        )
        cprint(enc, "green")
        _x = torch.ones(10 * 32).view(10, -1)
        _ei = torch.randint(0, 10, [2, 10])
        print(enc(_x, _ei, _ei[0].float()).size())

        enc = GraphEncoder(
            layer_name="GCN2Conv", num_layers=3, in_channels=32, hidden_channels=64, out_channels=128,
            activation="relu", use_bn=True, use_gn=True, use_skip=False, dropout_channels=0.0, dropout_edges=0.2,
            activate_last=True,
            alpha=0.5, shared_weights=True,
        )
        cprint(enc, "green")
        print(enc(_x, _ei).size())

        enc = GraphEncoder(
            layer_name="FAConv", num_layers=3, in_channels=32, hidden_channels=64, out_channels=128,
            activation="relu", use_bn=True, use_gn=True, use_skip=False, dropout_channels=0.0, dropout_edges=0.2,
            activate_last=True,
        )
        cprint(enc, "green")
        print(enc(_x, _ei).size())

        enc = GraphEncoder(
            layer_name="FAConv", num_layers=3, in_channels=32, hidden_channels=32, out_channels=128,
            activation="relu", use_bn=False, use_gn=True, use_skip=False, dropout_channels=0.0, dropout_edges=0.2,
            activate_last=True,
        )
        cprint(enc, "green")
        _x = torch.ones(10 * 32).view(10, -1)
        _ei = torch.randint(0, 10, [2, 10])
        print(enc(_x, _ei).size())

        enc = GraphEncoder(
            layer_name="GATConv", num_layers=3, in_channels=32, hidden_channels=64, out_channels=64,
            activation="elu", use_bn=True, use_gn=False, use_skip=True, dropout_channels=0.0,
            activate_last=True, add_self_loops=False, heads=8,
        )
        cprint(enc, "green")
        print(enc(_x, _ei).size())

    elif MODE == "GraphEncoderSequential":

        _ges = GraphEncoderSequential(
            ["Linear", "GCN2Conv"], [2, 4], in_channels=32, hidden_channels=64, out_channels=7,
            activation="relu", use_bn=False, use_gn=True, use_skip=False, dropout_channels=0.0, dropout_edges=0.2,
            activate_last=False,
            alpha=0.5, theta=1.0, shared_weights=False,
        )
        cprint(_ges, "green")
        _x = torch.ones(10 * 32).view(10, -1)
        _ei = torch.randint(0, 10, [2, 10])
        _batch = torch.ones(10).long()
        _batch[:4] = 0
        print(_ges(_x, _ei, _ei[0].float(), _batch).size())

        _ges = GraphEncoderSequential(
            ["GCN2Conv"], [2], in_channels=32, hidden_channels=64, out_channels=7,
            activation="relu", use_bn=True, use_gn=False, use_skip=False, dropout_channels=0.0, dropout_edges=0.2,
            activate_last=True,
            alpha=0.5, theta=1.0, shared_weights=False,
        )
        cprint(_ges, "green")
        print(_ges(_x, _ei, _ei[0].float()).size())

    elif MODE == "WL4Subgraph":
        _x = torch.ones(10).view(10, -1).long()
        _ei = torch.cat([
            torch.randint(0, 5, [2, 7]),
            torch.randint(5, 10, [2, 9]),
        ], dim=1)
        _batch = torch.zeros(10).long()
        _batch[5:] = 1
        _x_to_xs = torch.Tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]).long()

        wl4s = WL4Subgraph(num_layers=4, out_channels=2, dropout=0.2, concat=True, cache=True)
        print(wl4s)
        print(wl4s(_x, _ei, _batch))
        print(wl4s(_x, _ei, _batch))
        wl4s = WL4Subgraph(num_layers=3, out_channels=3, concat=False)
        print(wl4s(_x, _ei, _batch, _x_to_xs))

    elif MODE == "MyFAConv":

        _x = torch.ones(10 * 32).view(10, -1)
        _ei = torch.randint(0, 10, [2, 10])

        _fac = MyFAConv(32, 32)
        print(_fac)
        print(_fac(_x, _ei, x_0=_x).size())

        _fac = MyFAConv(32, 7)
        print(_fac)
        print(_fac(_x, _ei, x_0=_x).size())

    elif MODE == "MyGCN2Conv":

        _x = torch.ones(10 * 32).view(10, -1)
        _ei = torch.randint(0, 10, [2, 10])

        _fac = MyGCN2Conv(32, 32, alpha=0.1, theta=0.5, layer=1)
        print(_fac)
        print(_fac(_x, _ei, x_0=_x).size())

        _fac = MyGCN2Conv(32, 7, alpha=0.5, theta=1.0, layer=1)
        print(_fac)
        print(_fac(_x, _ei, x_0=_x).size())

    elif MODE == "VersatileEmbedding":
        _pte = torch.arange(11 * 32).view(11, 32).float()
        de = VersatileEmbedding(embedding_type="Pretrained", num_entities=11, num_channels=32,
                                pretrained_embedding=_pte, freeze_pretrained=False)
        cprint(de, "green")
        print("Embedding: {} +- {}".format(
            de.embedding.weight.mean().item(),
            de.embedding.weight.std().item(),
        ))

        _x = torch.arange(11)
        print(de(_x).size())
