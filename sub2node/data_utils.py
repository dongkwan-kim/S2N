from typing import Optional, Union

import torch
from torch import Tensor
from torch_geometric.data import HeteroData, Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import add_self_loops


class RelabelNodes(BaseTransform):

    def __call__(self, data):
        x, edge_index = data.x, data.edge_index

        x_before_relabeling = x.squeeze(-1)  # [*]
        num_nodes_before_relabeling = torch.max(x_before_relabeling).long().item()
        num_nodes_after_relabeling = x_before_relabeling.size(0)

        _node_idx = torch.full((num_nodes_before_relabeling + 1, ),
                               fill_value=-1, dtype=torch.long)
        _node_idx[x_before_relabeling] = torch.arange(num_nodes_after_relabeling)

        data.edge_index = _node_idx[edge_index]
        return data


class AddSelfLoopsV2(BaseTransform):
    r"""Adds self-loops to the given homogeneous or heterogeneous graph.

    Args:
        attr: (str, optional): The name of the attribute of edge weights
            or multi-dimensional edge features, to pass it to the
            :meth:`torch_geometric.utils.add_self_loops`.
            (default: :obj:`edge_weight`)
        fill_value (float or Tensor or str, optional): The way to generate
            edge features of self-loops (in case :obj:`edge_attr != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`1.`)
    """
    def __init__(self, attr: Optional[str] = 'edge_weight',
                 fill_value: Union[float, Tensor, str] = None):
        self.attr = attr
        self.fill_value = fill_value

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.edge_stores:
            if store.is_bipartite() or 'edge_index' not in store:
                continue

            store.edge_index, edge_weight = add_self_loops(
                store.edge_index,
                getattr(store, self.attr, None),
                fill_value=self.fill_value,
                num_nodes=store.size(0))

            setattr(store, self.attr, edge_weight)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
