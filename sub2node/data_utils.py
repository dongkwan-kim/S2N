import torch
from torch_geometric.transforms import BaseTransform


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
