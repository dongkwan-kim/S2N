"""PyG wrapper for graph_coarsening

The original code is adopted from
https://github.com/szzhang17/Scaling-Up-Graph-Neural-Networks-Via-Graph-Coarsening/blob/main/APPNP/utils.py

and some minor modifications are applied.
"""
from pprint import pprint

import numpy as np
import pygsp as gsp
import scipy.sparse.csc
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm

from graph_coarsening.coarsening_utils import coarsen


def extract_components(H):
    if H.A.shape[0] != H.A.shape[1]:
        H.logger.error("Inconsistent shape to extract components. "
                       "Square matrix required.")
        return None

    if H.is_directed():
        raise NotImplementedError("Directed graphs not supported yet.")

    graphs = []
    visited = np.zeros(H.A.shape[0], dtype=bool)

    while not visited.all():
        stack = set([np.nonzero(~visited)[0][0]])
        comp = []

        while len(stack):
            v = stack.pop()
            if not visited[v]:
                comp.append(v)
                visited[v] = True

                stack.update(set([idx for idx in H.A[v, :].nonzero()[1]
                                  if not visited[idx]]))

        comp = sorted(comp)
        G = H.subgraph(comp)
        G.info = {"orig_idx": comp}
        graphs.append(G)

    return graphs


def coarsening_pyg(data: Data, coarsening_ratio, coarsening_method, min_size):
    assert coarsening_method in ["variation_neighborhoods", "variation_edges",
                                 "variation_cliques", "heavy_edge",
                                 "algebraic_JC", "affinity_GS", "kron"]
    # if dataset == "dblp":
    #     dataset = CitationFull(root="./dataset", name=dataset)
    # elif dataset == "Physics":
    #     dataset = Coauthor(root="./dataset/Physics", name=dataset)
    # else:
    #     dataset = Planetoid(root="./dataset", name=dataset)
    # data = dataset[0]
    G = gsp.graphs.Graph(W=to_dense_adj(data.edge_index)[0])
    components = extract_components(G)
    print(f"The number of components is {len(components)}")

    candidate = sorted(components, key=lambda x: len(x.info["orig_idx"]), reverse=True)

    # Original codes:
    #
    # C_list, Gc_list = [], []
    # number = 0
    # while number < len(candidate):
    #     H = candidate[number]
    #     if len(H.info["orig_idx"]) > 10:
    #         C, Gc, Call, Gall = coarsen(H, r=coarsening_ratio, method=coarsening_method)
    #         C_list.append(C)
    #         Gc_list.append(Gc)
    #     number += 1
    index_list, rows_list, cols_list = [], [], []
    for number in tqdm(range(len(candidate)), desc="s2n_coarsening_utils.coarsening_pyg by candidate"):
        H = candidate[number]
        if len(H.info["orig_idx"]) > min_size:
            C, Gc, Call, Gall = coarsen(H, r=coarsening_ratio, method=coarsening_method, K=min_size)
            C: scipy.sparse.csc.csc_matrix
            rows, cols = C.nonzero()

            index_list.append(torch.Tensor(H.info["orig_idx"]).long())
            rows_list.append(torch.from_numpy(rows).long())
            cols_list.append(torch.from_numpy(cols).long())
        else:
            print(f"Small components: {H.info['orig_idx']}")
    # return data.x.shape[1], len(set(np.array(data.y))), candidate, C_list, Gc_list
    return index_list, rows_list, cols_list


def coarsening_pyg_batch(data: Data, coarsening_ratio, coarsening_method, min_size=4):
    assert coarsening_method in ["variation_neighborhoods", "variation_edges",
                                 "variation_cliques", "heavy_edge",
                                 "algebraic_JC", "affinity_GS", "kron"]
    index_list, rows_list, cols_list = coarsening_pyg(
        data, coarsening_ratio, coarsening_method, min_size
    )
    """
    ([tensor([0, 1, 2, 3, 4, 5]), tensor([ 8,  9, 10, 12, 13])],
     [tensor([0, 0, 0, 1, 1, 1]), tensor([0, 0, 0, 1, 1])],
     [tensor([0, 1, 2, 3, 4, 5]), tensor([0, 2, 4, 1, 3])])
    -->
    ([tensor([0, 1, 2, 3, 4, 5,   8,  9, 10, 12, 13])],
     [tensor([0, 0, 0, 1, 1, 1,   2, 2, 2, 3, 3])],     # + num_coarsened_nodes (2)
     [tensor([0, 1, 2, 3, 4, 5,   6, 8, 10, 7, 9])])    # + num_original_index (6)
    """

    cum_rows_list, cum_cols_list = [], []
    cum_rows_index, cum_cols_index = 0, 0
    for rows, cols in zip(rows_list, cols_list):
        cum_rows_list.append(rows + cum_rows_index)
        cum_cols_list.append(cols + cum_cols_index)
        cum_rows_index += torch.max(rows) + 1
        cum_cols_index += cols.size(0)

    x_ids = torch.cat(index_list)  # Tensor of original node ids
    batch = torch.cat(cum_rows_list)  # Tensor of coarsened node ids
    sub_x_index = torch.cat(cum_cols_list)  # Tensor of indices in xs (original node ids)
    # x_ids[sub_x_index] = Tensor of original node ids in xs
    return x_ids, batch, sub_x_index


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def splits(data, num_classes, split_type):
    if split_type != "fixed":
        indices = []
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        if split_type == "random":
            train_index = torch.cat([i[:20] for i in indices], dim=0)
            val_index = torch.cat([i[20:50] for i in indices], dim=0)
            test_index = torch.cat([i[50:] for i in indices], dim=0)
        else:
            train_index = torch.cat([i[:5] for i in indices], dim=0)
            val_index = torch.cat([i[5:10] for i in indices], dim=0)
            test_index = torch.cat([i[10:] for i in indices], dim=0)

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(test_index, size=data.num_nodes)

    return data


def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]


def load_data(dataset, candidate, C_list, Gc_list, split_type):
    # if dataset == "dblp":
    #     dataset = CitationFull(root="./dataset", name=dataset)
    # elif dataset == "Physics":
    #     dataset = Coauthor(root="./dataset/Physics", name=dataset)
    # else:
    #     dataset = Planetoid(root="./dataset", name=dataset)
    n_classes = len(set(np.array(dataset[0].y)))
    data = splits(dataset[0], n_classes, split_type)
    train_mask = data.train_mask
    val_mask = data.val_mask
    labels = data.y
    features = data.x

    coarsen_node = 0
    number = 0
    coarsen_row = None
    coarsen_col = None
    coarsen_features = torch.Tensor([])
    coarsen_train_labels = torch.Tensor([])
    coarsen_train_mask = torch.Tensor([]).bool()
    coarsen_val_labels = torch.Tensor([])
    coarsen_val_mask = torch.Tensor([]).bool()

    while number < len(candidate):
        H = candidate[number]
        keep = H.info["orig_idx"]
        H_features = features[keep]
        H_labels = labels[keep]
        H_train_mask = train_mask[keep]
        H_val_mask = val_mask[keep]
        if len(H.info["orig_idx"]) > 10 and torch.sum(H_train_mask) + torch.sum(H_val_mask) > 0:
            train_labels = one_hot(H_labels, n_classes)
            train_labels[~H_train_mask] = torch.Tensor([0 for _ in range(n_classes)])
            val_labels = one_hot(H_labels, n_classes)
            val_labels[~H_val_mask] = torch.Tensor([0 for _ in range(n_classes)])

            C = C_list[number]
            Gc = Gc_list[number]

            new_train_mask = torch.BoolTensor(np.sum(C.dot(train_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(train_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_train_mask[mix_mask > 1] = False

            new_val_mask = torch.BoolTensor(np.sum(C.dot(val_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(val_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_val_mask[mix_mask > 1] = False

            coarsen_features = torch.cat([coarsen_features, torch.FloatTensor(C.dot(H_features))], dim=0)
            coarsen_train_labels = torch.cat(
                [coarsen_train_labels, torch.argmax(torch.FloatTensor(C.dot(train_labels)), dim=1).float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, new_train_mask], dim=0)
            coarsen_val_labels = torch.cat(
                [coarsen_val_labels, torch.argmax(torch.FloatTensor(C.dot(val_labels)), dim=1).float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, new_val_mask], dim=0)

            if coarsen_row is None:
                coarsen_row = Gc.W.tocoo().row
                coarsen_col = Gc.W.tocoo().col
            else:
                current_row = Gc.W.tocoo().row + coarsen_node
                current_col = Gc.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += Gc.W.shape[0]

        elif torch.sum(H_train_mask) + torch.sum(H_val_mask) > 0:

            coarsen_features = torch.cat([coarsen_features, H_features], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, H_labels.float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, H_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, H_labels.float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, H_val_mask], dim=0)

            if coarsen_row is None:
                raise Exception("The graph does not need coarsening_pyg.")
            else:
                current_row = H.W.tocoo().row + coarsen_node
                current_col = H.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += H.W.shape[0]
        number += 1

    print("the size of coarsen graph features:", coarsen_features.shape)

    coarsen_edge = torch.LongTensor([coarsen_row, coarsen_col])
    coarsen_train_labels = coarsen_train_labels.long()
    coarsen_val_labels = coarsen_val_labels.long()

    return (data, coarsen_features, coarsen_train_labels, coarsen_train_mask,
            coarsen_val_labels, coarsen_val_mask, coarsen_edge)


if __name__ == "__main__":
    print("wow")
