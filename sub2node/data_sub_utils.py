import random
from typing import List, Any

import networkx as nx
from pathlib import Path

from termcolor import cprint

"""Adopted from https://github.com/mims-harvard/SubGNN/blob/main/prepare_dataset/prepare_dataset.py"""


def generate_mask(n_subgraphs):
    """
    Generate train/val/test masks for the subgraphs.
    Args
        - n_subgraphs (int): number of subgraphs
    Return
        - mask (list): 0 if subgraph is in train set, 1 if in val set, 2 if in test set
    """

    idx = set(range(n_subgraphs))
    train_mask = list(random.sample(idx, int(len(idx) * 0.8)))
    idx = idx.difference(set(train_mask))
    val_mask = list(random.sample(idx, len(idx) // 2))
    idx = idx.difference(set(val_mask))
    test_mask = list(random.sample(idx, len(idx)))
    mask = []
    for i in range(n_subgraphs):
        if i in train_mask:
            mask.append(0)
        elif i in val_mask:
            mask.append(1)
        elif i in test_mask:
            mask.append(2)
    return mask


def save_subgraphs(path: str, nodes_in_subgraphs: List[List[int]], labels: List[Any or List[Any]]):
    """
    Write subgraph information into the appropriate format for SubGNN (tab-delimited file where each row
    has dash-delimited nodes, subgraph label, and train/val/test label).
    Args
        - sub_f (str): file directory to save subgraph information
        - sub_G (list of lists): list of subgraphs, where each subgraph is a list of nodes
        - sub_G_label (list): subgraph labels
        - mask (list): 0 if subgraph is in train set, 1 if in val set, 2 if in test set
    """
    masks = generate_mask(len(labels))
    with open(path, "w") as fout:
        for g, l, m in zip(nodes_in_subgraphs, labels, masks):
            g = [str(val) for val in g]
            if len(g) == 0:
                continue
            if m == 0:
                fout.write("\t".join(["-".join(g), str(l), "train", "\n"]))
            elif m == 1:
                fout.write("\t".join(["-".join(g), str(l), "val", "\n"]))
            elif m == 2:
                fout.write("\t".join(["-".join(g), str(l), "test", "\n"]))
    cprint(f"Save {len(labels)} subgraphs in {path}", "green")
