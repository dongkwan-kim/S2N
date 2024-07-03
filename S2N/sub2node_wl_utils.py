from collections import namedtuple

import numpy as np
import torch
from sklearn.metrics.pairwise import linear_kernel

from utils_fscache_v2 import fscaches

K_ARGS = namedtuple("K_ARGS", ["k_to_sample", "stype", "norm", "dataset_name", "wl_cumcat", "wl_layers"])

# NOTE: hard-coded values
BEST_HPARAMS = {
    "EMUser": {"wl_i+1": 2, "wl_cumcat": False, "hist_norm": False, "a_c": 0.9, "a_s": 0.1},
    "HPOMetab": {"wl_i+1": 3, "wl_cumcat": True, "hist_norm": True, "a_c": 0.99, "a_s": 0.01},
    "HPONeuro": {"wl_i+1": 4, "wl_cumcat": False, "hist_norm": False, "a_c": 0.99, "a_s": 0.01},
    "PPIBP": {"wl_i+1": 2, "wl_cumcat": False, "hist_norm": True, "a_c": 0.99, "a_s": 0.01},
    "Component": {"wl_i+1": 1, "wl_cumcat": False, "hist_norm": False, "a_c": 0.99, "a_s": 0.01},
    "Coreness": {"wl_i+1": 1, "wl_cumcat": False, "hist_norm": False, "a_c": 0.001, "a_s": 0.999},
    "CutRatio": {"wl_i+1": 2, "wl_cumcat": True, "hist_norm": False, "a_c": 0.9, "a_s": 0.1},
    "Density": {"wl_i+1": 1, "wl_cumcat": False, "hist_norm": False, "a_c": 0.1, "a_s": 0.9},
}


def kk(k_to_sample, stype, norm, i, *args):
    return "_".join([str(s) for s in [*args, k_to_sample, stype, norm, i]])


def get_kernel_key(args, splits, i):
    kernel_key = kk(args.k_to_sample, args.stype, args.norm, i)
    if splits[-1] <= 250:  # synthetic graphs, backward compatibility
        kernel_key = kk(args.k_to_sample, args.stype[:3], args.norm, i, args.dataset_name)
    return kernel_key


@fscaches(path="../_caches", keys_to_exclude=["hist"], verbose=True)
def hist_linear_kernels(hist, splits, key):
    s = splits
    train_hist, val_hist, test_hist = hist[s[0]:s[1]], hist[s[1]:s[2]], hist[s[2]:s[3]]
    K_train = linear_kernel(train_hist, train_hist)
    K_val = linear_kernel(val_hist, train_hist)
    K_test = linear_kernel(test_hist, train_hist)
    return K_train, K_val, K_test


def get_all_kernels(args, splits):
    try:
        kernels = [hist_linear_kernels(hist=None, splits=splits, key=get_kernel_key(args, splits, i))
                   for i in range(args.wl_layers)]
    except:
        kernels = None
    return kernels


def get_one_kernel(args, splits, i):
    if not args.wl_cumcat:
        return hist_linear_kernels(hist=None, splits=splits, key=get_kernel_key(args, splits, i))
    else:
        all_kernels = get_all_kernels(args, splits)
        cumcat_list = []
        sum_k_list = [None, None, None]
        for _, k_list in enumerate(all_kernels):
            sum_k_list = k_list if sum_k_list[0] is None else [k + sum_k for k, sum_k in zip(k_list, sum_k_list)]
            cumcat_list.append(tuple(sum_k_list))
        kt = cumcat_list[i]
        return kt


def get_ewmat_by_wl_kernel_for_s2n(s2n_splits, dataset_name, **kwargs):
    assert len(s2n_splits) == 3  # [tr, tr+v, tr+v+te], e.g., [1272, 1432, 1591]
    splits = [0] + s2n_splits

    bh = BEST_HPARAMS[dataset_name]
    bh.update(kwargs)
    kws = {"k_to_sample": None, "norm": bh["hist_norm"], "dataset_name": dataset_name, "wl_cumcat": bh["wl_cumcat"],
           "wl_layers": 5}

    kt_s = get_one_kernel(K_ARGS(stype="separated", **kws), splits, bh["wl_i+1"] - 1)
    kt_c = get_one_kernel(K_ARGS(stype="connected", **kws), splits, bh["wl_i+1"] - 1)

    # shapes: [tr, tr], [v, tr], [te, tr]
    k_tr, k_v, k_te = tuple([(bh["a_c"] * k_c + bh["a_s"] * k_s) for k_c, k_s in zip(kt_c, kt_s)])

    # Reshape to [tr+v+te, tr+v+te]
    tr, tr_v, S = s2n_splits
    ewmat = np.zeros((S, S))
    ewmat[:tr, :tr] = k_tr
    ewmat[tr:tr_v, :tr] = k_v
    ewmat[tr_v:, :tr] = k_te
    ewmat[:tr, tr:tr_v] = k_v.transpose()
    ewmat[:tr, tr_v:] = k_te.transpose()
    return torch.from_numpy(ewmat).float()


if __name__ == '__main__':
    _ewmat = get_ewmat_by_wl_kernel_for_s2n([1272, 1432, 1591], "PPIBP")
    print(_ewmat)
    print(_ewmat.shape)

    _ewmat = get_ewmat_by_wl_kernel_for_s2n([1272, 1432, 1591], "PPIBP", wl_cumcat=True)
    print(_ewmat)
    print(_ewmat.shape)

    # _ewmat = get_ewmat_by_wl_kernel_for_s2n([226, 275, 324], "EMUser")
    # print(_ewmat)
    # print(_ewmat.shape)

    _ewmat = get_ewmat_by_wl_kernel_for_s2n([200, 225, 250], "Density")
    print(_ewmat)
    print(_ewmat.shape)

    _ewmat = get_ewmat_by_wl_kernel_for_s2n([200, 225, 250], "Density", hist_norm=True, wl_cumcat=True)
    print(_ewmat)
    print(_ewmat.shape)
