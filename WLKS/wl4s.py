import argparse
import copy
import gc
import inspect
import itertools
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import linear_kernel
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC, SVC
from termcolor import cprint
from torch_geometric.data import Batch
from torch_geometric.nn.conv import WLConv
from torch_geometric.nn.glob import global_add_pool
from tqdm import tqdm

from data_sub import HPONeuro, PPIBP, HPOMetab, EMUser, Density, Component, Coreness, CutRatio
from data_transform import KHopSubgraph, ShuffleAndSample
from utils import str2bool
from utils_fscache import fscaches
from visualize import plot_data_points_by_tsne

ModelType = Union[MultiOutputClassifier, LinearSVC, SVC]

DATASETS_REAL = ["PPIBP", "EMUser", "HPOMetab", "HPONeuro"]
DATASETS_SYN = ["CutRatio", "Density", "Coreness", "Component"]
MODEL_KWARGS_KEY = ["C", "kernel", "dual"]

parser = argparse.ArgumentParser()
parser.add_argument('--MODE', type=str, default="run_one")
parser.add_argument('--dataset_name', type=str, default="PPIBP",
                    choices=["PPIBP", "HPOMetab", "EMUser", "HPONeuro", "Density", "Component", "Coreness", "CutRatio"])
parser.add_argument('--stype', type=str, default="connected", choices=["connected", "separated"])
parser.add_argument('--dtype', type=str, default="kernel", choices=["histogram", "kernel"])
parser.add_argument('--wl_layers', type=int, default=5)
parser.add_argument('--wl_cumcat', type=str2bool, default=False, help="Whether to concat WL hists")
parser.add_argument('--hist_norm', type=str2bool, default=True, help="Whether to normalize WL hists")
parser.add_argument('--k_to_sample', type=int, default=None, help="For wl4s_k.py")
parser.add_argument('--ratio_samples', type=float, default=1.0, help="Only when k_to_sample != 0")
parser.add_argument('--model', type=str, default="SVC", choices=["LinearSVC", "SVC"])
parser.add_argument('--C', type=float, default=1.0)
parser.add_argument('--kernel', type=str, default="precomputed")
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--dataset_path', type=str, default="/mnt/nas2/GNN-DATA/SUBGRAPH")


class WL4S(torch.nn.Module):
    def __init__(self, dataset_name, stype, num_layers, norm,
                 dtype="histogram", splits=None, k_to_sample=None, precompute=False, de=None):
        super(WL4S, self).__init__()
        self.dataset_name = dataset_name
        self.stype = stype
        self.norm = self.hist_norm = norm
        self.dtype = dtype
        self.splits = splits
        self.precompute = precompute
        self.de = de
        self.k_to_sample = k_to_sample
        self.convs = torch.nn.ModuleList([WLConv() for _ in range(num_layers)])

    def forward(self, x, edge_index, batch_or_sub_batch, x_to_xs=None, mask=None):
        hists_or_kernels = []
        for i, conv in enumerate(tqdm(self.convs, desc="WL4S.forward")):

            x = conv(x, edge_index)

            if self.stype == "connected":
                h = conv.histogram(x[x_to_xs], batch_or_sub_batch, norm=self.norm)
            elif self.stype == "separated":
                _x, _b = x, batch_or_sub_batch
                if mask is not None:
                    _x, _b = x[mask], batch_or_sub_batch[mask]
                h = conv.histogram(_x, _b, norm=self.norm)
            else:
                raise ValueError

            if self.de is not None:
                des = 10000 * global_add_pool(self.de[x_to_xs], batch_or_sub_batch, size=None)
                h = torch.cat([h, des], dim=-1)

            if self.dtype == "kernel":
                kernel_key = get_kernel_key(self, self.splits, i)
                h = hist_linear_kernels(hist=h, splits=self.splits, key=kernel_key)
                if self.precompute:
                    del h
                    gc.collect()
                    h = None
            hists_or_kernels.append(h)
        return hists_or_kernels


def kk(k_to_sample, stype, norm, i, *args):
    return "_".join([str(s) for s in [*args, k_to_sample, stype, norm, i]])


def get_kernel_key(args, splits, i):
    kernel_key = kk(args.k_to_sample, args.stype, args.hist_norm, i)
    if splits[-1] <= 250:  # synthetic graphs, backward compatibility
        kernel_key = kk(args.k_to_sample, args.stype[:3], args.hist_norm, i, args.dataset_name)

    if hasattr(args, "de") and args.de is not None:
        kernel_key = f"{kernel_key}_de_{args.de.size(-1)}_add10"
        print(kernel_key)
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
    except Exception as e:
        print(e)
        kernels = None
    return kernels


def precompute_all_kernels(args):
    cprint("Precomputing kernels...", "yellow")
    get_data_and_model(args, precompute=True)


def get_data_and_model(args, precompute=False, use_de=False):
    args = copy.deepcopy(args)
    embedding_type = "glass"
    if args.dataset_name in ["Density", "Component", "Coreness", "CutRatio"] and use_de:
        embedding_type = "RWPE_K_64"

    dts: Union[HPONeuro, PPIBP, HPOMetab, EMUser, Density, Component, Coreness, CutRatio] = eval(args.dataset_name)(
        root=args.dataset_path,
        name=args.dataset_name,
        embedding_type=embedding_type,
        debug=False,
        load_rwpe=use_de,
    )
    if use_de:
        args.de = dts.global_data.pe
    # dts.print_summary()
    splits = [0] + dts.splits + [len(dts)]
    if args.ratio_samples < 1.0:
        splits = [int(s * args.ratio_samples) for s in splits]

    if args.dtype == "kernel" and precompute:
        k_list = get_all_kernels(args, splits)
        if k_list is not None:
            cprint(f"Use precomputed kernels...", "yellow")
            train_dts, val_dts, test_dts = dts.get_train_val_test()
            if args.ratio_samples < 1.0:
                train_dts, val_dts, test_dts = ShuffleAndSample(splits).map_list([train_dts, val_dts, test_dts])
            all_data = Batch.from_data_list(train_dts + val_dts + test_dts)
            return k_list, splits, all_data.y

    if args.k_to_sample is None:
        train_dts, val_dts, test_dts = dts.get_train_val_test_with_individual_relabeling()
        if args.ratio_samples < 1.0:
            train_dts, val_dts, test_dts = ShuffleAndSample(splits).map_list([train_dts, val_dts, test_dts])
    else:
        train_dts, val_dts, test_dts = dts.get_train_val_test()
        if args.ratio_samples < 1.0:
            train_dts, val_dts, test_dts = ShuffleAndSample(splits).map_list([train_dts, val_dts, test_dts])
        khs = KHopSubgraph(dts.global_data.edge_index, k=args.k_to_sample,
                           relabel_nodes=True, num_nodes=dts.global_data.num_nodes)
        train_dts, val_dts, test_dts = khs.map_list([train_dts, val_dts, test_dts])

    assert (len(train_dts) == splits[1] and len(train_dts + val_dts) == splits[2] and
            len(train_dts + val_dts + test_dts) == splits[3]), \
        f"{splits} != [{len(train_dts), len(val_dts), len(test_dts)}]"
    all_data = Batch.from_data_list(train_dts + val_dts + test_dts)

    wl = WL4S(dataset_name=args.dataset_name, stype=args.stype, num_layers=args.wl_layers, norm=args.hist_norm,
              dtype=args.dtype, splits=splits, k_to_sample=args.k_to_sample, precompute=precompute,
              de=dts.global_data.pe if use_de else None)

    if args.stype == "connected":
        dts.global_data.x = torch.ones((dts.global_data.x.size(0), 1)).long()
        data = dts.global_data
        h_or_k_list = wl(data.x, data.edge_index, batch_or_sub_batch=all_data.batch, x_to_xs=all_data.x.flatten())
    else:  # separated
        all_data.x = torch.ones((all_data.x.size(0), 1)).long()
        data = all_data
        h_or_k_list = wl(data.x, data.edge_index, batch_or_sub_batch=all_data.batch, mask=getattr(data, "mask", None))
    return h_or_k_list, splits, all_data.y


def experiment(args, h_or_k_list, splits, all_y, **model_kwargs):
    s = splits
    test_f1s = torch.zeros(args.runs, dtype=torch.float)
    best_val_f1s = torch.zeros(args.runs, dtype=torch.float)
    best_wl = torch.zeros(args.runs, dtype=torch.float)

    if args.wl_cumcat:
        cumcat_list, cumcat_h = [], torch.tensor([])
        sum_k_list = [None, None, None]
        if args.dtype == "histogram":
            for h in h_or_k_list:
                cumcat_h = torch.cat((cumcat_h, h), dim=-1)
                cumcat_list.append(cumcat_h)
        else:  # kernel
            for i, k_list in enumerate(h_or_k_list):
                sum_k_list = k_list if sum_k_list[0] is None else [k + sum_k for k, sum_k in zip(k_list, sum_k_list)]
                cumcat_list.append(tuple(sum_k_list))
        h_or_k_list = cumcat_list

    for run in range(1, args.runs + 1):
        best_val_f1s[run - 1] = 0

        for i_wl, h_or_k in enumerate(h_or_k_list):
            if h_or_k is None:
                continue
            if args.dtype == "histogram":
                train_x, val_x, test_x = h_or_k[s[0]:s[1]], h_or_k[s[1]:s[2]], h_or_k[s[2]:s[3]]
            else:
                assert model_kwargs["kernel"] == "precomputed"
                train_x, val_x, test_x = h_or_k
            train_y, val_y, test_y = all_y[s[0]:s[1]], all_y[s[1]:s[2]], all_y[s[2]:s[3]]

            model: ModelType = eval(args.model)(**model_kwargs)

            if args.dataset_name == "HPONeuro":
                model = MultiOutputClassifier(model)

            model.fit(train_x, train_y)

            val_f1 = f1_score(val_y, model.predict(val_x), average="micro")
            test_f1 = f1_score(test_y, model.predict(test_x), average="micro")

            print(f"\t - wl: {i_wl + 1}, shape: {list(train_x.shape)}, model: {args.model}, "
                  f"val_f1: {val_f1:.5f}, test_f1: {test_f1:.5f}")
            if val_f1 > best_val_f1s[run - 1]:
                best_val_f1s[run - 1] = val_f1
                test_f1s[run - 1] = test_f1
                best_wl[run - 1] = i_wl + 1
        print(f'Run: {run:02d}, Val: {best_val_f1s[run - 1]:.4f}, Test: {test_f1s[run - 1]:.4f}')
    cprint(f'Final Test Performance: {test_f1s.mean():.4f} ± {test_f1s.std():.4f} for {args.model} with {model_kwargs}',
           "green")
    return {
        "test_f1_mean": float(test_f1s.mean()),
        "test_f1_std": float(test_f1s.std()),
        "best_val_f1_mean": float(best_val_f1s.mean()),
        "best_val_f1_std": float(best_val_f1s.std()),
        "best_wl_list": best_wl.tolist(),
    }


def run_one(args, data_func=get_data_and_model, precompute=True, **kwargs):
    h_or_k_list, splits, all_y = data_func(args, precompute=precompute, **kwargs)
    model_kwargs = {k: getattr(args, k) for k in MODEL_KWARGS_KEY if hasattr(args, k)}
    experiment(args, h_or_k_list, splits, all_y, **model_kwargs)


def plot_tsne_all(args, data_func=get_data_and_model, path="../_figures", extension="png"):
    assert args.dtype == "histogram"
    kws = dict(path=path, extension=extension, alpha=0.5, s=5)
    for dataset_name in DATASETS_SYN + DATASETS_REAL:
        for stype in ["connected", "separated"]:
            args.dataset_name, args.stype = dataset_name, stype
            hists, splits, all_y = data_func(args)
            s = splits
            for i_wl, hist in enumerate(hists):
                sub_key = f"{dataset_name}-{stype}, WL: {i_wl + 1}"
                print(sub_key)
                train_hist, val_hist, test_hist = hist[s[0]:s[1]], hist[s[1]:s[2]], hist[s[2]:s[3]]
                train_y, val_y, test_y = all_y[s[0]:s[1]], all_y[s[1]:s[2]], all_y[s[2]:s[3]]
                all_hist, all_y = torch.cat([train_hist, val_hist, test_hist]), torch.cat([train_y, val_y, test_y])
                plot_data_points_by_tsne(all_hist, all_y, key=f"{sub_key}, Split: All", **kws)
                plot_data_points_by_tsne(train_hist, train_y, key=f"{sub_key}, Split: Train", **kws)
                plot_data_points_by_tsne(val_hist, val_y, key=f"{sub_key}, Split: Val", **kws)
                plot_data_points_by_tsne(test_hist, test_y, key=f"{sub_key}, Split: Test", **kws)


def hp_search_for_models(args, hparam_space, more_hparam_space,
                         data_func=get_data_and_model, file_dir="../_logs_wl4s", log_postfix=""):
    def space_to_kwl(space):
        return [dict(zip(space.keys(), cmb)) for cmb in itertools.product(*space.values())]

    def init_args(method_name):
        return [param.name for param in inspect.signature(eval(method_name)).parameters.values()]

    kwargs_list = []
    for kwargs in space_to_kwl(hparam_space):
        more_space = {k: more_hparam_space[k] for k in more_hparam_space if k in init_args(kwargs["model"])}
        kwargs_list += [{**kwargs, **mkw} for mkw in space_to_kwl(more_space)]

    stype_and_norm_to_data_and_model = {}
    for stype in hparam_space["stype"]:
        for hist_norm in hparam_space["hist_norm"]:
            args.stype, args.hist_norm = stype, hist_norm
            print(f"Compute WL hists: {stype} & norm={hist_norm}")
            stype_and_norm_to_data_and_model[(stype, hist_norm)] = data_func(args)

    Path(file_dir).mkdir(exist_ok=True)
    file_path = Path(file_dir) / f"{args.dataset_name}{log_postfix}.csv"
    for i, model_kwargs in enumerate(kwargs_list):
        print(model_kwargs)
        for k in model_kwargs.copy():
            if k not in MODEL_KWARGS_KEY:
                setattr(args, k, model_kwargs.pop(k))
        h_or_k_list, splits, all_y = stype_and_norm_to_data_and_model[(args.stype, args.hist_norm)]
        results = experiment(args, h_or_k_list, splits, all_y, **model_kwargs)

        df = pd.DataFrame([{**results, **args.__dict__, **model_kwargs}])
        if file_path.is_file():
            df = pd.concat([pd.read_csv(file_path), df], ignore_index=True)
        df.to_csv(file_path, index=False)
        cprint(f"Save logs at {file_path}", "blue")


def hp_search_real(args, hparam_space, more_hparam_space, data_func=get_data_and_model,
                   file_dir="../_logs_wl4s", log_postfix=""):
    for dataset_name in DATASETS_REAL:
        args.dataset_name = dataset_name
        hp_search_for_models(args, hparam_space, more_hparam_space, data_func, file_dir, log_postfix)


def hp_search_syn(args, hparam_space, more_hparam_space, data_func=get_data_and_model,
                  file_dir="../_logs_wl4s", log_postfix=""):
    for dataset_name in DATASETS_SYN:
        args.dataset_name = dataset_name
        hp_search_for_models(args, hparam_space, more_hparam_space, data_func, file_dir, log_postfix)


if __name__ == '__main__':

    HPARAM_SPACE = {
        "stype": ["connected", "separated"],
        "wl_cumcat": [False, True],
        "hist_norm": [False, True],
        "model": ["SVC"],
        "kernel": ["precomputed"],
        "dtype": ["kernel"],
    }
    Cx100 = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    MORE_HPARAM_SPACE = {
        "C": [c / 100 for c in Cx100],
        "dual": [True, False],  # for SVC
    }

    __args__ = parser.parse_args()

    # __args__.dataset_name = "EMUser"
    # "PPIBP", "HPOMetab", "EMUser", "HPONeuro", "Density", "Component", "Coreness", "CutRatio

    if __args__.MODE == "plot_tsne_all":
        plot_tsne_all(__args__)
    elif __args__.MODE == "run_one":
        run_one(__args__)
    elif __args__.MODE == "hp_search_for_models":
        hp_search_for_models(__args__, HPARAM_SPACE, MORE_HPARAM_SPACE)
