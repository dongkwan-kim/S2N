import argparse
import inspect
import itertools
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from termcolor import cprint
from torch_geometric.data import Batch
from torch_geometric.nn.conv import WLConv
from tqdm import tqdm

from data_sub import HPONeuro, PPIBP, HPOMetab, EMUser, Density, Component, Coreness, CutRatio
from data_transform import KHopSubgraph
from utils import str2bool
from utils_fscache import fscaches
from visualize import plot_data_points_by_tsne

ModelType = Union[MultiOutputClassifier, LinearSVC]

DATASETS_REAL = ["PPIBP", "EMUser", "HPOMetab", "HPONeuro"]
DATASETS_SYN = ["Component", "Density", "Coreness", "CutRatio"]

parser = argparse.ArgumentParser()
parser.add_argument('--MODE', type=str, default="hp_search_for_models")
parser.add_argument('--dataset_name', type=str, default="PPIBP",
                    choices=["PPIBP", "HPOMetab", "EMUser", "HPONeuro", "Density", "Component", "Coreness", "CutRatio"])
parser.add_argument('--stype', type=str, default="connected", choices=["connected", "separated"])
parser.add_argument('--wl_layers', type=int, default=5)
parser.add_argument('--wl_cumcat', type=str2bool, default=False)
parser.add_argument('--hist_norm', type=str2bool, default=True)
parser.add_argument('--hist_indices', type=str, default="None")
parser.add_argument('--cache_path', type=str, default=None)
parser.add_argument('--k_to_sample', type=int, default=None)
parser.add_argument('--model', type=str, default="LogisticRegression")
parser.add_argument('--runs', type=int, default=2)
parser.add_argument('--dataset_path', type=str, default="/mnt/nas2/GNN-DATA/SUBGRAPH")


class WL4S(torch.nn.Module):
    def __init__(self, stype, num_layers, norm, hist_indices=None, cache_path=None):
        super(WL4S, self).__init__()
        self.stype = stype
        self.norm = norm
        self.hist_indices = hist_indices
        self.cache_path = cache_path
        self.convs = torch.nn.ModuleList([WLConv() for _ in range(num_layers)])

    @staticmethod
    @fscaches(path_attrname_in_kwargs="path", keys_to_exclude=["x", "edge_index"], verbose=True)
    def cache_conv(path, conv, x, edge_index, N, E, stype, norm, i):
        return conv(x, edge_index)

    def forward(self, x, edge_index, batch_or_sub_batch, x_to_xs=None, mask=None):
        hists = []
        for i, conv in enumerate(tqdm(self.convs, desc="WL4S.forward")):
            if self.cache_path is None:
                x = conv(x, edge_index)
            else:
                x = self.cache_conv(path=self.cache_path, conv=conv, x=x, edge_index=edge_index,
                                    N=x.size(0), E=edge_index.size(1), stype=self.stype, norm=self.norm, i=i)
            h = None
            if self.hist_indices is None or (i in self.hist_indices):
                cprint(f"Compute histogram: {i} | hist_indices={self.hist_indices}", "yellow")
                if self.stype == "connected":
                    h = conv.histogram(x[x_to_xs], batch_or_sub_batch, norm=self.norm)
                elif self.stype == "separated":
                    _x, _b = x, batch_or_sub_batch
                    if mask is not None:
                        _x, _b = x[mask], batch_or_sub_batch[mask]
                    h = conv.histogram(_x, _b, norm=self.norm)
                else:
                    raise ValueError
            hists.append(h)
        return hists


def get_data_and_model(args):
    dts: Union[HPONeuro, PPIBP, HPOMetab, EMUser, Density, Component, Coreness, CutRatio] = eval(args.dataset_name)(
        root=args.dataset_path,
        name=args.dataset_name,
        embedding_type="glass",
        debug=False,
    )
    # dts.print_summary()
    splits = [0] + dts.splits + [len(dts)]
    if args.k_to_sample is None:
        train_dts, val_dts, test_dts = dts.get_train_val_test_with_individual_relabeling()
    else:
        khs = KHopSubgraph(dts.global_data.edge_index, k=args.k_to_sample,
                           relabel_nodes=True, num_nodes=dts.global_data.num_nodes)
        train_dts, val_dts, test_dts = dts.get_train_val_test()
        train_dts, val_dts, test_dts = khs.map_list([train_dts, val_dts, test_dts])
    all_data = Batch.from_data_list(train_dts + val_dts + test_dts)

    wl = WL4S(stype=args.stype, num_layers=args.wl_layers, norm=args.hist_norm,
              hist_indices=eval(args.hist_indices), cache_path=args.cache_path)

    if args.stype == "connected":
        dts.global_data.x = torch.ones((dts.global_data.x.size(0), 1)).long()
        data = dts.global_data
        hists = wl(data.x, data.edge_index, batch_or_sub_batch=all_data.batch, x_to_xs=all_data.x.flatten())
    else:  # separated
        all_data.x = torch.ones((all_data.x.size(0), 1)).long()
        data = all_data
        hists = wl(data.x, data.edge_index, batch_or_sub_batch=all_data.batch, mask=getattr(data, "mask", None))
    return hists, splits, all_data


def experiment(args, hists, splits, all_data, **model_kwargs):
    s = splits
    test_f1s = torch.zeros(args.runs, dtype=torch.float)
    best_val_f1s = torch.zeros(args.runs, dtype=torch.float)
    best_wl = torch.zeros(args.runs, dtype=torch.float)

    if args.wl_cumcat:
        cumcat_list, cumcat_h = [], torch.tensor([])
        for h in hists:
            cumcat_h = torch.cat((cumcat_h, h), dim=-1)
            cumcat_list.append(cumcat_h)
        hists = cumcat_list

    for run in range(1, args.runs + 1):
        best_val_f1s[run - 1] = 0

        for i_wl, hist in enumerate(hists):
            if hist is None:
                continue
            train_hist, val_hist, test_hist = hist[s[0]:s[1]], hist[s[1]:s[2]], hist[s[2]:s[3]]
            train_y, val_y, test_y = all_data.y[s[0]:s[1]], all_data.y[s[1]:s[2]], all_data.y[s[2]:s[3]]

            model: ModelType = eval(args.model)(**model_kwargs)

            if args.dataset_name == "HPONeuro":
                model = MultiOutputClassifier(model)

            model.fit(train_hist, train_y)

            val_f1 = f1_score(val_y, model.predict(val_hist), average="micro")
            test_f1 = f1_score(test_y, model.predict(test_hist), average="micro")

            print(f"\t - wl: {i_wl + 1}, shape: {list(train_hist.shape)}, model: {args.model}, "
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


def run_one(args, data_func=get_data_and_model):
    hists, splits, all_data = data_func(args)
    experiment(args, hists, splits, all_data)


def plot_tsne_all(args, data_func=get_data_and_model, path="../_figures", extension="png"):
    kws = dict(path=path, extension=extension, alpha=0.5, s=5)
    for dataset_name in DATASETS_SYN + DATASETS_REAL:
        for stype in ["connected", "separated"]:
            args.dataset_name, args.stype = dataset_name, stype
            hists, splits, all_data = data_func(args)
            s = splits
            for i_wl, hist in enumerate(hists):
                sub_key = f"{dataset_name}-{stype}, WL: {i_wl + 1}"
                print(sub_key)
                train_hist, val_hist, test_hist = hist[s[0]:s[1]], hist[s[1]:s[2]], hist[s[2]:s[3]]
                train_y, val_y, test_y = all_data.y[s[0]:s[1]], all_data.y[s[1]:s[2]], all_data.y[s[2]:s[3]]
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

    file_path = Path(file_dir) / f"{args.dataset_name}{log_postfix}.csv"
    for i, model_kwargs in enumerate(kwargs_list):
        print(model_kwargs)
        for k in model_kwargs.copy():
            if k in args.__dict__:
                setattr(args, k, model_kwargs.pop(k))
        hists, splits, all_data = stype_and_norm_to_data_and_model[(args.stype, args.hist_norm)]
        results = experiment(args, hists, splits, all_data, **model_kwargs)

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
        "wl_cumcat": [True, False],
        "hist_norm": [False, True],
        "model": ["LinearSVC"],
    }
    Cx100 = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    MORE_HPARAM_SPACE = {
        "C": [c / 100 for c in Cx100],
        "dual": [True, False],
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
    elif __args__.MODE == "hp_search_real":
        hp_search_real(__args__, HPARAM_SPACE, MORE_HPARAM_SPACE)
    elif __args__.MODE == "hp_search_syn":
        hp_search_syn(__args__, HPARAM_SPACE, MORE_HPARAM_SPACE)
