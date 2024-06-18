import argparse
import inspect
import itertools
from pathlib import Path
from pprint import pprint
from typing import Union

import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from termcolor import cprint
from torch_geometric.data import Batch
from torch_geometric.nn.conv import WLConv

from data_sub import HPONeuro, PPIBP, HPOMetab, EMUser, Density, Component, Coreness, CutRatio
from visualize import plot_data_points_by_tsne

ModelType = Union[
    MultiOutputClassifier, RandomForestClassifier, LogisticRegression, LinearSVC, AdaBoostClassifier,
    MLPClassifier, KNeighborsClassifier]

DATASETS_REAL = ["PPIBP", "EMUser", "HPOMetab", "HPONeuro"]
DATASETS_SYN = ["Component", "Density", "Coreness", "CutRatio"]

parser = argparse.ArgumentParser()
parser.add_argument('--MODE', type=str, default="hp_search_for_models")
parser.add_argument('--dataset_name', type=str, default="PPIBP",
                    choices=["PPIBP", "HPOMetab", "EMUser", "HPONeuro", "Density", "Component", "Coreness", "CutRatio"])
parser.add_argument('--stype', type=str, default="connected", choices=["connected", "separated"])
parser.add_argument('--wl_layers', type=int, default=5)
parser.add_argument('--model', type=str, default="LogisticRegression")
parser.add_argument('--runs', type=int, default=3)
parser.add_argument('--dataset_path', type=str, default="/mnt/nas2/GNN-DATA/SUBGRAPH")


class MyMLPClassifier(MLPClassifier):

    def __init__(self, C, hidden_layer_sizes, learning_rate_init, **kwargs):
        kwargs["alpha"] = 1.0 / C
        kwargs["hidden_layer_sizes"] = hidden_layer_sizes
        kwargs["learning_rate_init"] = learning_rate_init
        kwargs["solver"] = "sgd"
        super().__init__(**kwargs)


class WL4S(torch.nn.Module):
    def __init__(self, stype, num_layers, norm):
        super(WL4S, self).__init__()
        self.stype = stype
        self.norm = norm
        self.convs = torch.nn.ModuleList([WLConv() for _ in range(num_layers)])

    def forward(self, x, edge_index, batch_or_sub_batch, x_to_xs=None):
        hists = []
        for conv in self.convs:
            x = conv(x, edge_index)
            if self.stype == "connected":
                h = conv.histogram(x[x_to_xs], batch_or_sub_batch, norm=self.norm)
            elif self.stype == "separated":
                h = conv.histogram(x, batch_or_sub_batch, norm=self.norm)
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
    train_dts, val_dts, test_dts = dts.get_train_val_test_with_individual_relabeling()
    all_data = Batch.from_data_list(train_dts + val_dts + test_dts)

    wl = WL4S(stype=args.stype, num_layers=args.wl_layers, norm=True)

    if args.stype == "connected":
        dts.global_data.x = torch.ones((dts.global_data.x.size(0), 1)).long()
        data = dts.global_data
        hists = wl(data.x, data.edge_index, batch_or_sub_batch=all_data.batch, x_to_xs=all_data.x.flatten())
    else:  # separated
        all_data.x = torch.ones((all_data.x.size(0), 1)).long()
        data = all_data
        hists = wl(data.x, data.edge_index, batch_or_sub_batch=all_data.batch)
    return hists, splits, all_data


def experiment(args, hists, splits, all_data, **model_kwargs):
    s = splits
    test_f1s = torch.zeros(args.runs, dtype=torch.float)
    best_val_f1s = torch.zeros(args.runs, dtype=torch.float)
    best_wl = torch.zeros(args.runs, dtype=torch.float)

    for run in range(1, args.runs + 1):
        best_val_f1s[run - 1] = 0

        for i_wl, hist in enumerate(hists):
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


def hp_search_for_models(args, hparam_space, more_hparam_space, data_func=get_data_and_model, file_dir="../_logs_wl4s"):
    def space_to_kwl(space):
        return [dict(zip(space.keys(), cmb)) for cmb in itertools.product(*space.values())]

    def init_args(method_name):
        return [param.name for param in inspect.signature(eval(method_name)).parameters.values()]

    kwargs_list = []
    for kwargs in space_to_kwl(hparam_space):
        more_space = {k: more_hparam_space[k] for k in more_hparam_space if k in init_args(kwargs["model"])}
        more_kwl = space_to_kwl(more_space)
        new_kwl = [{**kwargs, **mkw} for mkw in more_kwl]
        kwargs_list += new_kwl

    stype_to_data_and_model = {}
    for stype in hparam_space["stype"]:
        args.stype = stype
        stype_to_data_and_model[stype] = data_func(args)

    file_path = Path(file_dir) / f"{args.dataset_name}.csv"
    results_dict_list = []
    for i, model_kwargs in enumerate(kwargs_list):
        print(model_kwargs)
        for k in model_kwargs.copy():
            if k in args.__dict__:
                setattr(args, k, model_kwargs.pop(k))
        hists, splits, all_data = stype_to_data_and_model[args.stype]
        results = experiment(args, hists, splits, all_data, **model_kwargs)
        results_dict_list.append({**results, **args.__dict__, **model_kwargs})
        pprint(results_dict_list)

        df = pd.DataFrame(results_dict_list)
        if file_path.is_file():
            df = pd.concat([pd.read_csv(file_path), df], ignore_index=True)
        df.to_csv(file_path, index=False)
        cprint(f"Save logs at {file_path}", "blue")


def hp_search_real(args, hparam_space, more_hparam_space, data_func=get_data_and_model, file_dir="../_logs_wl4s"):
    for dataset_name in DATASETS_REAL:
        args.dataset_name = dataset_name
        hp_search_for_models(args, hparam_space, more_hparam_space, data_func, file_dir)


def hp_search_syn(args, hparam_space, more_hparam_space, data_func=get_data_and_model, file_dir="../_logs_wl4s"):
    for dataset_name in DATASETS_SYN:
        args.dataset_name = dataset_name
        hp_search_for_models(args, hparam_space, more_hparam_space, data_func, file_dir)


if __name__ == '__main__':

    HPARAM_SPACE = {
        "stype": ["connected", "separated"],
        "model": ["MyMLPClassifier", "LogisticRegression", "LinearSVC"],
    }
    Cx100 = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    MORE_HPARAM_SPACE = {
        "C": [c / 100 for c in Cx100],
        "hidden_layer_sizes": [(20,), (40,), (80,), (160,)],
        "learning_rate_init": [0.01, 0.001, 0.0001],
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
