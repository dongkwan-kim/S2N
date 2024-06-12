import argparse
import inspect
import itertools
import warnings
from pathlib import Path
from pprint import pprint
from typing import Union

import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from termcolor import cprint
from torch_geometric.data import Batch
from torch_geometric.nn.conv import WLConv

from data_sub import HPONeuro, PPIBP, HPOMetab, EMUser, Density, Component, Coreness, CutRatio

ModelType = Union[MultiOutputClassifier, RandomForestClassifier, LogisticRegression, LinearSVC, AdaBoostClassifier]

warnings.filterwarnings('ignore', category=ConvergenceWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--MODE', type=str, default="hp_search_for_models")
parser.add_argument('--dataset_name', type=str, default="PPIBP",
                    choices=["PPIBP", "HPOMetab", "EMUser", "HPONeuro", "Density", "Component", "Coreness", "CutRatio"])
parser.add_argument('--stype', type=str, default="connected", choices=["connected", "separated"])
parser.add_argument('--wl_layers', type=int, default=5)
parser.add_argument('--model', type=str, default="LogisticRegression")
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--dataset_path', type=str, default="/mnt/nas2/GNN-DATA/SUBGRAPH")


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
        "best_val_f1_mean": float(best_val_f1s.mean()),
        "test_f1_mean": float(test_f1s.mean()),
        "best_wl_list": best_wl.tolist(),
    }


def run_one(args):
    hists, splits, all_data = get_data_and_model(args)
    experiment(args, hists, splits, all_data)


def hp_search_for_models(args, file_dir="../_logs_wl4s"):
    HPARAM_SPACE = {
        "stype": ["connected", "separated"],
        "model": ["AdaBoostClassifier", "RandomForestClassifier", "LogisticRegression", "LinearSVC"],
    }
    MORE_HPARAM_SPACE = {
        "n_estimators": [10, 100, 200, 400],
        "max_depth": [None, 5, 10, 20],
        "C": [0.2, 0.4, 0.6, 0.8, 1.0] + [2.0, 4.0, 6.0, 8.0, 10.0] + [20.0, 40.0],
    }

    def space_to_kwl(space):
        return [dict(zip(space.keys(), cmb)) for cmb in itertools.product(*space.values())]

    def init_args(method_name):
        return [param.name for param in inspect.signature(eval(method_name)).parameters.values()]

    kwargs_list = []
    for kwargs in space_to_kwl(HPARAM_SPACE):
        more_space = {k: MORE_HPARAM_SPACE[k] for k in MORE_HPARAM_SPACE if k in init_args(kwargs["model"])}
        more_kwl = space_to_kwl(more_space)
        new_kwl = [{**kwargs, **mkw} for mkw in more_kwl]
        kwargs_list += new_kwl

    file_path = Path(file_dir) / f"{args.dataset_name}.csv"
    hists, splits, all_data = get_data_and_model(args)
    results_dict_list = []
    for i, model_kwargs in enumerate(kwargs_list):
        print(model_kwargs)
        for k in model_kwargs.copy():
            if k in args.__dict__:
                setattr(args, k, model_kwargs.pop(k))
        results = experiment(args, hists, splits, all_data, **model_kwargs)
        results_dict_list.append({**results, **args.__dict__, **model_kwargs})
        pprint(results_dict_list)

        pd.DataFrame(results_dict_list).to_csv(file_path, index=False)
        cprint(f"Save logs at {file_path}", "blue")


if __name__ == '__main__':
    __args__ = parser.parse_args()

    if __args__.MODE == "run_one":
        run_one(__args__)
    elif __args__.MODE == "hp_search_for_models":
        hp_search_for_models(__args__)
