import csv
from pathlib import Path
from types import BuiltinFunctionType
from typing import Dict, List, Any

from omegaconf import OmegaConf, DictConfig, ListConfig
from termcolor import cprint


def _eval(o, *args, **kwargs):
    try:
        _o = eval(o)
    except:
        _o = o
    if isinstance(_o, BuiltinFunctionType):
        return o
    else:
        return _o


def load_kwargs(dataset_name, batching_type, model_name) -> (str, DictConfig, str, DictConfig):
    assert model_name in ["fa", "gat", "gcn", "gcn2", "gin", "linkx", "sage"]
    dataset_name = {
        "PPIBP": "ppi_bp",
        "HPONeuro": "hpo_neuro",
        "HPOMetab": "hpo_metab",
        "EMUser": "em_user"
    }[dataset_name]
    if batching_type == "s2n":
        datamodule_yaml = f"../configs/datamodule/s2n/{dataset_name}/for-{model_name}.yaml"
    else:
        datamodule_yaml = f"../configs/datamodule/{batching_type}/{dataset_name}.yaml"

    model_yaml = f"../configs/model/{model_name}/{batching_type}/for-{dataset_name}.yaml"

    datamodule_cfg = OmegaConf.load(datamodule_yaml)
    model_cfg = OmegaConf.load(model_yaml)
    return datamodule_yaml, datamodule_cfg, model_yaml, model_cfg


def load_best_hparams_per_dataset(dataset_name, log_path="../_aggr_logs") -> List[Dict[str, Any]]:
    the_log = list(Path(log_path).glob(f"_log_{dataset_name}_*"))
    assert len(the_log) == 1
    the_log = the_log[0]

    hparams_list = []
    for line in csv.DictReader(the_log.open()):
        if line["best_of_model"]:
            hparams = {"datamodule": {}, "model": {}}
            for k, v in line.items():
                if k.startswith("datamodule") or k.startswith("model"):
                    k1, k2 = k.split("/")
                    hparams[k1][k2] = _eval(v)
            hparams_list.append(hparams)
    return hparams_list


def replace_and_dump_hparams_to_args(dataset_name):
    best_hparams = load_best_hparams_per_dataset(dataset_name)
    for bh_dict in best_hparams:
        bh_datamodule, bh_model = bh_dict["datamodule"], bh_dict["model"]

        # NOTE: hard-coded
        batching_type = "s2n" if bh_datamodule["subgraph_batching"] == "" else bh_datamodule["subgraph_batching"]
        model_name = {
            "FA": "fa",
            "GAT": "gat",
            "GCN": "gcn",
            "GCNII": "gcn2",
            "GIN": "gin",
            "ILINKX": "linkx",
            "SAGE": "sage",
        }[bh_model["subname"].split("-")[0]]

        datamodule_yaml, datamodule_cfg, model_yaml, model_cfg = load_kwargs(
            dataset_name, batching_type=batching_type, model_name=model_name,
        )

        for k in model_cfg:
            if k in bh_model and bh_model[k] != "":
                # NOTE: hard-coded
                if k == "num_layers" and isinstance(model_cfg[k], ListConfig):
                    model_cfg[k][-1] = bh_model[k]
                elif k == "subname":
                    pass
                else:
                    model_cfg[k] = bh_model[k]

        for k in datamodule_cfg:
            if k in bh_datamodule and bh_datamodule[k] != "":
                datamodule_cfg[k] = bh_datamodule[k]

        # NOTE: uncomment these lines to dump datamodule args.
        # with open(datamodule_yaml, "w") as f:
        #     OmegaConf.save(datamodule_cfg, f)
        #     cprint(f"Saved: {datamodule_yaml}", "blue")

        with open(model_yaml, "w") as f:
            OmegaConf.save(model_cfg, f)
            cprint(f"Saved: {model_yaml}", "blue")


if __name__ == '__main__':
    replace_and_dump_hparams_to_args("PPIBP")
    replace_and_dump_hparams_to_args("HPOMetab")
    replace_and_dump_hparams_to_args("HPONeuro")
    replace_and_dump_hparams_to_args("EMUser")
