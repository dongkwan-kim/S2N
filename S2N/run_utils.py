import csv
import logging
import os
import time
import warnings
from collections import defaultdict, OrderedDict
from datetime import datetime
from multiprocessing import Manager
from pathlib import Path
from pprint import pprint
from typing import List, Sequence, Dict

import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from p_tqdm import p_imap
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

from utils import repr_kvs

"""Most of the codes are adopted from
    https://github.com/ashleve/lightning-hydra-template/blob/main/src/utils/utils.py"""


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    use_debug_any = False
    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = use_debug_any = True

    if config.get("debug_test") or config.get("debug_gpu"):
        log.info("Running in debug_* mode! <config.debug_*=True>")
        config.trainer.min_epochs = 1
        config.trainer.max_epochs = 2
        use_debug_any = True

    # force debugger friendly configuration if <use_debug_any=True>
    if use_debug_any:
        log.info("Forcing debugger friendly configuration!")
        if config.get("num_averaging"):
            config.num_averaging = 1
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus") and not config.get("debug_gpu"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
        config: DictConfig,
        fields: Sequence[str] = (
                "trainer",
                "model",
                "datamodule",
                "callbacks",
                "logger",
                "seed",
        ),
        resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dark_green"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Additionally saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    try:
        hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
        hparams["model/params_trainable"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        hparams["model/params_not_trainable"] = sum(
            p.numel() for p in model.parameters() if not p.requires_grad
        )
    except ValueError:
        pass

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def finish(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


def aggregate_csv_metrics(in_path, out_path,
                          key_hparams=None,
                          num_path_hparams: int = 2,
                          metric=None,
                          model_key_hparams=None,
                          min_aggr_sample_counts=10,
                          dump_best_of_model_only=False):
    import yaml
    import pandas as pd
    import numpy as np

    metric = metric or "test/micro_f1"
    assert metric.startswith("test"), f"Wrong metric format: {metric}"
    model_key_hparams = model_key_hparams or [
        "model/subname",
        "datamodule/embedding_type",
        "datamodule/coarsening_method",
        "datamodule/coarsening_ratio",
        "datamodule/s2n_set_sub_x_weight",
    ]
    key_hparams = key_hparams or [
        "datamodule/dataset_subname",
        "datamodule/custom_splits",

        "model/subname",
        "datamodule/embedding_type",
        "datamodule/s2n_set_sub_x_weight",
        "datamodule/coarsening_method",
        "datamodule/coarsening_ratio",
        "datamodule/use_coarsening",

        "datamodule/subgraph_batching",
        "datamodule/use_s2n",
        "datamodule/s2n_mapping_matrix_type",
        "datamodule/s2n_is_weighted",
        "datamodule/s2n_target_matrix",
        "datamodule/s2n_add_sub_x_wl",
        "datamodule/s2n_use_sub_edge_index",
        "datamodule/use_consistent_processing",
        "datamodule/post_edge_normalize",
        "datamodule/post_edge_normalize_arg_1",
        "datamodule/post_edge_normalize_arg_2",
        "model/learning_rate",
        "model/activation",
        "model/encoder_layer_name",
        "model/hidden_channels",
        "model/num_layers",
        "model/use_bn",
        "model/use_skip",
        "model/use_s2n_jk",
        "model/sub_node_encoder_name",
        "model/sub_node_num_layers",
        "model/sub_node_encoder_aggr",
        "model/weight_decay",
        "model/dropout_channels",
        "model/dropout_edges",
        "model/_gradient_clip_val",
        "model/layer_kwargs",
        "model/sub_node_encoder_layer_kwargs",
    ]
    path_hparams = key_hparams[:num_path_hparams]

    in_path = Path(in_path)
    key_to_values = defaultdict(lambda: defaultdict(list))
    key_to_ingredients = dict()

    def get_model_name(ingredients: dict):
        return repr_kvs(**{k: v for k, v in ingredients.items()
                           if k in model_key_hparams})

    def parse_csv_to_k2vi(_path):
        _path = Path(_path)
        yaml_path = _path.parent / "hparams.yaml"
        if not yaml_path.is_file():
            return None

        with open(yaml_path, "r") as stream:
            yaml_data = yaml.safe_load(stream)
            _key_dict = OrderedDict()
            for h in key_hparams:
                try:
                    parsed = h.split("/")
                    yd = yaml_data
                    for p in parsed:
                        yd = yd[p]
                    _key_dict[h] = yd
                except (KeyError, TypeError) as e:
                    pass
            _experiment_key = "+".join(str(s) for s in _key_dict.values())
            _path_key = "+".join(str(v) for k, v in _key_dict.items()
                                 if k in path_hparams)

        try:
            csv_data = pd.read_csv(_path)
            return _path_key, _experiment_key, float(csv_data[metric].tail(1)), _key_dict
        except (KeyError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            # Not finished experiments.
            print(f"Error ({e}) in {_path}")
            return None

    for parsed_csv in p_imap(parse_csv_to_k2vi, in_path.glob("**/*.csv"),
                             desc=f"Reading CSVs from {in_path}",
                             total=len(list(in_path.glob("**/*.csv")))):
        if parsed_csv is None:
            continue
        path_key, experiment_key, metric_value, key_dict = parsed_csv
        key_to_values[path_key][experiment_key].append(metric_value)
        key_to_ingredients[experiment_key] = key_dict

    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)
    for path_key, experiment_key_to_values in key_to_values.items():
        out_file = out_path / f"_log_{path_key}_{datetime.now()}.csv"
        with open(out_file, "w") as f:
            writer = csv.DictWriter(
                f, fieldnames=[
                    "best_of_model",
                    *key_hparams[:num_path_hparams],
                    f"mean/{metric}", f"std/{metric}",
                    f"N/{metric}", f"N_total/{metric}",
                    *key_hparams[num_path_hparams:],
                    "list",
                    "in_path",
                ])
            writer.writeheader()

            model_to_bom_metric = defaultdict(float)
            model_to_n_total = defaultdict(int)
            for experiment_key, values in experiment_key_to_values.items():
                if len(values) >= min_aggr_sample_counts:
                    model_subname = get_model_name(key_to_ingredients[experiment_key])
                    model_to_n_total[model_subname] += 1
                    model_to_bom_metric[model_subname] = max(float(np.mean(values)),
                                                             model_to_bom_metric[model_subname])

            num_lines = 0
            model_to_bom_logged = defaultdict(bool)
            for experiment_key, values in experiment_key_to_values.items():
                if len(values) >= min_aggr_sample_counts:
                    key_dict = key_to_ingredients[experiment_key]
                    model_subname = get_model_name(key_dict)
                    mean_metric = float(np.mean(values))
                    bom = True if (mean_metric == model_to_bom_metric[model_subname]) else ""

                    if dump_best_of_model_only and bom == "":
                        continue

                    writer.writerow({
                        "best_of_model": bom if not model_to_bom_logged[model_subname] else "",
                        **key_dict,
                        f"mean/{metric}": mean_metric,
                        f"std/{metric}": float(np.std(values)),
                        f"N/{metric}": len(values),
                        f"N_total/{metric}": model_to_n_total[model_subname],
                        "list": str(values),
                        "in_path": in_path,
                    })
                    num_lines += 1
                    if bom != "":
                        model_to_bom_logged[get_model_name(key_dict)] = True
            print(f"Saved (lines {num_lines}): {out_file.resolve()}")


if __name__ == '__main__':
    aggregate_csv_metrics(
        # "../_logs_csv_2023/", "../logs_multi_csv", "../_logs_csv_coarsening"
        "../logs_multi_csv/",
        # "../_logs_csv_2023/relu/baseline",
        "../_aggr_logs",
        dump_best_of_model_only=True,  # True, FalseA
    )
