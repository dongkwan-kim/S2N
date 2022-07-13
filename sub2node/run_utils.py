import csv
import logging
import os
import time
import warnings
from collections import defaultdict, OrderedDict
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import List, Sequence, Dict

import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

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
                          path_hparams=None,
                          metric=None,
                          model_key="model/subname"):
    import yaml
    import pandas as pd
    import numpy as np

    metric = metric or "test/micro_f1"
    assert metric.startswith("test"), f"Wrong metric format: {metric}"
    key_hparams = key_hparams or [
        "datamodule/dataset_subname",
        "datamodule/embedding_type",
        "model/subname",
        "datamodule/subgraph_batching",
        "datamodule/use_s2n",
        "datamodule/s2n_is_weighted",
        "datamodule/s2n_target_matrix",
        "datamodule/use_consistent_processing",
        "datamodule/edge_normalize",
        "datamodule/edge_normalize_arg_1",
        "datamodule/edge_normalize_arg_2",
        "model/num_layers",
        "model/sub_node_num_layers",
        "model/sub_node_encoder_aggr",
        "model/weight_decay",
        "model/dropout_channels",
        "model/dropout_edges",
        "model/layer_kwargs",
        "model/use_skip",
        "trainer/gradient_clip_val",
    ]
    path_hparams = path_hparams or key_hparams[:1]

    in_path = Path(in_path)
    key_to_values = defaultdict(lambda: defaultdict(list))
    key_to_ingredients = dict()
    for csv_path in tqdm(in_path.glob("**/*.csv"),
                         desc=f"Reading CSVs from {in_path}",
                         total=len(list(in_path.glob("**/*.csv")))):
        csv_path = Path(csv_path)
        yaml_path = csv_path.parent / "hparams.yaml"

        with open(yaml_path, "r") as stream:
            yaml_data = yaml.safe_load(stream)
            key_dict = OrderedDict()
            for h in key_hparams:
                try:
                    parsed = h.split("/")
                    yd = yaml_data
                    for p in parsed:
                        yd = yd[p]
                    key_dict[h] = yd
                except (KeyError, TypeError) as e:
                    pass
            experiment_key = "+".join(str(s) for s in key_dict.values())
            path_key = "+".join(str(v) for k, v in key_dict.items()
                                if k in path_hparams)

        csv_data = pd.read_csv(csv_path)
        try:
            metric_value = csv_data[metric].tail(1)
            key_to_values[path_key][experiment_key].append(float(metric_value))
            key_to_ingredients[experiment_key] = key_dict
        except KeyError:
            pass

    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)
    for path_key, experiment_key_to_values in key_to_values.items():
        out_file = out_path / f"_log_{path_key}_{datetime.now()}.csv"
        with open(out_file, "w") as f:
            writer = csv.DictWriter(
                f, fieldnames=[
                    "best_of_model",
                    *key_hparams[:2],
                    f"mean/{metric}", f"std/{metric}", f"N/{metric}",
                    *key_hparams[2:],
                    "list",
                ])
            writer.writeheader()

            model_to_bom_metric = defaultdict(float)
            for experiment_key, values in experiment_key_to_values.items():
                model_subname = key_to_ingredients[experiment_key][model_key]
                model_to_bom_metric[model_subname] = max(float(np.mean(values)),
                                                         model_to_bom_metric[model_subname])

            num_lines = 0
            model_to_bom_logged = defaultdict(bool)
            for experiment_key, values in experiment_key_to_values.items():
                key_dict = key_to_ingredients[experiment_key]
                mean_metric = float(np.mean(values))
                bom = True if (mean_metric == model_to_bom_metric[key_dict[model_key]]) else ""
                writer.writerow({
                    "best_of_model": bom if not model_to_bom_logged[key_dict[model_key]] else "",
                    **key_dict,
                    f"mean/{metric}": mean_metric,
                    f"std/{metric}": float(np.std(values)),
                    f"N/{metric}": len(values),
                    "list": str(values),
                })
                num_lines += 1
                if bom != "":
                    model_to_bom_logged[key_dict[model_key]] = True
            print(f"Saved (lines {num_lines}): {out_file.resolve()}")


if __name__ == '__main__':
    aggregate_csv_metrics(
        "../logs_multi_csv",  # e.g., ../logs_multi_csv/WLHistSubgraphBA_5
        "../_aggr_logs",
    )
