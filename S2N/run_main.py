import os
from typing import List, Optional, Dict, Union, Any

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from run_utils import get_logger, log_hyperparameters, finish
from utils import make_deterministic_everything

"""Codes are adopted from
    https://github.com/ashleve/lightning-hydra-template/blob/main/src/train.py
    https://github.com/ashleve/lightning-hydra-template/blob/main/run.py"""


log = get_logger(__name__)


def train(config: DictConfig, seed_forced: int = None) -> Dict[str, Any]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        seed_forced: This value will be replaced for config.seed for multiruns.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
        Optional[Dict[str, float]]: Metric score for averaging scores.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        if seed_forced is not None:
            config.seed = seed_forced
        seed_everything(config.seed, workers=True)
        make_deterministic_everything(config.seed)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule, log_func=log.info)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model, given_datamodule=datamodule)
    log.info(model)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                # Lazy load metrics and set the first as monitor.
                if "monitor" in cb_conf and cb_conf.monitor is None:
                    cb_conf.monitor = f"valid/{config.model.metrics[0]}"
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set, using the best model achieved during training
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test(dataloaders=datamodule.test_dataloader())

    # Make sure everything closed properly
    log.info("Finalizing!")
    finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")
    if config.get("remove_best_model_ckpt") and trainer.checkpoint_callback.best_model_path:
        os.remove(trainer.checkpoint_callback.best_model_path)
        log.info(f"Removed: {trainer.checkpoint_callback.best_model_path}")

    return {trainer.checkpoint_callback.monitor: trainer.checkpoint_callback.best_model_score.cpu(),
            **trainer.callback_metrics}


@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    import run_utils
    import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    run_utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        run_utils.print_config(config, resolve=True)

    # Train model, if num_averaging is given, run multiple `train`.
    num_averaging: Optional[int] = config.get("num_averaging", default_value=1)
    seed = config.seed
    trained_metrics = []
    for run_no in range(num_averaging):
        log.info(f"Running experiment {run_no + 1} out of {num_averaging}")
        seed_forced = (seed + run_no) if seed is not None else seed
        trained_metrics.append(train(config, seed_forced=seed_forced))
    trained_metrics = utils.ld_to_dl(trained_metrics)

    # Log the summary
    log.info("--- Summary ({} runs) ---".format(num_averaging))
    for m, vs in trained_metrics.items():
        if m.startswith("test"):
            log.info("{}: {:.5f} +- {:.5f}".format(m, *utils.mean_std(vs)))

    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        om, _ = utils.mean_std(trained_metrics[optimized_metric])
        log.info("Return {}: {:.5f}".format(optimized_metric, om))
        return om


if __name__ == "__main__":
    main()
