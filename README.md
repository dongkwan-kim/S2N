# sub2node
Official implementation of Subgraph-To-Node (S2N) Translation

## Run

- `${model}`: `fa, gat, gcn, gcn2, linkx, mlp`
- `${dataset}`: `em_user, hpo_metab, hpo_neuro, ppi_bp, wl_ba`
- `${batching_type}`: `s2n, separated, connected`

```shell
# For individual experiments
python run_main.py trainer.gpus="[0]" datamodule=${batching_type}/${dataset} model=${model}/${batching_type}/for-${dataset}

# For hparams tuning
python run_main.py --multirun hparams_search=sgn_optuna_as_is trainer.gpus="[1]" datamodule=${batching_type}/${dataset} model=${model}/${batching_type}/for-${dataset}

# Example
python run_main.py --multirun hparams_search=sgn_optuna_as_is trainer.gpus="[1]" datamodule=separated/wl_ba model=gcn/separated/for-wl_ba datamodule.transform_args="[0]"
python run_main.py --multirun hparams_search=sgn_optuna_s2n trainer.gpus="[1]" datamodule=s2n/wl_ba model=gcn/s2n/for-wl_ba datamodule.s2n_transform_args="[0]"
```
