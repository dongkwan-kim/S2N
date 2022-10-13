# sub2node
Official implementation of Subgraph-To-Node (S2N) Translation

## Run

- `${model}`: `fa, gat, gcn, gcn2, linkx, mlp`
- `${dataset}`: `em_user, hpo_metab, hpo_neuro, ppi_bp, wlks`
- `${batching_type}`: `s2n, separated, connected`

```shell
# Print args (--cfg all)
python run_main.py datamodule=${batching_type}/${dataset} model=${model}/${batching_type}/for-${dataset} --cfg all
python run_main.py datamodule=s2n/${dataset}/for-${model} model=${model}/s2n/for-${dataset} --cfg all

# For individual experiments (separated, connected)
python run_main.py trainer.gpus="[0]" datamodule=${batching_type}/${dataset} model=${model}/${batching_type}/for-${dataset}
# For individual experiments (s2n)
python run_main.py trainer.gpus="[0]" datamodule=s2n/${dataset}/for-${model} model=${model}/s2n/for-${dataset}

# For hparams tuning
python run_main.py --multirun hparams_search=optuna_as_is trainer.gpus="[1]" datamodule=${batching_type}/${dataset} model=${model}/${batching_type}/for-${dataset}
python run_main.py --multirun hparams_search=optuna_s2n trainer.gpus="[1]" datamodule=s2n/${dataset}/for-${model} model=${model}/s2n/for-${dataset}

# Examples of WLKS
python run_main.py --multirun hparams_search=optuna_as_is trainer.gpus="[1]" datamodule=separated/wlks model=gcn/separated/for-wlks datamodule.transform_args="[0]"
python run_main.py --multirun hparams_search=optuna_s2n trainer.gpus="[1]" datamodule=s2n/wlks model=gcn/s2n/for-wlks datamodule.s2n_transform_args="[0]"

# Example by a number of training subgraphs
python run_main.py trainer.gpus="[2]" datamodule=connected/ppi_bp model=gcn/connected/for-ppi_bp datamodule.custom_splits="[0.7, 0.1, 0.1]"
```
