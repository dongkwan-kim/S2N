# sub2node
Official implementation of Subgraph-To-Node (S2N) Translation

## Run

- `${model}`: `fa_s2n, gat_s2n, gcn_s2n, linkx_s2n`
- `${dataset}`: `em_user, hpo_metab, hpo_neuro, ppi_bp`

```shell
# For individual experiments
python run_main.py trainer.gpus="[0]" datamodule=${dataset} model=${model}-${dataset}

# For hparams tuning
python run_main.py --multirun hparams_search=sgn_optuna \
    trainer.gpus="[0]" datamodule=${dataset} model=${model}-${dataset}
```
