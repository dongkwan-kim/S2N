# Subgraph-To-Node (S2N) Translation

Official implementation of Subgraph-To-Node (S2N) Translation from ["Translating Subgraphs to Nodes Makes Simple GNNs
Strong and Efficient for Subgraph Representation Learning"](https://arxiv.org/abs/2204.04510v3), International
Conference on Machine Learning (ICML 2024)

## BibTeX

```
@inproceedings{kim2024translating,
    title={Translating Subgraphs to Nodes Makes Simple GNNs Strong and Efficient for Subgraph Representation Learning},
    author={Kim, Dongkwan and Oh, Alice},
    booktitle={International Conference on Machine Learning},
    year={2024},
    organization={PMLR}
}
```

## Install

```bash
bash install.sh {CUDA:cu102,cu111} {TORCH}

# default values: CUDA=cu102 TORCH=1.9.0
bash install.sh 
```

This repository has been confirmed to be working on `nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04`
and `nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04`

## Run

- `${model}`: `gcn, gcn2`
- `${dataset}`: `em_user, hpo_metab, hpo_neuro, ppi_bp`
- `${batching_type}`: `s2n (S2N+0), sub_s2n (S2N+A), s2n_co (CoS2N+0), sub_s2n_co (CoS2N+A), separated, connected`

```shell
# Print args: --cfg all
python run_main.py datamodule=${batching_type}/${dataset} model=${model}/${batching_type}/for-${dataset} --cfg all
python run_main.py datamodule=s2n/${dataset}/for-${model} model=${model}/s2n/for-${dataset} --cfg all

# For individual experiments (separated, connected)
python run_main.py trainer.gpus="[0]" datamodule=${batching_type}/${dataset} model=${model}/${batching_type}/for-${dataset}
# For individual experiments (s2n)
python run_main.py trainer.gpus="[0]" datamodule=s2n/${dataset}/for-${model} model=${model}/s2n/for-${dataset}
# For individual experiments (sub_s2n)
python run_main.py trainer.gpus="[0]" datamodule=sub_s2n/${dataset}/for-${model} model=${model}/sub_s2n/for-${dataset}

# For hparams tuning
python run_main.py --multirun hparams_search=optuna_as_is trainer.gpus="[1]" datamodule=${batching_type}/${dataset} model=${model}/${batching_type}/for-${dataset}
python run_main.py --multirun hparams_search=optuna_s2n trainer.gpus="[1]" datamodule=s2n/${dataset}/for-${model} model=${model}/s2n/for-${dataset}
# Examples (s2n)
python run_main.py --multirun hparams_search=optuna_s2n_lr trainer.gpus="[0]" datamodule=s2n/ppi_bp/for-gcn model=gcn/s2n/for-ppi_bp
python run_main.py --multirun hparams_search=optuna_s2n_gcn2_lr trainer.gpus="[0]" datamodule=s2n/ppi_bp/for-gcn2 model=gcn2/s2n/for-ppi_bp
python run_main.py --multirun hparams_search=optuna_s2n_fa_lr trainer.gpus="[0]" datamodule=s2n/ppi_bp/for-fa model=fa/s2n/for-ppi_bp
# Examples (others)
python run_main.py --multirun hparams_search=optuna_as_is_lr trainer.gpus="[1]" datamodule=connected/ppi_bp model=gcn/connected/for-ppi_bp
python run_main.py --multirun hparams_search=optuna_s2n_gcn2_lr trainer.gpus="[1]" datamodule=connected/ppi_bp model=gcn2/connected/for-ppi_bp
python run_main.py --multirun hparams_search=optuna_s2n_fa_lr trainer.gpus="[1]" datamodule=connected/ppi_bp model=fa/connected/for-ppi_bp

# Examples of ablation studies by the ratio of training subgraphs ([num_start, num_train, num_val]):
python run_main.py trainer.gpus="[2]" datamodule=connected/ppi_bp model=gcn/connected/for-ppi_bp datamodule.custom_splits="[0.7, 0.1, 0.1]"
# Examples of ablation studies by the number of training subgraphs ([num_train_per_class]):
python run_main.py trainer.gpus="[2]" datamodule=connected/ppi_bp model=gcn/connected/for-ppi_bp datamodule.custom_splits="[5]"
python run_main.py trainer.gpus='[3]' datamodule=s2n_co/em_user/for-gcn2 model=gcn2/s2n_co/for-em_user datamodule.custom_splits='[10]'
python run_main.py trainer.gpus='[3]' datamodule=sub_s2n_co/em_user/for-gcn2 model=gcn2/sub_s2n_co/for-em_user datamodule.custom_splits='[20]'

# To compute time & memory: callbacks=efficiency
python run_main.py datamodule=${batching_type}/${dataset} model=${model}/${batching_type}/for-${dataset} callbacks=efficiency
python run_main.py datamodule=s2n/${dataset}/for-${model} model=${model}/s2n/for-${dataset} callbacks=efficiency
python run_main.py trainer.gpus="[2]" datamodule=connected/ppi_bp model=gcn/connected/for-ppi_bp callbacks=efficiency
python run_main.py trainer.gpus="[2]" datamodule=s2n/ppi_bp/for-gcn model=gcn/s2n/for-ppi_bp callbacks=efficiency
python run_main.py trainer.gpus="[2]" datamodule=connected/ppi_bp model=gcn2/connected/for-ppi_bp callbacks=efficiency datamodule.custom_splits="[0.7, 0.1, 0.1]"
python run_main.py trainer.gpus='[3]' datamodule=sub_s2n_co/em_user/for-gcn2 model=gcn2/sub_s2n_co/for-em_user callbacks=efficiency datamodule.custom_splits='[5]'
```
