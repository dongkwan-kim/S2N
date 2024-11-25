# WL Kernel for Subgraphs (WLKS)

## Install

This repository has been confirmed to be working on `nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04`
and `nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04`. GPUs are not required.

```bash
bash install.sh
```

## Datasets

Set `--dataset_path`.

Dataset files (`raw`) can be downloaded from https://github.com/mims-harvard/SubGNN.
Additionally,  `raw/glass_embeddings.pth ` can be downloaded from https://github.com/Xi-yuanWang/GLASS/tree/main/Emb.
Then,  `SubgraphDataset.process` automatically generates `processed_{dataset}_42_False_undirected` folder.

```
ls /mnt/nas2/GNN-DATA/SUBGRAPH
COMPONENT  CORENESS  CUTRATIO  DENSITY  EMUSER  HPOMETAB  HPONEURO  PPIBP

ls /mnt/nas2/GNN-DATA/SUBGRAPH/PPIBP
processed_PPIBP_42_False_undirected  raw

ls  /mnt/nas2/GNN-DATA/SUBGRAPH/PPIBP/raw
degree_sequence.txt  edge_list.txt  ego_graphs.txt  gin_embeddings.pth  glass_embeddings.pth  graphsaint_gcn_embeddings.pth  shortest_path_matrix.npy  similarities  subgraphs.pth

ls /mnt/nas2/GNN-DATA/SUBGRAPH/PPIBP/processed_PPIBP_42_False_undirected
args.txt  data.pt  global_gin.pt  global_glass.pt  global_graphsaint_gcn.pt  meta.pt  pre_filter.pt  pre_transform.pt
```

## Run

```bash
# Best HParams
python wl4s2v2.py --MODE run_one --runs 1 --dataset_name EMUser --wl_layers 2 --wl_cumcat False --hist_norm False --a_c 0.9 --a_s 0.1 --C 0.08  
python wl4s2v2.py --MODE run_one --runs 1 --dataset_name HPOMetab --wl_layers 2 --wl_cumcat False --hist_norm True --a_c 0.999 --a_s 0.001 --C 1.28
python wl4s2v2.py --MODE run_one --runs 1 --dataset_name HPONeuro --wl_layers 3 --wl_cumcat False --hist_norm False --a_c 0.999 --a_s 0.001 --C 0.64
python wl4s2v2.py --MODE run_one --runs 1 --dataset_name PPIBP --wl_layers 2 --wl_cumcat False --hist_norm True --a_c 0.99 --a_s 0.01 --C 1.28

# For HPOMetab, search hparams with one run.
python wl4s.py --MODE hp_search_for_models --dataset_name HPOMetab --runs 1
# For PPIBP, k=D (or inf) (connected), precompute kernels, then run SVC (C=1.28) with 2 seeds.
python wl4s.py --MODE run_one --dataset_name PPIBP --stype connected --dtype kernel --wl_cumcat False --hist_norm True --runs 2 --C 1.28
# For EMUser, k=0 (separated), use histogram as features, then run SVC with a linear kernel.
python wl4s.py --MODE run_one --dataset_name EMUser --stype separated --dtype histogram --kernel linear --wl_cumcat False --hist_norm False --runs 3 --C 0.08
 
# For PPIBP, run k=0&D with 0.99 and 0.01 coefficients.
python3 wl4s2v2.py --MODE run_one --dataset_name PPIBP --wl_cumcat False --hist_norm True --a_c 0.99 --a_s 0.01 --runs 3 --C 1.28
# Search hparams for k=0&D
python3 wl4s2v2.py --MODE hp_search_real
python3 wl4s2v2.py --MODE hp_search_syn
python3 wl4s2v2.py --MODE hp_search_for_models --runs 1 --dataset_name PPIBP 
python3 wl4s2v2.py --MODE hp_search_for_models --runs 1 --dataset_name EMUser 
python3 wl4s2v2.py --MODE hp_search_for_models --runs 1 --dataset_name HPOMetab
python3 wl4s2v2.py --MODE hp_search_for_models --runs 1 --dataset_name HPONeuro

# Run wl4s for all k in [0, 1, 2, D]
python3 wl4s_k.py --MODE real_k
python3 wl4s_k.py --MODE syn_k
# Pre-compute kernels for PPIBP and EMUser (If k > 0, intensive time and memory are required)
python3 wl4s_k.py --MODE real_precomputation
```