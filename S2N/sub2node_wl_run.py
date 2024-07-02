import torch

from sub2node import SubgraphToNode
from utils import repr_kvs

if __name__ == '__main__':

    from data_sub import HPOMetab, HPONeuro, PPIBP, EMUser, Density, Component, Coreness, CutRatio

    MODE = "EMUser"
    # PPIBP, HPOMetab, HPONeuro, EMUser
    # Density, Component, Coreness, CutRatio
    PURPOSE = "MANY"
    # PRECURSOR, MANY
    TARGET_MATRIX = "wl_kernel"

    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    E_TYPE = "glass"
    DEBUG = False

    if PURPOSE == "PRECURSOR":
        _cls = eval(MODE)
        dts = _cls(root=PATH, name=MODE, debug=DEBUG, embedding_type=E_TYPE)
        _subgraph_data_list = dts.get_data_list_with_split_attr()
        _global_data = dts.global_data

        for _wl_cumcat in [True, False]:
            for _wl_hist_norm in [True, False]:
                s2n = SubgraphToNode(
                    _global_data, _subgraph_data_list,
                    name=MODE,
                    path=f"{PATH}/{MODE.upper()}/sub2node/",
                    undirected=True,
                    splits=dts.splits,
                    target_matrix=TARGET_MATRIX,
                    wl_cumcat=_wl_cumcat,
                    wl_hist_norm=_wl_hist_norm
                )
                s2n.node_task_data_precursor(matrix_type="unnormalized", use_sub_edge_index=True)
                s2n.node_task_data_precursor(matrix_type="unnormalized", use_sub_edge_index=False)

        exit()

    if MODE in ["HPOMetab", "PPIBP", "HPONeuro", "EMUser",
                "Density", "Component", "Coreness", "CutRatio"]:
        _cls = eval(MODE)
        dts = _cls(root=PATH, name=MODE, debug=DEBUG, embedding_type=E_TYPE)
        _subgraph_data_list = dts.get_data_list_with_split_attr()
        _global_data = dts.global_data

        for _wl_cumcat in [True, False]:
            for _wl_hist_norm in [True, False]:
                s2n = SubgraphToNode(
                    _global_data, _subgraph_data_list,
                    name=MODE,
                    path=f"{PATH}/{MODE.upper()}/sub2node/",
                    undirected=True,
                    splits=dts.splits,
                    target_matrix=TARGET_MATRIX,

                    wl_cumcat=_wl_cumcat,
                    wl_hist_norm=_wl_hist_norm,
                )
                print(s2n)

                for i in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                          2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]:
                    for j in [0.5, 1.0, 1.5, 2.0]:
                        for usei in [True, False]:

                            ntds = s2n.node_task_data_splits(
                                mapping_matrix_type="unnormalized",
                                set_sub_x_weight=None,
                                use_sub_edge_index=usei,
                                post_edge_normalize="standardize_then_trunc_thres_max_linear",
                                post_edge_normalize_args=[i, j],
                                edge_thres=0.0,
                                use_consistent_processing=True,
                                save=True,
                            )
                            for _d in ntds:
                                print(_d)
                                print(f"\t- density: {_d.edge_index.size(1) / (_d.num_nodes ** 2)}")
                                if hasattr(_d, "sub_x_weight"):
                                    _sub_x_weight_stats = repr_kvs(
                                        min=torch.min(_d.sub_x_weight), max=torch.max(_d.sub_x_weight),
                                        avg=torch.mean(_d.sub_x_weight), std=torch.std(_d.sub_x_weight), sep=", ")
                                    print(f"\t- sub_x_weight: {_sub_x_weight_stats}")
                            s2n._node_task_data_list = []  # flush
