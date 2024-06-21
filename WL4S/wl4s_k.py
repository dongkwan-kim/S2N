import gc

from wl4s import parser, hp_search_for_models, precompute_all_kernels

if __name__ == '__main__':
    HPARAM_SPACE = {
        "stype": ["separated"],
        "wl_cumcat": [False],
        "hist_norm": [False, True],
        "model": ["SVC"],
        "kernel": ["precomputed"],
        "dtype": ["kernel"],
    }
    Cx100 = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    MORE_HPARAM_SPACE = {
        "C": [c / 100 for c in Cx100],
    }

    __args__ = parser.parse_args()
    __args__.stype = "separated"
    __args__.dtype = "kernel"
    __args__.wl_cumcat = False

    MODE = "real"  # syn, real

    if MODE == "syn":
        for k_to_sample in [1, 2, 3, 4]:
            __args__.k_to_sample = k_to_sample
            kws = dict(file_dir="../_logs_wl4s_k", log_postfix=f"_{k_to_sample}")
            for dataset_name in ["Component", "Density", "Coreness", "CutRatio"]:
                __args__.dataset_name = dataset_name
                hp_search_for_models(__args__, HPARAM_SPACE, MORE_HPARAM_SPACE, **kws)

    elif MODE == "real":
        for k_to_sample in [0, 1, 2, 3, 4]:
            for dataset_name in ["PPIBP", "EMUser"]:
                __args__.k_to_sample = k_to_sample
                __args__.dataset_name = dataset_name
                kws = dict(file_dir="../_logs_wl4s_k", log_postfix=f"_{k_to_sample}")
                for hist_norm in [False, True]:
                    __args__.hist_norm = hist_norm
                    precompute_all_kernels(__args__)

                hp_search_for_models(__args__, HPARAM_SPACE, MORE_HPARAM_SPACE, **kws)
                gc.collect()
