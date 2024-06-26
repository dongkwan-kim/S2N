import gc

from wl4s import parser, hp_search_for_models, precompute_all_kernels, hp_search_syn, hp_search_real

if __name__ == '__main__':
    HPARAM_SPACE = {
        "stype": ["separated"],
        "wl_cumcat": [False, True],
        "hist_norm": [False, True],
    }
    Cx100 = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    MORE_HPARAM_SPACE = {
        "C": [c / 100 for c in Cx100],
    }
    DATA_TO_RATIO_SAMPLES = {"HPOMetab": 1400 / 2400, "HPONeuro": 1400 / 4000}  # for sliced

    __args__ = parser.parse_args()

    MODE = __args__.MODE
    if MODE == "syn_k":
        __args__.stype = "separated"
        HPARAM_SPACE = {**HPARAM_SPACE, "model": ["LinearSVC"]}
        MORE_HPARAM_SPACE = {**MORE_HPARAM_SPACE, "dual": [True, False]}
        for k_to_sample in [None, 1, 2]:
            __args__.k_to_sample = k_to_sample
            kws = dict(file_dir="../_logs_wl4s_k", log_postfix=f"_{k_to_sample or 0}")
            hp_search_syn(__args__, HPARAM_SPACE, MORE_HPARAM_SPACE, **kws)

        __args__.stype = "connected"
        HPARAM_SPACE["stype"] = ["connected"]
        kws = dict(file_dir="../_logs_wl4s_k", log_postfix=f"_inf")
        hp_search_syn(__args__, HPARAM_SPACE, MORE_HPARAM_SPACE, **kws)

    else:
        HPARAM_SPACE = {
            **HPARAM_SPACE,
            "model": ["SVC"], "kernel": ["precomputed"], "dtype": ["kernel"],
        }
        __args__.dtype = "kernel"

        if MODE == "real_precomputation":
            for k_to_sample in [None, 1, 2]:
                for dataset_name in ["PPIBP", "EMUser"]:
                    __args__.k_to_sample = k_to_sample
                    __args__.dataset_name = dataset_name
                    for hist_norm in [False, True]:
                        __args__.hist_norm = hist_norm
                        precompute_all_kernels(__args__)
                        gc.collect()

        elif MODE == "sliced_real_precomputation":
            # NOTE: DATA_TO_RATIO_SAMPLES exists
            for k_to_sample in [2, 1, None]:
                for dataset_name in ["HPONeuro", "HPOMetab"]:
                    __args__.k_to_sample = k_to_sample
                    __args__.dataset_name = dataset_name
                    __args__.ratio_samples = DATA_TO_RATIO_SAMPLES[dataset_name]
                    for hist_norm in [False, True]:
                        __args__.hist_norm = hist_norm
                        precompute_all_kernels(__args__)
                        gc.collect()

        elif MODE == "real_k":
            for dataset_name in ["PPIBP", "EMUser", "HPOMetab", "HPONeuro"]:
                k_to_sample_list = [None] if dataset_name in ["HPOMetab", "HPONeuro"] else [None, 1, 2]
                for k_to_sample in k_to_sample_list:
                    __args__.k_to_sample = k_to_sample
                    __args__.dataset_name = dataset_name
                    kws = dict(file_dir="../_logs_wl4s_k", log_postfix=f"_{k_to_sample or 0}")
                    hp_search_for_models(__args__, HPARAM_SPACE, MORE_HPARAM_SPACE, **kws)

            __args__.stype = "connected"
            HPARAM_SPACE["stype"] = ["connected"]
            kws = dict(file_dir="../_logs_wl4s_k", log_postfix=f"_inf")
            hp_search_real(__args__, HPARAM_SPACE, MORE_HPARAM_SPACE, **kws)

        elif MODE == "sliced_real_k":
            # NOTE: DATA_TO_RATIO_SAMPLES exists
            for k_to_sample in [None, 1, 2]:
                for dataset_name in ["HPONeuro", "HPOMetab"]:
                    __args__.k_to_sample = k_to_sample
                    __args__.dataset_name = dataset_name
                    __args__.ratio_samples = DATA_TO_RATIO_SAMPLES[dataset_name]
                    kws = dict(file_dir="../_logs_wl4s_k", log_postfix=f"_sliced_{k_to_sample or 0}")
                    hp_search_for_models(__args__, HPARAM_SPACE, MORE_HPARAM_SPACE, **kws)

            __args__.stype = "connected"
            HPARAM_SPACE["stype"] = ["connected"]
            for dataset_name in ["HPONeuro", "HPOMetab"]:
                __args__.dataset_name = dataset_name
                __args__.ratio_samples = DATA_TO_RATIO_SAMPLES[dataset_name]
                kws = dict(file_dir="../_logs_wl4s_k", log_postfix=f"_sliced_inf")
                hp_search_for_models(__args__, HPARAM_SPACE, MORE_HPARAM_SPACE, **kws)

        else:
            raise ValueError(f"Not supported MODE: {MODE}")
