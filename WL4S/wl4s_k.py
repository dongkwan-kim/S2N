from wl4s import parser, hp_search_syn, hp_search_real

if __name__ == '__main__':
    HPARAM_SPACE = {
        "stype": ["separated"],
        "wl_cumcat": [False],
        "hist_norm": [False, True],
        "model": ["LinearSVC"],
    }
    Cx100 = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    MORE_HPARAM_SPACE = {
        "C": [c / 100 for c in Cx100],
        "dual": [True, False],
    }

    __args__ = parser.parse_args()
    for k_to_sample in [1, 2, 3]:
        __args__.k_to_sample = k_to_sample
        hp_search_syn(__args__, HPARAM_SPACE, MORE_HPARAM_SPACE, file_dir="../_logs_wl4s_k")
        hp_search_real(__args__, HPARAM_SPACE, MORE_HPARAM_SPACE, file_dir="../_logs_wl4s_k")
