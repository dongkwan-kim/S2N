from wl4s import parser, hp_search_syn, hp_search_real

if __name__ == '__main__':
    HPARAM_SPACE = {
        "stype": ["separated"],
        "wl_cumcat": [True, False],
        "hist_norm": [False, True],
        "model": ["LinearSVC"],
    }
    MORE_HPARAM_SPACE = {
        "C": [1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3],
        "dual": [True, False],
    }

    __args__ = parser.parse_args()
    for k_to_sample in [1, 2, 3]:
        __args__.k_to_sample = k_to_sample
        hp_search_syn(__args__, HPARAM_SPACE, MORE_HPARAM_SPACE, file_dir="../_logs_wl4s_k")
        hp_search_real(__args__, HPARAM_SPACE, MORE_HPARAM_SPACE, file_dir="../_logs_wl4s_k")
