from wl4s import parser, hp_search_for_models, hp_search_real, get_data_and_model, hp_search_syn, run_one


def get_data_mixed_kernels(args):
    args.stype = "separated"
    k_list_s, splits_s, y_s = get_data_and_model(args)

    args.stype = "connected"
    k_list_c, splits_c, y_c = get_data_and_model(args)

    k_list_new = []
    for kt_c, kt_s in zip(k_list_c, k_list_s):
        kt_new = tuple([(args.a_c * k_c + args.a_s * k_s) for k_c, k_s in zip(kt_c, kt_s)])
        k_list_new.append(kt_new)

    return k_list_new, splits_c, y_c


if __name__ == '__main__':

    parser.add_argument("--a_c", type=float, default=0.9, help="a_c parameter")
    parser.add_argument("--a_s", type=float, default=0.1, help="a_s parameter")

    HPARAM_SPACE = {
        "stype": [None],  # NOTE: None
        "wl_cumcat": [False, True],
        "hist_norm": [False, True],
        "model": ["SVC"],
        "kernel": ["precomputed"],
        "dtype": ["kernel"],
    }
    Cx100 = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    MORE_HPARAM_SPACE = {
        "C": [c / 100 for c in Cx100],
    }
    WL4S2_KWS = dict(
        data_func=get_data_mixed_kernels,  # NOTE: important
        file_dir="../_logs_wl4s2",
    )

    __args__ = parser.parse_args()
    __args__.dtype = "kernel"

    if __args__.MODE == "run_one":
        run_one(__args__)
    else:
        for _a_c, _a_s in [
            (0.99, 0.01), (0.9, 0.1), (0.5, 0.1),
            (0.01, 0.99), (0.1, 0.9), (0.1, 0.5),
        ]:
            __args__.a_c, __args__.a_s = _a_c, _a_s

            if __args__.MODE == "hp_search_for_models":
                hp_search_for_models(__args__, HPARAM_SPACE, MORE_HPARAM_SPACE, **WL4S2_KWS)
            elif __args__.MODE == "hp_search_real":
                hp_search_real(__args__, HPARAM_SPACE, MORE_HPARAM_SPACE, **WL4S2_KWS)
            elif __args__.MODE == "hp_search_syn":
                hp_search_syn(__args__, HPARAM_SPACE, MORE_HPARAM_SPACE, **WL4S2_KWS)
