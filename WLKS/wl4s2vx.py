from termcolor import cprint

from wl4s import (
    parser, hp_search_for_models, hp_search_real, hp_search_syn,
    get_data_and_model, run_one, get_all_kernels,
)


def get_data_x_mixed_kernels(args, precompute=True):
    assert args.dtype == "kernel"

    given_layer_name = args.layer_name
    given_kernel_type = args.kernel_type
    given_scaler = args.scaler

    args.layer_name = "WLConv"
    args.kernel_type = "linear"
    args.scaler = ""

    args.stype = "connected"
    k_list_c, splits_c, y_c = get_data_and_model(args, precompute)

    args.stype = "separated"
    k_list_s = get_all_kernels(args, splits_c) if precompute else None
    if k_list_s is None:
        k_list_s, _, _ = get_data_and_model(args, precompute)
    else:
        cprint(f"Use precomputed kernels...", "yellow")

    args.layer_name = given_layer_name
    args.kernel_type = given_kernel_type
    args.scaler = given_scaler
    args.hist_norm = True
    args.stype = args.xtype
    k_list_x = get_all_kernels(args, splits_c) if precompute else None
    if k_list_x is None:
        k_list_x, _, _ = get_data_and_model(args, precompute)
    else:
        cprint(f"Use precomputed kernels...", "yellow")

    k_list_new = []
    for kt_c, kt_s, kt_x in zip(k_list_c, k_list_s, k_list_x):
        kt_new = tuple([(args.a_c * k_c + args.a_s * k_s + args.a_x * k_x)
                        for k_c, k_s, k_x in zip(kt_c, kt_s, kt_x)])
        k_list_new.append(kt_new)

    return k_list_new, splits_c, y_c


if __name__ == '__main__':

    parser.add_argument("--a_c", type=float, default=0.9, help="a_c parameter")
    parser.add_argument("--a_s", type=float, default=0.1, help="a_s parameter")
    parser.add_argument("--a_x", type=float, default=0.0, help="a_x parameter")
    parser.add_argument("--xtype", type=str, default="separated")

    HPARAM_SPACE = {
        "stype": [None],  # NOTE: None
        "wl_cumcat": [False],
        "hist_norm": [False, True],
        "model": ["SVC"],
        "kernel": ["precomputed"],
        "dtype": ["kernel"],
    }
    Cx100 = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    MORE_HPARAM_SPACE = {
        "C": [c / 100 for c in Cx100],
    }
    __args__ = parser.parse_args()
    __args__.dtype = "kernel"
    # __args__.kernel_type = "rbf"
    # __args__.scaler = "StandardScaler"
    assert __args__.layer_name != "WLConv"

    WL4S2_KWS = dict(
        data_func=get_data_x_mixed_kernels,  # NOTE: important
        file_dir=f"../_logs_wl4s2_x_{__args__.layer_name}_{__args__.kernel_type}_{__args__.scaler}",
    )

    if __args__.MODE == "run_one":
        run_one(__args__, data_func=get_data_x_mixed_kernels, precompute=True)

    else:
        for _a_x in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]:
            for _xtype in ["connected", "separated"]:
                for _a_c in [0.999, 0.99, 0.9, 0.5, 0.1, 0.01, 0.001]:
                    __args__.xtype = _xtype
                    __args__.a_x = _a_x
                    _t = 1 / (1.0 + _a_x)
                    __args__.a_c, __args__.a_s = _a_c * _t, (1.0 - _a_c) * _t

                    if __args__.MODE == "hp_search_for_models":
                        hp_search_for_models(__args__, HPARAM_SPACE, MORE_HPARAM_SPACE, **WL4S2_KWS)
                    elif __args__.MODE == "hp_search_real":
                        hp_search_real(__args__, HPARAM_SPACE, MORE_HPARAM_SPACE, **WL4S2_KWS)
                    elif __args__.MODE == "hp_search_syn":
                        hp_search_syn(__args__, HPARAM_SPACE, MORE_HPARAM_SPACE, **WL4S2_KWS)
