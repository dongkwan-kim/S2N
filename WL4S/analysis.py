import pandas as pd
import seaborn as sns

from visualize import plot_line


def load_best_df(path, cols_to_maintain, query=None):
    df = pd.read_csv(path)
    if query is not None:
        df = df.query(query)

    # Find the row with the best (max) test_f1_mean
    best_row = df.loc[df['test_f1_mean'].idxmax()]

    # Filter the rows that have the same values in cols_to_maintain as the best row
    condition = (df[cols_to_maintain] == best_row[cols_to_maintain]).all(axis=1)
    filtered_df = df[condition]

    return filtered_df


def analyze_hparams_sensitivity(dataset_name="PPIBP", extension="png"):
    kws = dict(
        path="../_figures",
        key=f"hp_{dataset_name}",
        extension=extension,
        markers=True, dashes=False,
        aspect=1.0,
    )

    df = load_best_df(f"../_logs_wl4s2/complete_real/{dataset_name}.csv", ["wl_cumcat", "hist_norm", "C"],
                      query="best_wl_list != '[manual]'")
    plot_line(
        scales_kws={"xscale": "log"},
        xs=df["a_s"].to_list(), xlabel="alpha_0",
        ys=df["test_f1_mean"].to_list(), ylabel="Performance",
        styles=df["model"].to_list(), style_name="Model",
        legend=False,
        **kws,
    )

    df = load_best_df(f"../_logs_wl4s2/complete_real/{dataset_name}.csv", ["wl_cumcat", "hist_norm", "a_c"],
                      query="best_wl_list != '[manual]'")
    df = df[df["C"] < 100]
    plot_line(
        scales_kws={"xscale": "log"},
        xs=df["C"].to_list(), xlabel="L2",
        ys=df["test_f1_mean"].to_list(), ylabel="Performance",
        styles=df["model"].to_list(), style_name="Model",
        legend=False,
        **kws,
    )

    df = load_best_df(f"../_logs_wl4s2/complete_real/{dataset_name}.csv", ["wl_cumcat", "hist_norm", "C", "a_c"],
                      query="best_wl_list == '[manual]'")
    plot_line(
        xticks=[1, 2, 3, 4, 5],
        xs=df["wl_layers"].to_list(), xlabel="# iterations",
        ys=df["test_f1_mean"].to_list(), ylabel="Performance",
        styles=df["model"].to_list(), style_name="Model",
        legend=False,
        **kws,
    )


if __name__ == '__main__':
    sns.set(style="whitegrid")
    sns.set_context("poster")

    _extension = "pdf"
    analyze_hparams_sensitivity("EMUser", extension=_extension)
    analyze_hparams_sensitivity("PPIBP", extension=_extension)
    analyze_hparams_sensitivity("HPOMetab", extension=_extension)
    analyze_hparams_sensitivity("HPONeuro", extension=_extension)
