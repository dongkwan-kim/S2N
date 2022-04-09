from typing import List, Dict, Any

from termcolor import cprint

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError:
    pass
import pandas as pd


def plot_scatter(xs, ys, xlabel, ylabel,
                 path, key, extension="pdf",
                 hues=None, hue_name=None,
                 styles=None, style_name=None,
                 cols=None, col_name=None,
                 label_kws: Dict[str, Any] = None,
                 scales_kws: Dict[str, Any] = None,
                 yticks=None,
                 **kwargs):
    data = {
        xlabel: xs,
        ylabel: ys,
        **{obj_name: obj for obj_name, obj in zip([hue_name, style_name, col_name],
                                                  [hues, styles, cols])
           if obj_name is not None}
    }
    df = pd.DataFrame(data)

    plot = sns.relplot(
        kind="scatter",
        x=xlabel, y=ylabel, hue=hue_name, style=style_name, col=col_name,
        data=df,
        **kwargs,
    )
    if "legend" in kwargs and kwargs["legend"] is not False:
        for lh in plot._legend.legendHandles:
            lh.set_sizes([kwargs["s"]])

    if label_kws is not None:
        plot.set(**label_kws)  # e.g., xlabel=None
    if scales_kws is not None:
        plot.set(**scales_kws)  # e.g., xscale="log", yscale="log"
    if yticks is not None:
        plt.yticks(yticks)

    plot_info = "_".join([k for k in [xlabel, ylabel, hue_name, style_name]])
    plot_info = plot_info.replace("/", "+").replace("#", "Num")
    path_and_name = "{}/fig_scatter_{}_{}.{}".format(path, key, plot_info, extension)
    path_and_name = path_and_name.replace(" ", "_").replace("(", "_").replace(")", "_")

    plot.savefig(path_and_name, bbox_inches='tight')
    cprint(f"Save at {path_and_name}", "blue")
    plt.clf()


def plot_efficiency_scatter(extension, dataset):
    sns.set_theme(style="whitegrid")
    sns.set_context(context="talk", font_scale=1.1, rc={"legend.fontsize": 18})

    common_kwargs = {
        "ylabel": "Performance",
        "path": "../figures",
        "extension": extension,
        "s": 400,
        "aspect": 0.8,
    }

    if dataset == "HPONeuro" or dataset == "ALL":
        key = "HPO-Neuro"
        yticks = [0.5, 0.55, 0.6, 0.65]
        plot_scatter(
            xs=[2535188, 1880202, 1892958, 2157066, 1880576],
            ys=[0.632, 0.59, 0.531, 0.629, 0.645],
            xlabel="Params.",
            key=key,
            hues=["No", "Yes", "Yes", "Yes", "Yes"],
            hue_name="Use S2N",
            hue_order=["Yes", "No"],
            styles=["SubGNN", "GCN", "GAT", "LINKX-I", "FAGCN"],
            style_name="Model",
            cols=["*"] * 5,
            col_name="Placeholder",
            scales_kws=None,
            yticks=yticks,
            legend=False,
            **common_kwargs,
        )
        plot_scatter(
            xs=[482.7, 135022.3, 27590.3, 52914.9, 28881.5] + [642.7, 20297.1, 6512.5, 10183.4, 5076.8],
            ys=[0.632, 0.59, 0.531, 0.629, 0.645] * 2,
            xlabel="Throughput (#/sec)",
            key=key,
            hues=["No", "Yes", "Yes", "Yes", "Yes"] * 2,
            hue_name="Use S2N",
            hue_order=["Yes", "No"],
            styles=["SubGNN", "GCN", "GAT", "LINKX-I", "FAGCN"] * 2,
            style_name="Model",
            cols=["Training"] * 5 + ["Inference"] * 5,
            col_name="Stage",
            scales_kws={"xscale": "log"},
            label_kws={"ylabel": None},
            yticks=yticks,
            legend=False,
            **common_kwargs,
        )
        plot_scatter(
            xs=[0.266, 0.024, 0.116, 0.061, 0.111] + [0.156, 0.020, 0.061, 0.039, 0.079],
            ys=[0.632, 0.59, 0.531, 0.629, 0.645] * 2,
            xlabel="Latency (sec/forward)",
            key=key,
            hues=["No", "Yes", "Yes", "Yes", "Yes"] * 2,
            hue_name="Use S2N",
            hue_order=["Yes", "No"],
            styles=["SubGNN", "GCN", "GAT", "LINKX-I", "FAGCN"] * 2,
            style_name="Model",
            cols=["Training"] * 5 + ["Inference"] * 5,
            col_name="Stage",
            # scales_kws={"xscale": "log"},
            label_kws={"ylabel": None},
            yticks=yticks,
            legend="full",
            **common_kwargs,
        )
    if dataset == "HPOMetab" or dataset == "ALL":
        key = "HPO-Metab"
        yticks = [0.45, 0.5, 0.55, 0.6]
        plot_scatter(
            xs=[2843566, 1879942, 1884370, 2046086, 1880320],
            ys=[0.537, 0.516, 0.479, 0.559, 0.582],
            xlabel="Params.",
            key=key,
            hues=["No", "Yes", "Yes", "Yes", "Yes"],
            hue_name="Use S2N",
            hue_order=["Yes", "No"],
            styles=["SubGNN", "GCN", "GAT", "LINKX-I", "FAGCN"],
            style_name="Model",
            cols=["*"] * 5,
            col_name="Placeholder",
            scales_kws=None,
            yticks=yticks,
            legend=False,
            **common_kwargs,
        )
        plot_scatter(
            xs=[471.1, 142426.6, 6086.3, 59810.1, 28532.6] + [850.3, 29845.6, 9091.1, 13288.7, 6988.0],
            ys=[0.537, 0.516, 0.479, 0.559, 0.582] * 2,
            xlabel="Throughput (#/sec)",
            key=key,
            hues=["No", "Yes", "Yes", "Yes", "Yes"] * 2,
            hue_name="Use S2N",
            hue_order=["Yes", "No"],
            styles=["SubGNN", "GCN", "GAT", "LINKX-I", "FAGCN"] * 2,
            style_name="Model",
            cols=["Training"] * 5 + ["Inference"] * 5,
            col_name="Stage",
            scales_kws={"xscale": "log"},
            label_kws={"ylabel": None},
            yticks=yticks,
            legend=False,
            **common_kwargs,
        )
        plot_scatter(
            xs=[0.136, 0.013, 0.316, 0.032, 0.067] + [0.072, 0.008, 0.027, 0.018, 0.035],
            ys=[0.537, 0.516, 0.479, 0.559, 0.582] * 2,
            xlabel="Latency (sec/forward)",
            key=key,
            hues=["No", "Yes", "Yes", "Yes", "Yes"] * 2,
            hue_name="Use S2N",
            hue_order=["Yes", "No"],
            styles=["SubGNN", "GCN", "GAT", "LINKX-I", "FAGCN"] * 2,
            style_name="Model",
            cols=["Training"] * 5 + ["Inference"] * 5,
            col_name="Stage",
            # scales_kws={"xscale": "log"},
            label_kws={"ylabel": None},
            yticks=yticks,
            legend="full",
            **common_kwargs,
        )
    if dataset == "EMUser" or dataset == "ALL":
        key = "EM-User"
        yticks = [0.7, 0.75, 0.8, 0.85]
        plot_scatter(
            xs=[8113668, 7363778, 7359494, 7392770, 7359616],
            ys=[0.814, 0.702, 0.714, 0.833, 0.8],
            xlabel="Params.",
            key=key,
            hues=["No", "Yes", "Yes", "Yes", "Yes"],
            hue_name="Use S2N",
            hue_order=["Yes", "No"],
            styles=["SubGNN", "GCN", "GAT", "LINKX-I", "FAGCN"],
            style_name="Model",
            cols=["*"] * 5,
            col_name="Placeholder",
            scales_kws=None,
            yticks=yticks,
            legend=False,
            **common_kwargs,
        )
        plot_scatter(
            xs=[682.0, 9637.0, 8300.6, 14419.8, 10586.0] + [328.1, 4200.4, 4166.2, 7885.8, 4476.7],
            ys=[0.814, 0.702, 0.714, 0.833, 0.8] * 2,
            xlabel="Throughput (#/sec)",
            key=key,
            hues=["No", "Yes", "Yes", "Yes", "Yes"] * 2,
            hue_name="Use S2N",
            hue_order=["Yes", "No"],
            styles=["SubGNN", "GCN", "GAT", "LINKX-I", "FAGCN"] * 2,
            style_name="Model",
            cols=["Training"] * 5 + ["Inference"] * 5,
            col_name="Stage",
            scales_kws={"xscale": "log"},
            label_kws={"ylabel": None},
            yticks=yticks,
            legend=False,
            **common_kwargs,
        )
        plot_scatter(
            xs=[0.331, 0.023, 0.027, 0.016, 0.021] + [0.149, 0.012, 0.012, 0.006, 0.011],
            ys=[0.814, 0.702, 0.714, 0.833, 0.8] * 2,
            xlabel="Latency (sec/forward)",
            key=key,
            hues=["No", "Yes", "Yes", "Yes", "Yes"] * 2,
            hue_name="Use S2N",
            hue_order=["Yes", "No"],
            styles=["SubGNN", "GCN", "GAT", "LINKX-I", "FAGCN"] * 2,
            style_name="Model",
            cols=["Training"] * 5 + ["Inference"] * 5,
            col_name="Stage",
            # scales_kws={"xscale": "log"},
            label_kws={"ylabel": None},
            yticks=yticks,
            legend="full",
            **common_kwargs,
        )


if __name__ == '__main__':

    MODE = "plot_efficiency_scatter"

    if MODE == "plot_efficiency_scatter":
        plot_efficiency_scatter(
            extension="pdf",
            dataset="ALL",
        )
