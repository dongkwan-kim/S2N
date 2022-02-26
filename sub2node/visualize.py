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
    plot_info = plot_info.replace("/", "|").replace("#", "Num")
    path_and_name = "{}/fig_scatter_{}_{}.{}".format(path, key, plot_info, extension)

    plot.savefig(path_and_name, bbox_inches='tight')
    cprint(f"Save at {path_and_name}", "blue")
    plt.clf()
