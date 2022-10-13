from typing import List, Dict, Any

from sklearn.manifold import TSNE
from termcolor import cprint

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError:
    pass
import pandas as pd
import numpy as np


def plot_data_points_by_tsne(xs: np.ndarray, ys: np.ndarray,
                             path=None, key=None, extension="png", **kwargs):

    def plot_for_one_y(_ys, _key, _title=None):
        df = pd.DataFrame({
            "coord_1": x_embed[:, 0],
            "coord_2": x_embed[:, 1],
            "class": _ys,
        })
        plot = sns.scatterplot(x="coord_1", y="coord_2", hue="class", data=df,
                               legend=False, palette="Set1", **kwargs)
        if _title:
            plt.title(_title)
        plot.set_xlabel("")
        plot.set_ylabel("")
        plot.get_xaxis().set_visible(False)
        plot.get_yaxis().set_visible(False)
        sns.despine(left=False, right=False, bottom=False, top=False)

        if path is not None:
            plot.get_figure().savefig("{}/fig_tsne_{}.{}".format(path, _key, extension), bbox_inches='tight')
        else:
            plt.show()
        plt.clf()

    x_embed = TSNE(n_components=2).fit_transform(xs)

    if len(ys.shape) == 1:
        plot_for_one_y(ys, _key=key, _title=f"TSNE: {key}")
    elif len(ys.shape) == 2:  # [N, C]
        for y_idx in range(ys.shape[1]):
            plot_for_one_y(ys[:, y_idx], _key=f"{key}_y{y_idx}", _title=f"TSNE: {key} of y={y_idx}")


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
    cprint(f"Saved: {path_and_name}", "blue")
    plt.clf()
