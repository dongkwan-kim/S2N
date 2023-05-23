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


def plot_catplot(kind,
                 xs, ys, xlabel, ylabel,
                 path, key, extension="pdf",
                 orient="v",
                 hues=None, hue_name=None,
                 rows=None, row_name=None,
                 cols=None, col_name=None,
                 label_kws: Dict[str, Any] = None,
                 scales_kws: Dict[str, Any] = None,
                 xticks=None, yticks=None,
                 tight_layout_kwargs: dict = None,
                 **kwargs):
    data = {
        xlabel: xs,
        ylabel: ys,
        **{obj_name: obj for obj_name, obj in zip([hue_name, col_name, row_name],
                                                  [hues, cols, rows])
           if obj_name is not None}
    }
    df = pd.DataFrame(data)

    plot = sns.catplot(
        kind=kind,
        data=df, orient=orient,
        x=xlabel, y=ylabel, hue=hue_name, row=row_name, col=col_name,
        **kwargs,
    )

    if label_kws is not None:
        plot.set(**label_kws)  # e.g., xlabel=None
    if scales_kws is not None:
        plot.set(**scales_kws)  # e.g., xscale="log", yscale="log"
    if yticks is not None:
        plt.yticks(yticks)
    if xticks is not None:
        plt.xticks(xticks)

    if tight_layout_kwargs is not None:
        # e.g., rect=[0, 0, 0.8, 1]
        plt.tight_layout(**tight_layout_kwargs)

    plot_info = "_".join([k for k in [xlabel, ylabel, hue_name]
                          if k is not None])
    plot_info = plot_info.replace("/", "|").replace("#", "Num")
    path_and_name = "{}/fig_{}_{}_{}.{}".format(path, kind, key, plot_info, extension)

    plot.savefig(path_and_name, bbox_inches='tight')
    cprint(f"Saved: {path_and_name}", "blue")
    plt.clf()


def plot_box(xs, ys, xlabel, ylabel,
             path, key, extension="pdf",
             orient="v",
             hues=None, hue_name=None,
             rows=None, row_name=None,
             cols=None, col_name=None,
             label_kws: Dict[str, Any] = None,
             scales_kws: Dict[str, Any] = None,
             xticks=None, yticks=None,
             tight_layout_kwargs: dict = None,
             **kwargs):
    plot_catplot("box", xs=xs, ys=ys, xlabel=xlabel, ylabel=ylabel,
                 path=path, key=key, extension=extension,
                 orient=orient,
                 hues=hues, hue_name=hue_name,
                 rows=rows, row_name=row_name,
                 cols=cols, col_name=col_name,
                 label_kws=label_kws,
                 scales_kws=scales_kws,
                 tight_layout_kwargs=tight_layout_kwargs,
                 xticks=xticks, yticks=yticks,
                 **kwargs)


def plot_bar(xs, ys, xlabel, ylabel,
             path, key, extension="pdf",
             orient="v",
             hues=None, hue_name=None,
             rows=None, row_name=None,
             cols=None, col_name=None,
             label_kws: Dict[str, Any] = None,
             scales_kws: Dict[str, Any] = None,
             xticks=None, yticks=None,
             tight_layout_kwargs: dict = None,
             **kwargs):
    plot_catplot("bar", xs=xs, ys=ys, xlabel=xlabel, ylabel=ylabel,
                 path=path, key=key, extension=extension,
                 orient=orient,
                 hues=hues, hue_name=hue_name,
                 rows=rows, row_name=row_name,
                 cols=cols, col_name=col_name,
                 label_kws=label_kws,
                 scales_kws=scales_kws,
                 tight_layout_kwargs=tight_layout_kwargs,
                 xticks=xticks, yticks=yticks,
                 **kwargs)


def plot_relplot(kind,
                 xs, ys, xlabel, ylabel,
                 path, key, extension="pdf",
                 hues=None, hue_name=None,
                 styles=None, style_name=None,
                 rows=None, row_name=None,
                 cols=None, col_name=None,
                 elm_sizes=None, elm_size_name=None,
                 label_kws: Dict[str, Any] = None,
                 scales_kws: Dict[str, Any] = None,
                 xticks=None, yticks=None,
                 tight_layout_kwargs: dict = None,
                 **kwargs):

    assert kind in ["scatter", "line"]

    data = {
        xlabel: xs,
        ylabel: ys,
        **{obj_name: obj for obj_name, obj in zip([hue_name, style_name, row_name, col_name, elm_size_name],
                                                  [hues, styles, rows, cols, elm_sizes])
           if obj_name is not None}
    }
    df = pd.DataFrame(data)

    plot = sns.relplot(
        kind=kind,
        x=xlabel, y=ylabel, hue=hue_name, style=style_name, col=col_name, size=elm_size_name,
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
    if xticks is not None:
        plt.xticks(xticks)

    if tight_layout_kwargs is not None:
        # e.g., rect=[0, 0, 0.8, 1]
        plt.tight_layout(**tight_layout_kwargs)

    plot_info = "_".join([k for k in [xlabel, ylabel, hue_name, style_name]
                          if k is not None])
    plot_info = plot_info.replace("/", "|").replace("#", "Num")
    path_and_name = f"{path}/fig_{kind}_{key}_{plot_info}.{extension}"

    plot.savefig(path_and_name, bbox_inches='tight')
    cprint(f"Saved: {path_and_name}", "blue")
    plt.clf()


def plot_scatter(xs, ys, xlabel, ylabel,
                 path, key, extension="pdf",
                 hues=None, hue_name=None,
                 styles=None, style_name=None,
                 rows=None, row_name=None,
                 cols=None, col_name=None,
                 elm_sizes=None, elm_size_name=None,
                 label_kws: Dict[str, Any] = None,
                 scales_kws: Dict[str, Any] = None,
                 xticks=None, yticks=None,
                 tight_layout_kwargs: dict = None,
                 **kwargs):
    plot_relplot("scatter", xs=xs, ys=ys, xlabel=xlabel, ylabel=ylabel,
                 path=path, key=key, extension=extension,
                 hues=hues, hue_name=hue_name,
                 styles=styles, style_name=style_name,
                 rows=rows, row_name=row_name,
                 cols=cols, col_name=col_name,
                 elm_sizes=elm_sizes, elm_size_name=elm_size_name,
                 label_kws=label_kws,
                 scales_kws=scales_kws,
                 xticks=xticks, yticks=yticks,
                 tight_layout_kwargs=tight_layout_kwargs,
                 **kwargs)


def plot_line(xs, ys, xlabel, ylabel,
              path, key, extension="pdf",
              hues=None, hue_name=None,
              styles=None, style_name=None,
              rows=None, row_name=None,
              cols=None, col_name=None,
              elm_sizes=None, elm_size_name=None,
              label_kws: Dict[str, Any] = None,
              scales_kws: Dict[str, Any] = None,
              xticks=None, yticks=None,
              tight_layout_kwargs: dict = None,
              **kwargs):
    plot_relplot("line", xs=xs, ys=ys, xlabel=xlabel, ylabel=ylabel,
                 path=path, key=key, extension=extension,
                 hues=hues, hue_name=hue_name,
                 styles=styles, style_name=style_name,
                 rows=rows, row_name=row_name,
                 cols=cols, col_name=col_name,
                 elm_sizes=elm_sizes, elm_size_name=elm_size_name,
                 label_kws=label_kws,
                 scales_kws=scales_kws,
                 xticks=xticks, yticks=yticks,
                 tight_layout_kwargs=tight_layout_kwargs,
                 **kwargs)
