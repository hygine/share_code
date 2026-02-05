# utils/clustering.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.annotation import prepare_annotation, build_annotation_colors



def _zscore(df: pd.DataFrame, axis: int):
    """
    axis = 0 → 按列
    axis = 1 → 按行
    """
    return df.sub(df.mean(axis=axis), axis=1-axis).div(df.std(axis=axis), axis=1-axis)


def pheatmap_like(
    df,
    annotation_col=None,
    z_score=None,
    cluster_rows=True,
    cluster_cols=True,
    cmap="RdBu_r",
    figsize=(8, 10)
):
    plot_data = df.copy()
    plot_data = plot_data.apply(pd.to_numeric, errors="coerce")
    plot_data = plot_data.dropna(how="any")

    # Z-score
    if z_score == "row":
        plot_data = plot_data.sub(plot_data.mean(axis=1), axis=0)
        plot_data = plot_data.div(plot_data.std(axis=1), axis=0)
    elif z_score == "col":
        plot_data = plot_data.sub(plot_data.mean(axis=0), axis=1)
        plot_data = plot_data.div(plot_data.std(axis=0), axis=1)

    plot_data = plot_data.dropna(how="any")

    col_colors = None
    lut_dict = None

    if annotation_col is not None:
        annotation_col = annotation_col.loc[plot_data.columns]
        col_colors, lut_dict = build_annotation_colors(annotation_col)

    cg = sns.clustermap(
        plot_data,
        cmap=cmap,
        row_cluster=cluster_rows,
        col_cluster=cluster_cols,
        col_colors=col_colors,
        figsize=figsize,
        xticklabels=True,
        yticklabels=True
    )

    # === 手动添加 legend（pheatmap 风格）===
    if lut_dict:
        for label, lut in lut_dict.items():
            for cat, color in lut.items():
                cg.ax_col_dendrogram.bar(
                    0, 0, color=color, label=f"{label}: {cat}", linewidth=0
                )
        cg.ax_col_dendrogram.legend(
            loc="center",
            ncol=1,
            bbox_to_anchor=(1.1, 0.5),
            frameon=False
        )

    return cg
