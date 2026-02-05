# modules/correlation.py
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np

from utils.clustering import pheatmap_like


def _annotate_heatmap_values(cg, data_2d: np.ndarray, fmt: str = ".2f", fontsize: int = 8, color: str = "black"):
    """
    在 pheatmap_like 生成的热图上叠加数值（兼容 seaborn clustermap 风格对象）
    注意：当 n 很大时非常慢（O(n^2) text），建议限制 n 或默认关闭。
    """
    ax = getattr(cg, "ax_heatmap", None)
    if ax is None:
        return

    nrows, ncols = data_2d.shape
    for i in range(nrows):
        for j in range(ncols):
            val = data_2d[i, j]
            if np.isfinite(val):
                ax.text(
                    j + 0.5, i + 0.5,
                    format(float(val), fmt),
                    ha="center", va="center",
                    fontsize=fontsize,
                    color=color
                )


def _default_step(n: int) -> int:
    if n <= 25:
        return 1
    if n <= 50:
        return 2
    if n <= 80:
        return 3
    if n <= 120:
        return 4
    return 6


def correlation_block(
    df: pd.DataFrame,
    df_show: Optional[pd.DataFrame],
    annotation_col: Optional[pd.DataFrame]
):
    st.subheader("📐 样本相关性热图（pheatmap 风格 + annotation 分组条）")

    try:
        # =========================
        # 数据来源
        # =========================
        source = st.radio(
            "基因来源",
            ["全部", "搜索结果"],
            horizontal=True,
            key="corr_source"
        )
        data = df_show if (source == "搜索结果" and df_show is not None and df_show.shape[0] > 0) else df

        if data is None or data.shape[1] < 3:
            st.warning("样本数不足（<3），无法计算相关性")
            return

        method = st.selectbox("相关性方法", ["pearson", "spearman"], key="corr_method")
        cluster_rows = st.checkbox("行聚类（样本）", True, key="corr_cluster_rows")
        cluster_cols = st.checkbox("列聚类（样本）", True, key="corr_cluster_cols")

        # =========================
        # 标签/画布优化（新增，解决挤在一起）
        # =========================
        n_samples = int(data.shape[1])

        with st.expander("🔧 标签/画布显示优化（解决文字挤在一起）", expanded=True):
            show_row_names = st.checkbox("显示行名（样本名）", value=True, key="corr_show_row_names")
            show_col_names = st.checkbox("显示列名（样本名）", value=True, key="corr_show_col_names")

            col_rotate = st.selectbox("列名旋转角度", [0, 45, 60, 90], index=3, key="corr_col_rotate")

            row_step = st.slider(
                "行名显示间隔（每 N 个显示一个）",
                1, max(1, min(30, n_samples)),
                value=_default_step(n_samples),
                key="corr_row_step"
            )
            col_step = st.slider(
                "列名显示间隔（每 N 个显示一个）",
                1, max(1, min(30, n_samples)),
                value=_default_step(n_samples),
                key="corr_col_step"
            )

            row_font = st.slider("行名字体大小", 4, 14, 7, key="corr_row_font")
            col_font = st.slider("列名字体大小", 4, 14, 7, key="corr_col_font")

            auto_size = st.checkbox("自动调节画布大小", value=True, key="corr_auto_size")
            width_scale = st.slider("宽度系数（自动尺寸）", 0.6, 2.5, 1.2, key="corr_w_scale")
            height_scale = st.slider("高度系数（自动尺寸）", 0.6, 2.5, 1.2, key="corr_h_scale")

        # =========================
        # 数值显示设置（大样本默认关闭，避免非常慢）
        # =========================
        show_values_default = False if n_samples > 40 else False
        show_values = st.checkbox(
            "在热图上显示相关系数数值（样本多时会明显变慢）",
            value=show_values_default,
            key="corr_show_values"
        )
        value_fmt = st.selectbox("数值格式", [".2f", ".3f"], index=0, key="corr_value_fmt")
        value_fontsize = st.slider("数值字体大小", 5, 14, 8, key="corr_value_fontsize")
        value_color = st.selectbox("数值颜色", ["black", "white", "gray"], index=0, key="corr_value_color")

        if show_values and n_samples > 60:
            st.warning("当前样本数较多（>60），叠加数值会非常慢且导出图很乱，建议关闭。")

        # =========================
        # 样本-样本相关性（samples × samples）
        # data: genes × samples
        # corr: samples × samples
        # =========================
        corr = data.corr(method=method)

        # =========================
        # annotation 对齐（保持方阵）
        # =========================
        anno_used = None
        if annotation_col is not None and not annotation_col.empty:
            common = corr.index.intersection(annotation_col.index)
            if len(common) >= 3:
                corr = corr.loc[common, common]
                anno_used = annotation_col.loc[common]
            else:
                st.warning("annotation 与样本匹配不足（<3），未显示 annotation 分组条")

        # =========================
        # 绘图：复用 pheatmap_like（风格一致）
        # =========================
        cg = pheatmap_like(
            corr,
            annotation_col=anno_used,
            z_score=None,
            cluster_rows=cluster_rows,
            cluster_cols=cluster_cols
        )

        # =========================
        # ✅ 统一做 ticklabel 控制（抽样/旋转/字体/隐藏）
        # =========================
        ax = getattr(cg, "ax_heatmap", None)
        if ax is not None:
            # x tick labels
            xt = ax.get_xticklabels()
            if not show_col_names:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            else:
                for i, lab in enumerate(xt):
                    if col_step > 1 and (i % col_step != 0):
                        lab.set_text("")
                ax.set_xticklabels(xt, rotation=col_rotate, ha="right" if col_rotate else "center", fontsize=col_font)

            # y tick labels
            yt = ax.get_yticklabels()
            if not show_row_names:
                ax.set_yticklabels([])
                ax.set_ylabel("")
            else:
                for i, lab in enumerate(yt):
                    if row_step > 1 and (i % row_step != 0):
                        lab.set_text("")
                ax.set_yticklabels(yt, rotation=0, fontsize=row_font)

            ax.tick_params(axis="both", length=2, width=0.5)

        # =========================
        # ✅ 自动画布尺寸（避免标签溢出/挤压）
        # =========================
        try:
            if auto_size:
                # 相关性热图为方阵，宽高与样本数线性相关
                w = min(32.0, max(10.0, (n_samples * 0.22) * width_scale))
                h = min(32.0, max(10.0, (n_samples * 0.22) * height_scale))
                cg.fig.set_size_inches(w, h)
            cg.fig.tight_layout()
        except Exception:
            pass

        # =========================
        # 叠加数值（按聚类后顺序重排）
        # =========================
        if show_values:
            row_order = None
            col_order = None
            if hasattr(cg, "dendrogram_row") and cg.dendrogram_row is not None:
                row_order = cg.dendrogram_row.reordered_ind
            if hasattr(cg, "dendrogram_col") and cg.dendrogram_col is not None:
                col_order = cg.dendrogram_col.reordered_ind

            corr_values = corr.values
            if row_order is not None:
                corr_values = corr_values[np.ix_(row_order, np.arange(corr_values.shape[1]))]
            if col_order is not None:
                corr_values = corr_values[np.ix_(np.arange(corr_values.shape[0]), col_order)]

            _annotate_heatmap_values(
                cg,
                corr_values,
                fmt=value_fmt,
                fontsize=value_fontsize,
                color=value_color
            )

        st.pyplot(cg.fig, clear_figure=True)

        # =========================
        # 导出
        # =========================
        import io

        st.download_button(
            "📥 下载相关性矩阵 CSV",
            corr.to_csv().encode(),
            file_name="sample_correlation.csv",
            mime="text/csv",
            key="corr_dl_csv"
        )

        buf_png = io.BytesIO()
        cg.fig.savefig(buf_png, format="png", dpi=300, bbox_inches="tight")
        st.download_button(
            "📷 下载 PNG",
            buf_png.getvalue(),
            file_name="sample_correlation_heatmap.png",
            mime="image/png",
            key="corr_dl_png"
        )

        buf_svg = io.BytesIO()
        cg.fig.savefig(buf_svg, format="svg", bbox_inches="tight")
        st.download_button(
            "📸 下载 SVG",
            buf_svg.getvalue(),
            file_name="sample_correlation_heatmap.svg",
            mime="image/svg+xml",
            key="corr_dl_svg"
        )

    except ValueError as ve:
        st.error("❌ 相关性热图绘制失败：数据格式或数值异常")
        st.exception(ve)

    except Exception as e:
        st.error("❌ 相关性模块发生未知错误")
        st.exception(e)
