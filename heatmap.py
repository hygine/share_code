import streamlit as st
from utils.annotation import prepare_annotation, build_annotation_colors
from utils.clustering import pheatmap_like

def heatmap_block(df, df_show, annotation_col):
    st.subheader("🔥 表达量聚类热图")

    try:
        # ✅ 解决 DuplicateElementId：所有 widget 都加 key
        source = st.radio(
            "基因来源",
            ["全部", "搜索结果"],
            horizontal=True,
            key="heatmap_source"
        )
        plot_data = df_show if source == "搜索结果" else df

        if plot_data.shape[0] == 0:
            st.warning("没有可用于绘图的基因")
            return

        if plot_data.shape[0] > 500:
            st.warning("基因数过多，请缩小范围")
            return

        z_score = st.selectbox(
            "Z-score",
            ["不标准化", "按行", "按列"],
            key="heatmap_zscore"
        )
        cluster_rows = st.checkbox(
            "行聚类",
            True,
            key="heatmap_cluster_rows"
        )
        cluster_cols = st.checkbox(
            "列聚类",
            True,
            key="heatmap_cluster_cols"
        )

        # =========================
        # ✅ 标签显示优化（新增）
        # =========================
        with st.expander("🔧 标签/画布显示优化（解决文字挤在一起）", expanded=True):
            show_row_names = st.checkbox("显示行名（基因名）", value=True, key="heatmap_show_row_names")
            show_col_names = st.checkbox("显示列名（样本名）", value=True, key="heatmap_show_col_names")

            col_rotate = st.selectbox("列名旋转角度", [0, 45, 60, 90], index=3, key="heatmap_col_rotate")

            # 根据维度给一个“合理默认”的抽样间隔
            n_genes = int(plot_data.shape[0])
            n_samples = int(plot_data.shape[1])

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

            col_step = st.slider(
                "列名显示间隔（每 N 个显示一个）",
                min_value=1, max_value=max(1, min(20, n_samples)),
                value=_default_step(n_samples),
                key="heatmap_col_step"
            )
            row_step = st.slider(
                "行名显示间隔（每 N 个显示一个）",
                min_value=1, max_value=max(1, min(50, n_genes)),
                value=_default_step(n_genes),
                key="heatmap_row_step"
            )

            # 字体大小：给默认值，同时可调
            col_font = st.slider("列名字体大小", 4, 14, 7, key="heatmap_col_font")
            row_font = st.slider("行名字体大小", 4, 14, 7, key="heatmap_row_font")

            # 画布大小：自动 + 可微调
            auto_size = st.checkbox("自动调节画布大小", value=True, key="heatmap_auto_size")
            width_scale = st.slider("宽度系数（仅用于自动尺寸）", 0.6, 2.5, 1.2, key="heatmap_w_scale")
            height_scale = st.slider("高度系数（仅用于自动尺寸）", 0.6, 2.5, 1.2, key="heatmap_h_scale")

        plot_data, anno_used = prepare_annotation(plot_data, annotation_col)

        cg = pheatmap_like(
            plot_data,
            annotation_col=anno_used,
            z_score={"不标准化": None, "按行": "row", "按列": "col"}[z_score],
            cluster_rows=cluster_rows,
            cluster_cols=cluster_cols
        )

        # =========================
        # ✅ 关键：对 heatmap 的 tick label 做“抽样+旋转+字体”控制
        # =========================
        import matplotlib.pyplot as plt

        ax = getattr(cg, "ax_heatmap", None)
        if ax is not None:
            # ----- 列名（x）
            xticklabels = ax.get_xticklabels()
            if not show_col_names:
                ax.set_xticklabels([])
                ax.set_xlabel("")
            else:
                # 每 col_step 个显示一个，其余置空
                for i, lab in enumerate(xticklabels):
                    if col_step > 1 and (i % col_step != 0):
                        lab.set_text("")
                ax.set_xticklabels(xticklabels, rotation=col_rotate, ha="right" if col_rotate else "center", fontsize=col_font)

            # ----- 行名（y）
            yticklabels = ax.get_yticklabels()
            if not show_row_names:
                ax.set_yticklabels([])
                ax.set_ylabel("")
            else:
                for i, lab in enumerate(yticklabels):
                    if row_step > 1 and (i % row_step != 0):
                        lab.set_text("")
                ax.set_yticklabels(yticklabels, rotation=0, fontsize=row_font)

            # tick 线条也适当变细一点
            ax.tick_params(axis="both", length=2, width=0.5)

        # =========================
        # ✅ 自动画布尺寸（避免标签溢出/挤压）
        # =========================
        try:
            if auto_size:
                # 简单经验：宽度跟样本数相关，高度跟基因数相关
                # 给一个上限防止图过大
                w = min(30.0, max(10.0, (n_samples * 0.22) * width_scale))
                h = min(30.0, max(8.0, (n_genes * 0.12) * height_scale))
                cg.fig.set_size_inches(w, h)
            cg.fig.tight_layout()
        except Exception:
            pass

        st.pyplot(cg.fig, clear_figure=True)

        # =========================
        # 导出
        # =========================
        import io

        buf_png = io.BytesIO()
        cg.fig.savefig(buf_png, format="png", dpi=300, bbox_inches="tight")
        st.download_button(
            "📷 下载 PNG",
            buf_png.getvalue(),
            file_name="heatmap_pheatmap_style.png",
            mime="image/png",
            key="heatmap_download_png"
        )

        buf_svg = io.BytesIO()
        cg.fig.savefig(buf_svg, format="svg", bbox_inches="tight")
        st.download_button(
            "📸 下载 SVG",
            buf_svg.getvalue(),
            file_name="heatmap_pheatmap_style.svg",
            mime="image/svg+xml",
            key="heatmap_download_svg"
        )

    except ValueError as ve:
        st.error("❌ 热图绘制失败：数据格式或数值异常")
        st.exception(ve)

    except Exception as e:
        st.error("❌ 热图模块发生未知错误")
        st.exception(e)
