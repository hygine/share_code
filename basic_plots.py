import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple


config = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d"
    ],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "plot",
        "height": 800,
        "width": 1200,
        "scale": 2
    }
}


def _plotly(fig):
    """统一 Plotly 渲染方式（兼容新旧 Streamlit，避免弃用警告）"""
    try:
        st.plotly_chart(
            fig,
            width="stretch",
            config=config
        )
    except TypeError:
        # 兼容老版本 Streamlit
        st.plotly_chart(
            fig,
            use_container_width=True,
            config=config
        )



def _align_expr_and_anno(df: pd.DataFrame, annotation_col: pd.DataFrame):
    """df: genes×samples；annotation_col: samples×meta。返回对齐后的 df_sub, anno_sub"""
    common = df.columns.intersection(annotation_col.index)
    if len(common) < 2:
        raise ValueError("表达矩阵列名与 annotation 样本名不匹配或交集过少")
    df_sub = df.loc[:, common]
    anno_sub = annotation_col.loc[common].copy()
    return df_sub, anno_sub


def barplot_block(df: pd.DataFrame, annotation_col: pd.DataFrame = None):
    st.subheader("📊 柱状图（单基因 / 分组柱状图 / 多基因柱状图）")

    try:
        if df is None or df.empty:
            st.warning("表达矩阵为空")
            return

        # -------------------------
        # 模式选择
        # -------------------------
        mode = st.radio(
            "柱状图类型",
            ["单基因-按样本", "单基因-按分组", "多基因-按分组", "多基因-按样本"],
            horizontal=True,
            key="bar_mode"
        )

        # -------------------------
        # 基因选择
        # -------------------------
        all_genes = df.index.tolist()

        if mode.startswith("单基因"):
            gene = st.selectbox("选择基因", all_genes, key="bar_gene")
            genes = [gene]
        else:
            genes = st.multiselect(
                "选择基因（建议 ≤ 20）",
                all_genes,
                default=all_genes[:5] if len(all_genes) >= 5 else all_genes,
                key="bar_genes"
            )
            if not genes:
                st.info("请选择至少一个基因")
                return
            if len(genes) > 50:
                st.warning("基因数过多（>50）会影响可读性与性能，建议减少")
                return

        # -------------------------
        # 数值处理
        # -------------------------
        with st.expander("⚙️ 数值处理", expanded=False):
            do_log2 = st.checkbox("log2(x+1) 转换", value=False, key="bar_log2")
            clip_neg = st.checkbox("clip 负值到 0（蛋白组/强度一般不需要）", value=False, key="bar_clipneg")

        # 取子矩阵
        mat = df.loc[genes].apply(pd.to_numeric, errors="coerce")
        if clip_neg:
            mat = mat.clip(lower=0)
        if do_log2:
            mat = np.log2(mat + 1)

        # -------------------------
        # 单基因：按样本
        # -------------------------
        if mode == "单基因-按样本":
            s = mat.loc[genes[0]].dropna()
            plot_df = pd.DataFrame({"sample": s.index, "value": s.values})

            # 可选：按 annotation 上色
            if annotation_col is not None and not annotation_col.empty:
                try:
                    _, anno = _align_expr_and_anno(df.loc[[genes[0]]], annotation_col)
                    group_col = st.selectbox("按 annotation 分组上色", [None] + anno.columns.tolist(), key="bar_single_colorby")
                    if group_col:
                        plot_df["group"] = anno.loc[plot_df["sample"], group_col].astype(str).values
                except Exception:
                    pass

            fig = px.bar(
                plot_df,
                x="sample",
                y="value",
                color="group" if "group" in plot_df.columns else None,
                labels={"sample": "样本", "value": "表达量"},
                title=f"{genes[0]} 表达量（按样本）"
            )
            fig.update_layout(xaxis_tickangle=60)
            _plotly(fig)
            return

        # -------------------------
        # 分组相关：检查 annotation
        # -------------------------
        if annotation_col is None or annotation_col.empty:
            st.warning("当前选择的模式需要 annotation 分组信息，请先上传 annotation")
            return

        # 对齐 df 与 annotation
        df_aligned, anno = _align_expr_and_anno(df.loc[genes], annotation_col)

        group_col = st.selectbox("分组变量", anno.columns.tolist(), key="bar_group_col")
        anno[group_col] = anno[group_col].astype(str)

        # 整理 long format
        long_df = df_aligned.T.reset_index().rename(columns={"index": "sample"})
        long_df[group_col] = anno.loc[long_df["sample"], group_col].values
        long_df = long_df.melt(
            id_vars=["sample", group_col],
            var_name="gene",
            value_name="value"
        )
        long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

        # -------------------------
        # 单基因：按分组
        # -------------------------
        if mode == "单基因-按分组":
            agg = st.selectbox("聚合方式", ["mean", "median"], index=0, key="bar_agg_single")
            err_type = st.selectbox("误差条", ["无", "SEM", "SD"], index=1, key="bar_err_single")
            show_points = st.checkbox("叠加样本散点", value=True, key="bar_points_single")

            one = long_df[long_df["gene"] == genes[0]].copy()

            # 聚合
            g = one.groupby(group_col)["value"]
            if agg == "mean":
                center = g.mean()
            else:
                center = g.median()

            if err_type == "SEM":
                err = g.sem()
            elif err_type == "SD":
                err = g.std()
            else:
                err = None

            sum_df = pd.DataFrame({group_col: center.index, "value": center.values})
            if err is not None:
                sum_df["err"] = err.values

            fig = px.bar(
                sum_df,
                x=group_col,
                y="value",
                error_y="err" if err_type != "无" else None,
                labels={group_col: "分组", "value": "表达量"},
                title=f"{genes[0]} 表达量（按分组：{group_col}）"
            )

            if show_points:
                fig2 = px.strip(
                    one.dropna(),
                    x=group_col,
                    y="value"
                )
                for tr in fig2.data:
                    fig.add_trace(tr)

            _plotly(fig)
            return

        # -------------------------
        # 多基因：按分组（x=group, color=gene）
        # -------------------------
        if mode == "多基因-按分组":
            agg = st.selectbox("聚合方式", ["mean", "median"], index=0, key="bar_agg_multi")
            err_type = st.selectbox("误差条", ["无", "SEM", "SD"], index=0, key="bar_err_multi")
            barmode = st.selectbox("柱状排列方式", ["group", "stack"], index=0, key="bar_barmode_multi")
            topn = st.slider("可选：只显示 Top N 基因（按组间方差）", 0, min(30, len(genes)), 0, key="bar_topn_var")

            work = long_df.copy()

            # 可选：按组间方差筛 TopN 基因（增强可读性）
            if topn and topn > 0 and len(genes) > topn:
                var_by_gene = (
                    work.groupby(["gene", group_col])["value"].mean().reset_index()
                    .groupby("gene")["value"].var()
                    .sort_values(ascending=False)
                )
                keep_genes = var_by_gene.head(topn).index.tolist()
                work = work[work["gene"].isin(keep_genes)]

            grp = work.groupby([group_col, "gene"])["value"]

            if agg == "mean":
                center = grp.mean()
            else:
                center = grp.median()

            sum_df = center.reset_index().rename(columns={"value": "center"})

            if err_type == "SEM":
                sum_df["err"] = grp.sem().reset_index(drop=True)
            elif err_type == "SD":
                sum_df["err"] = grp.std().reset_index(drop=True)

            fig = px.bar(
                sum_df,
                x=group_col,
                y="center",
                color="gene",
                barmode=barmode,
                error_y="err" if err_type != "无" else None,
                labels={group_col: "分组", "center": "表达量"},
                title=f"多基因表达量（按分组：{group_col}）"
            )
            _plotly(fig)
            return

        # -------------------------
        # 多基因：按样本（facet 或者 color）
        # -------------------------
        if mode == "多基因-按样本":
            display = st.selectbox("展示方式", ["分面（推荐）", "同图多色"], index=0, key="bar_multi_sample_display")
            max_samples = st.slider("最多显示样本数（太多会挤）", 10, 300, 180, key="bar_max_samples")

            # 样本过多时，限制显示
            sample_order = long_df["sample"].unique().tolist()
            if len(sample_order) > max_samples:
                st.info(f"样本数较多（{len(sample_order)}），已仅展示前 {max_samples} 个样本（按原顺序）。")
                keep_samples = set(sample_order[:max_samples])
                work = long_df[long_df["sample"].isin(keep_samples)].copy()
            else:
                work = long_df.copy()

            # 加入分组信息（用于 hover）
            work[group_col] = work[group_col].astype(str)

            if display == "分面（推荐）":
                fig = px.bar(
                    work,
                    x="sample",
                    y="value",
                    facet_row="gene",
                    color=group_col,
                    labels={"sample": "样本", "value": "表达量"},
                    title=f"多基因表达量（按样本，按 {group_col} 上色）"
                )
                fig.update_layout(xaxis_tickangle=60)
                _plotly(fig)
            else:
                fig = px.bar(
                    work,
                    x="sample",
                    y="value",
                    color="gene",
                    hover_data=[group_col],
                    labels={"sample": "样本", "value": "表达量"},
                    title="多基因表达量（按样本，同图多色）"
                )
                fig.update_layout(xaxis_tickangle=60, barmode="group")
                _plotly(fig)

            return

    except Exception as e:
        st.error("❌ 柱状图绘制失败")
        st.exception(e)


def lineplot_block(df: pd.DataFrame, annotation_col: pd.DataFrame = None):
    st.subheader("📈 多基因表达趋势")

    try:
        genes = st.multiselect(
            "选择基因",
            df.index.tolist(),
            default=df.index.tolist()[:3],
            key="line_genes"
        )

        if not genes:
            st.info("请选择至少一个基因")
            return

        mat = df.loc[genes].T.apply(pd.to_numeric, errors="coerce")

        # 可选：按 annotation 上色
        color_by = None
        if annotation_col is not None and not annotation_col.empty:
            try:
                _, anno = _align_expr_and_anno(df.loc[genes], annotation_col)
                color_by = st.selectbox("按 annotation 分组上色（可选）", [None] + anno.columns.tolist(), key="line_colorby")
                if color_by:
                    mat[color_by] = anno.loc[mat.index, color_by].astype(str).values
            except Exception:
                color_by = None

        fig = px.line(
            mat.reset_index().rename(columns={"index": "sample"}),
            x="sample",
            y=genes,
            color=color_by if color_by else None,
            labels={"sample": "样本", "value": "表达量"}
        )

        fig.update_layout(xaxis_tickangle=60)
        _plotly(fig)

    except Exception as e:
        st.error("❌ 折线图绘制失败")
        st.exception(e)


def violin_block(df: pd.DataFrame, annotation_col: pd.DataFrame):
    st.subheader("🎻 Violin 图（分组表达分布 + 统计检验）")

    # -------------------------
    # 内部工具：统计检验与星号
    # -------------------------
    def _p_to_star(p: float) -> str:
        if p is None or (not np.isfinite(p)):
            return "na"
        if p < 1e-4:
            return "****"
        if p < 1e-3:
            return "***"
        if p < 1e-2:
            return "**"
        if p < 5e-2:
            return "*"
        return "ns"

    def _adjust_pvals(pvals: list, method: str) -> list:
        pvals = np.array([float(p) if p is not None else np.nan for p in pvals], dtype=float)
        n = np.sum(np.isfinite(pvals))
        if n == 0:
            return pvals.tolist()

        if method == "不校正":
            return pvals.tolist()

        if method == "Bonferroni":
            out = pvals.copy()
            out[np.isfinite(out)] = np.minimum(out[np.isfinite(out)] * n, 1.0)
            return out.tolist()

        # BH / FDR
        if method == "BH(FDR)":
            out = pvals.copy()
            idx = np.where(np.isfinite(out))[0]
            pv = out[idx]
            order = np.argsort(pv)
            ranked = pv[order]
            m = len(ranked)
            q = ranked * m / (np.arange(1, m + 1))
            # monotonicity
            q = np.minimum.accumulate(q[::-1])[::-1]
            q = np.clip(q, 0, 1)
            out_idx = idx[order]
            out[out_idx] = q
            return out.tolist()

        return pvals.tolist()

    def _pairwise_tests(values_df: pd.DataFrame, group_col: str, test: str, compare_mode: str,
                        ref_group: Optional[str], max_pairs: int) -> pd.DataFrame:
        """
        values_df columns: group, value
        返回：pairwise 比较结果表
        """
        try:
            from scipy import stats
        except Exception:
            st.error("缺少依赖 scipy，无法进行统计检验。请安装：pip install scipy")
            return pd.DataFrame()

        gvals = {}
        for g, sub in values_df.groupby(group_col):
            arr = pd.to_numeric(sub["value"], errors="coerce").dropna().values.astype(float)
            if arr.size > 0:
                gvals[str(g)] = arr

        groups = list(gvals.keys())
        if len(groups) < 2:
            return pd.DataFrame()

        pairs = []
        if compare_mode == "仅与参考组比较":
            if ref_group is None or str(ref_group) not in gvals:
                return pd.DataFrame()
            rg = str(ref_group)
            for g in groups:
                if g == rg:
                    continue
                pairs.append((rg, g))
        else:
            # 全部两两
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    pairs.append((groups[i], groups[j]))

        # 限制对比数量，防止组数太多
        if len(pairs) > max_pairs:
            pairs = pairs[:max_pairs]

        out = []
        for a, b in pairs:
            x = gvals[a]
            y = gvals[b]

            # 检验选择
            p = np.nan
            stat = np.nan

            try:
                if test == "t-test":
                    stat, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
                elif test == "Mann–Whitney U":
                    stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
                else:
                    stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
            except Exception:
                p = np.nan

            out.append({"group1": a, "group2": b, "pvalue": float(p) if np.isfinite(p) else np.nan})

        return pd.DataFrame(out)

    def _kw_test(values_df: pd.DataFrame, group_col: str) -> Optional[float]:
        try:
            from scipy import stats
        except Exception:
            st.error("缺少依赖 scipy，无法进行统计检验。请安装：pip install scipy")
            return None

        arrays = []
        for g, sub in values_df.groupby(group_col):
            arr = pd.to_numeric(sub["value"], errors="coerce").dropna().values.astype(float)
            if arr.size > 0:
                arrays.append(arr)
        if len(arrays) < 2:
            return None
        try:
            _, p = stats.kruskal(*arrays)
            return float(p)
        except Exception:
            return None

    # -------------------------
    # 主逻辑
    # -------------------------
    try:
        if annotation_col is None or annotation_col.empty:
            st.warning("Violin 图需要 annotation 分组信息")
            return

        genes = st.multiselect(
            "选择基因（建议 ≤ 20）",
            df.index.tolist(),
            default=df.index.tolist()[:3],
            key="violin_genes"
        )
        if not genes:
            st.info("请选择至少一个基因")
            return
        if len(genes) > 50:
            st.warning("基因数过多（>50）会严重影响可读性，建议减少")
            return

        group_col = st.selectbox(
            "分组变量",
            annotation_col.columns.tolist(),
            key="violin_group_col"
        )

        with st.expander("⚙️ 数值处理", expanded=False):
            do_log2 = st.checkbox("log2(x+1) 转换", value=False, key="violin_log2")
            clip_neg = st.checkbox("clip 负值到 0", value=False, key="violin_clipneg")

        mat = df.loc[genes].apply(pd.to_numeric, errors="coerce")
        if clip_neg:
            mat = mat.clip(lower=0)
        if do_log2:
            mat = np.log2(mat + 1)

        common = mat.columns.intersection(annotation_col.index)
        if len(common) < 2:
            st.error("表达矩阵与 annotation 样本名不匹配")
            return

        mat = mat[common]
        anno = annotation_col.loc[common].copy()
        anno[group_col] = anno[group_col].astype(str)

        long_df = mat.T.reset_index().rename(columns={"index": "sample"})
        long_df[group_col] = anno.loc[long_df["sample"], group_col].values
        long_df = long_df.melt(id_vars=["sample", group_col], var_name="gene", value_name="value")
        long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

        display = st.radio(
            "展示方式",
            ["单图（颜色区分基因）", "分面（每个基因一行，推荐）"],
            horizontal=True,
            key="violin_display"
        )
        show_box = st.checkbox("显示 box（中位数/IQR）", value=True, key="violin_show_box")
        show_points = st.checkbox("显示样本散点（jitter）", value=True, key="violin_show_points")

        # -------------------------
        # ✅ 统计检验设置（新增）
        # -------------------------
        with st.expander("🧪 统计检验 + 星号标注", expanded=True):
            enable_stats = st.checkbox("启用统计检验", value=True, key="violin_enable_stats")
            test = st.selectbox("两组检验方法", ["Mann–Whitney U", "t-test"], index=0, key="violin_test")
            p_adjust = st.selectbox("多重校正", ["BH(FDR)", "Bonferroni", "不校正"], index=0, key="violin_p_adjust")

            compare_mode = st.selectbox(
                "组间比较策略",
                ["仅与参考组比较", "全部两两比较（组多时不建议）"],
                index=0,
                key="violin_compare_mode"
            )

            groups = sorted(long_df[group_col].dropna().astype(str).unique().tolist())
            ref_group = None
            if compare_mode == "仅与参考组比较":
                ref_group = st.selectbox("参考组（作为 baseline）", groups, index=0, key="violin_ref_group")

            max_pairs = st.slider("最多显示的比较对数（防止过密）", 1, 30, 12, key="violin_max_pairs")

            star_use = st.selectbox(
                "星号基于哪种 p 值",
                ["padj（校正后）", "pvalue（原始）"],
                index=0,
                key="violin_star_use"
            )

        # -------------------------
        # 作图
        # -------------------------
        if display == "单图（颜色区分基因）":
            fig = px.violin(
                long_df,
                x=group_col,
                y="value",
                color="gene",
                box=show_box,
                points="all" if show_points else False,
                hover_data=["sample"],
                labels={group_col: "分组", "value": "表达量"},
                title="Violin 图（分组表达分布）"
            )
            # 单图模式：不做星号（因为多个基因叠在一个坐标系，标注会很乱）
            if enable_stats:
                st.info("单图（多基因叠加）模式下星号标注容易混乱，建议使用“分面模式”以获得清晰标注。")
            _plotly(fig)

        else:
            fig = px.violin(
                long_df,
                x=group_col,
                y="value",
                color=group_col,
                facet_row="gene",
                box=show_box,
                points="all" if show_points else False,
                hover_data=["sample"],
                labels={group_col: "分组", "value": "表达量"},
                title="Violin 图（分面展示）"
            )
            fig.update_layout(height=max(320, 240 * len(genes)))

            # -------------------------
            # ✅ 统计检验 + 星号标注（分面模式）
            # -------------------------
            if enable_stats:
                # 对每个基因分别做检验
                all_anno = []
                gene_results = []

                for gi in genes:
                    sub = long_df[long_df["gene"] == gi].dropna(subset=["value", group_col]).copy()
                    if sub.empty:
                        continue

                    uniq_groups = sorted(sub[group_col].astype(str).unique().tolist())
                    if len(uniq_groups) < 2:
                        continue

                    # 先整体检验（>2 组时给一个 KW p 值提示）
                    kw_p = None
                    if len(uniq_groups) > 2:
                        kw_p = _kw_test(sub, group_col)

                    # pairwise
                    pw = _pairwise_tests(
                        sub[[group_col, "value"]],
                        group_col=group_col,
                        test=test,
                        compare_mode=compare_mode,
                        ref_group=ref_group,
                        max_pairs=max_pairs
                    )
                    if pw.empty:
                        continue

                    # 校正
                    pw["padj"] = _adjust_pvals(pw["pvalue"].tolist(), p_adjust)

                    # 选择用于星号的值
                    star_p = pw["padj"] if star_use.startswith("padj") else pw["pvalue"]
                    pw["star"] = [ _p_to_star(p) for p in star_p.tolist() ]
                    pw["gene"] = gi
                    if kw_p is not None:
                        pw["kw_p"] = kw_p
                    gene_results.append(pw)

                    # ---- 在图上加 annotation（Plotly facet 的 axis 很多，采用 paper 坐标在每个 facet 左上角放一行摘要）
                    # 这里采用“摘要式标注”：仅显示前 N 对比（按 padj/pvalue 排序）
                    pw_show = pw.copy()
                    sort_col = "padj" if star_use.startswith("padj") else "pvalue"
                    pw_show = pw_show.sort_values(sort_col, ascending=True).head(min(6, len(pw_show)))

                    # 拼一行简短文本
                    pieces = []
                    for _, rr in pw_show.iterrows():
                        pieces.append(f'{rr["group1"]} vs {rr["group2"]}: {rr["star"]}')
                    txt = " | ".join(pieces)

                    # facet 的 yaxis 名称：按 plotly 生成顺序，最稳定做法是遍历 layout.annotations 里 facet 标签定位
                    # 我们用“在图上方增加一行 gene->text 的表格式输出”，同时也在下方输出可下载表，避免 facet 精确定位太复杂。
                    all_anno.append({"gene": gi, "summary": txt, "KW_p(>2 groups)": kw_p})

                if gene_results:
                    st.markdown("### 🧾 统计检验结果（摘要）")
                    st.dataframe(pd.DataFrame(all_anno), use_container_width=True)

                    full_stats = pd.concat(gene_results, ignore_index=True)
                    st.download_button(
                        "📥 下载统计检验结果（CSV）",
                        full_stats.to_csv(index=False).encode(),
                        file_name="violin_stats_results.csv",
                        mime="text/csv",
                        key="violin_dl_stats"
                    )

                    # 在图整体顶部加一行提示（避免 facet 定位不稳定）
                    fig.add_annotation(
                        x=0, y=1.08, xref="paper", yref="paper",
                        text="统计标注请见下方「统计检验结果（摘要）」表（分面内精确放置星号在 Plotly facet 中不稳定，采用摘要方式更可靠）",
                        showarrow=False,
                        align="left"
                    )

            _plotly(fig)

    except Exception as e:
        st.error("❌ Violin 图绘制失败")
        st.exception(e)

