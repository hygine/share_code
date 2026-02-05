# modules/diff_analysis.py
from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from itertools import combinations
from upsetplot import UpSet, from_contents
import matplotlib.pyplot as plt

from utils.stats import diff_test
from modules.heatmap import heatmap_block


# =====================================================
# Streamlit helpers
# =====================================================
def _plotly_show(fig):
    """兼容 Streamlit 新参数：use_container_width -> width='stretch'"""
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def _has_pkg(import_name: str) -> bool:
    try:
        __import__(import_name)
        return True
    except Exception:
        return False


def _pip_install_hint(pkgs: List[str]) -> str:
    # 单行更利于复制
    return "pip install " + " ".join(pkgs)


def _need_python_pkgs(required: List[str], title: str = "依赖缺失") -> bool:
    """
    UI 级友好报错：告诉用户缺哪个包 + pip 命令
    返回 True 表示依赖齐全；False 表示缺失（但不强制 stop，方便用户切换方法）
    """
    missing = [p for p in required if not _has_pkg(p)]
    if not missing:
        return True

    st.error(f"❌ {title}：缺少 Python 包：{', '.join(missing)}")
    st.code(_pip_install_hint(missing), language="bash")
    return False


# =====================================================
# Counts detection / conversion helpers
# =====================================================
def _counts_diagnostics(mat: pd.DataFrame) -> Dict[str, Any]:
    """
    mat: genes × samples 数值矩阵
    """
    x = mat.values.astype(float)
    finite_mask = np.isfinite(x)
    x = x[finite_mask]
    if x.size == 0:
        return {
            "n_total": 0,
            "neg_count": 0,
            "neg_ratio": 0.0,
            "near_int_ratio": 0.0,
            "has_decimal": False,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
        }

    neg_count = int((x < 0).sum())
    n_total = int(x.size)

    frac = np.abs(x - np.rint(x))
    near_int_ratio = float((frac < 1e-6).sum() / n_total)
    has_decimal = bool((frac > 1e-6).any())

    return {
        "n_total": n_total,
        "neg_count": neg_count,
        "neg_ratio": float(neg_count / n_total),
        "near_int_ratio": near_int_ratio,
        "has_decimal": has_decimal,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }


def _prepare_counts_matrix(mat: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """
    strategy:
      - "不处理"
      - "clip负值并四舍五入"
      - "仅四舍五入"
      - "仅clip负值"
    """
    out = mat.copy()
    out = out.apply(pd.to_numeric, errors="coerce").fillna(0)

    if strategy in ["仅clip负值", "clip负值并四舍五入"]:
        out = out.clip(lower=0)

    if strategy in ["仅四舍五入", "clip负值并四舍五入"]:
        out = np.rint(out).astype(int)

    return out


def _align_two_groups(
    expr_df: pd.DataFrame,
    annotation_col: pd.DataFrame,
    group_col: str,
    g1: str,
    g2: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对齐表达矩阵与 annotation，并仅保留 g1/g2 两组样本。
    返回：
      mat: genes × samples
      coldata: samples × meta（含 group_col）
    """
    mat = expr_df.copy().apply(pd.to_numeric, errors="coerce")

    samples = mat.columns.intersection(annotation_col.index)
    if len(samples) < 4:
        raise ValueError("样本名必须匹配 annotation，且建议每组>=2（总样本>=4）")

    mat = mat[samples]
    coldata = annotation_col.loc[samples].copy()

    coldata = coldata[coldata[group_col].isin([g1, g2])]
    mat = mat[coldata.index]

    coldata[group_col] = pd.Categorical(coldata[group_col], categories=[g1, g2], ordered=True)

    if (coldata[group_col] == g1).sum() < 2 or (coldata[group_col] == g2).sum() < 2:
        raise ValueError("每组至少需要 2 个样本")

    return mat, coldata


def _ordered_samples_g1_g2(coldata: pd.DataFrame, group_col: str, g1: str, g2: str) -> List[str]:
    """用于热图：样本顺序按组排列（g1 在前 g2 在后）"""
    s1 = coldata.index[coldata[group_col] == g1].tolist()
    s2 = coldata.index[coldata[group_col] == g2].tolist()
    return s1 + s2


# =====================================================
# Python backends: PyDESeq2 / InMoose edgepy / InMoose limma
# =====================================================
def _run_pydeseq2(
    counts_df: pd.DataFrame,
    annotation_col: pd.DataFrame,
    group_col: str,
    g1: str,
    g2: str,
    alpha: float = 0.05,
    n_cpus: int = 8,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    DESeq2（Python: PyDESeq2）
    返回：(结果表, 过滤信息)
    结果列：log2FC / pvalue / padj
    """
    if not _need_python_pkgs(["pydeseq2"], title="PyDESeq2 未安装，无法运行 DESeq2"):
        raise RuntimeError("缺少 pydeseq2")

    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    mat, coldata = _align_two_groups(counts_df, annotation_col, group_col, g1, g2)

    # counts 约束：非负整数
    mat = mat.fillna(0).clip(lower=0)
    mat = np.rint(mat).astype(int)

    # PyDESeq2：counts 需要 samples×genes
    counts = mat.T
    meta = coldata[[group_col]].copy()

    # 过滤：给出“类似 filterByExpr 的提示”——这里采用一个更直观的阈值示意
    # 注意：真正 edgeR::filterByExpr 是自适应的；Python-only 先给可解释提示即可
    before_n = mat.shape[0]
    keep = (mat.sum(axis=1) > 1)
    after_n = int(keep.sum())

    # PyDESeq2 自带 independent filtering，这里的 keep 仅用于提示，不强制剔除（避免行为与 R 版不一致）
    # 若你想强制剔除：mat = mat.loc[keep]
    dds = DeseqDataSet(
        counts=counts,
        metadata=meta,
        design_factors=group_col,
        ref_level=[group_col, g1],
        n_cpus=int(max(1, n_cpus)),
    )
    dds.deseq2()

    stat = DeseqStats(
        dds,
        contrast=[group_col, g2, g1],
        alpha=float(alpha),
        n_cpus=int(max(1, n_cpus)),
        quiet=True,
    )
    stat.summary()

    res_df = stat.results_df.copy()
    # 统一列名
    out = pd.DataFrame(index=res_df.index)
    out["log2FC"] = res_df["log2FoldChange"].astype(float)
    out["pvalue"] = res_df["pvalue"].astype(float)
    out["padj"] = res_df["padj"].astype(float)

    filt = {
        "filter_method": "提示：sum(counts) > 1（展示用阈值）",
        "genes_before": int(before_n),
        "genes_after": int(after_n),
        "note": (
            "PyDESeq2 内部包含独立过滤/离群处理逻辑；此处的阈值仅用于帮助你理解低表达过滤的必要性。"
        ),
    }
    return out.sort_values("padj"), filt

def _run_inmoose_edger_like(
    counts_df: pd.DataFrame,
    annotation_col: pd.DataFrame,
    group_col: str,
    g1: str,
    g2: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    edgeR（Python: InMoose edgepy, GLM LRT）
    关键：design 必须是 patsy.DesignMatrix
    这个版本会“强制包装”design，彻底规避：design must be a patsy DesignMatrix
    """
    if not _need_python_pkgs(["inmoose", "patsy"], title="edgeR (InMoose edgepy) 依赖未安装"):
        raise RuntimeError("缺少 inmoose / patsy")

    import numpy as _np
    from patsy import dmatrix, DesignMatrix
    from inmoose.edgepy import DGEList, glmLRT, topTags

    # 对齐并仅保留 g1/g2
    mat, coldata = _align_two_groups(counts_df, annotation_col, group_col, g1, g2)

    # edgeR counts 约束：非负整数
    mat = mat.fillna(0).clip(lower=0)
    mat = _np.rint(mat).astype(int)

    # === 1) 生成 patsy DesignMatrix（不要 dataframe / values） ===
    # 用 C() 明确分类变量，避免 patsy/pandas 类型差异导致不一致
    dm = dmatrix(f"~ C({group_col})", data=coldata)

    # === 2) 强制确保类型为 patsy.DesignMatrix（硬修复） ===
    # 有些环境/版本可能会让 dm 变成 ndarray；这里强制重新包装
    if not isinstance(dm, DesignMatrix):
        dm = DesignMatrix(_np.asarray(dm), dm.design_info)

    # === 3) 跑 edgepy GLM pipeline ===
    # 注意：counts 建议直接传 DataFrame（文档写 counts Type: pd.DataFrame），更少类型坑
    dge = DGEList(
        counts=mat,          # ✅ 传 DataFrame（推荐）
        samples=coldata,     # ✅ sample 信息
        group=None,          # ✅ 不用 group 也可，design 已显式给出
        group_col=group_col,
    )

    before_n = int(mat.shape[0])

    # common dispersion + GLM fit + LRT
    dge.estimateGLMCommonDisp(design=dm)
    fit = dge.glmFit(design=dm)
    lrt = glmLRT(fit)

    tt = topTags(lrt, n=_np.inf).copy()

    # === 4) 统一输出列 ===
    cols = tt.columns.tolist()

    def pick(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in cols:
                return c
        return None

    fc_col = pick(["logFC", "log2FoldChange", "log2FC"])
    p_col = pick(["PValue", "P.Value", "pvalue", "p"])
    fdr_col = pick(["FDR", "padj", "adj_pvalue", "adj.P.Val"])

    if fc_col is None or p_col is None:
        raise RuntimeError(f"edgepy 输出列无法识别：{cols}")

    out = pd.DataFrame(index=tt.index)
    out["log2FC"] = tt[fc_col].astype(float)
    out["pvalue"] = tt[p_col].astype(float)
    out["padj"] = tt[fdr_col].astype(float) if fdr_col else out["pvalue"]

    filt = {
        "filter_method": "edgepy GLM LRT（Python-only）",
        "genes_before": before_n,
        "genes_after": before_n,
        "note": "design 强制包装为 patsy.DesignMatrix（彻底修复类型检查报错）",
    }
    return out.sort_values("padj"), filt


def _counts_to_logcpm(counts: pd.DataFrame, prior_count: float = 0.5) -> pd.DataFrame:
    """counts（genes×samples）→ log2CPM（genes×samples）"""
    x = counts.copy().astype(float)
    x = x.clip(lower=0)
    lib_size = x.sum(axis=0).replace(0, np.nan)
    cpm = (x + prior_count) / lib_size * 1e6
    return np.log2(cpm)


def _run_inmoose_limma(
    expr_df: pd.DataFrame,
    annotation_col: pd.DataFrame,
    group_col: str,
    g1: str,
    g2: str,
    trend: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    limma（Python: InMoose limma）
    - trend=False: 适用于 log2/normalized 连续值（蛋白组/芯片等）
    - trend=True : limma-trend（适用于 counts→logCPM 后的 RNA-seq；近似 voom 的一条成熟路线）
    返回：(结果表, 过滤信息)
    """
    if not _need_python_pkgs(["inmoose"], title="InMoose limma 依赖未安装，无法运行 limma"):
        raise RuntimeError("缺少 inmoose")

    from inmoose.limma import lmFit, eBayes, contrasts_fit, topTable

    mat, coldata = _align_two_groups(expr_df, annotation_col, group_col, g1, g2)
    mat = mat.fillna(0)

    # 设计矩阵：样本×2列（g1,g2）
    design = pd.get_dummies(coldata[group_col].astype(str), drop_first=False)
    # 确保列顺序 g1 在前 g2 在后
    for need in [g1, g2]:
        if need not in design.columns:
            raise RuntimeError(f"设计矩阵缺少分组列：{need}")
    design = design[[g1, g2]]

    # contrast：g2 - g1
    contrast = np.zeros((design.shape[1], 1), dtype=float)
    contrast[design.columns.get_loc(g2), 0] = 1.0
    contrast[design.columns.get_loc(g1), 0] = -1.0

    # limma 输入：genes×samples
    fit = lmFit(mat.values, design.values)
    fit2 = contrasts_fit(fit, contrast)
    fit2 = eBayes(fit2, trend=bool(trend))
    tt = topTable(fit2, number=np.inf, sort_by="P")
    tt = tt.copy()

    # 统一列名
    cols = tt.columns.tolist()
    if "logFC" not in cols or ("P.Value" not in cols and "pvalue" not in cols):
        # 不同版本可能是小写列名，做兼容
        pass

    def _pick_col(cols, candidates):
        for c in candidates:
            if c in cols:
                return c
        return None

    c_lfc = _pick_col(cols, ["logFC", "log2FC", "log2FoldChange"])
    c_p = _pick_col(cols, ["P.Value", "pvalue", "p_value", "p"])
    c_adj = _pick_col(cols, ["adj.P.Val", "FDR", "padj", "adj_pvalue"])

    if c_lfc is None or c_p is None:
        raise RuntimeError(f"limma 输出列无法识别：{cols}")

    out = pd.DataFrame(index=tt.index)
    out["log2FC"] = tt[c_lfc].astype(float)
    out["pvalue"] = tt[c_p].astype(float)
    out["padj"] = tt[c_adj].astype(float) if c_adj is not None else out["pvalue"]

    filt = {
        "filter_method": "none（建议自行过滤低表达/低方差）",
        "genes_before": int(mat.shape[0]),
        "genes_after": int(mat.shape[0]),
        "note": "limma-trend（counts→logCPM）" if trend else "limma（连续值：log2/normalized）",
    }
    return out.sort_values("padj"), filt


# =====================================================
# Main block
# =====================================================
def diff_block(
    df: pd.DataFrame,
    df_show: Optional[pd.DataFrame],
    annotation_col: Optional[pd.DataFrame] = None,
):
    st.subheader("🧪 多组合差异分析")

    # =====================================================
    # 差异结果统一仓库（核心）
    # =====================================================
    if "diff_results" not in st.session_state:
        st.session_state["diff_results"] = {}

    # =====================================================
    # 基本校验
    # =====================================================
    if annotation_col is None or annotation_col.empty:
        st.warning("请先上传或提供分组信息文件")
        return

    # =====================================================
    # 数据来源（全量 or 搜索结果）
    # =====================================================
    source = st.radio(
        "基因来源",
        ["全部", "搜索结果"],
        horizontal=True,
        key="diff_source",
    )
    plot_base = df
    if source == "搜索结果":
        if df_show is None or df_show.empty:
            st.warning("当前没有搜索结果，已切换为全部基因")
        else:
            plot_base = df_show

    # =====================================================
    # 分组设置
    # =====================================================
    group_col = st.selectbox("分组变量", annotation_col.columns, key="diff_group_col")

    groups = annotation_col[group_col].dropna().unique()
    if len(groups) < 2:
        st.warning("至少需要两个分组")
        return

    combo_labels = [f"{g1}_vs_{g2}" for g1, g2 in combinations(groups, 2)]
    selected_combos = st.multiselect(
        "选择要进行差异分析的组合",
        combo_labels,
        default=combo_labels[:1],
        key="diff_selected_combos",
    )

    # =====================================================
    # 方法选择（Python-only 成熟路线）
    # =====================================================
    method = st.selectbox(
        "差异分析方法（Python-only）",
        [
            "t-test",
            "wilcoxon",
            "DESeq2 (PyDESeq2, counts)",
            "edgeR (InMoose edgepy, counts)",
            "limma-trend (InMoose limma, counts->logCPM)",
            "limma (InMoose limma, log2/normalized)",
        ],
        key="diff_method",
    )

    is_counts_method = (
        method.startswith("DESeq2")
        or method.startswith("edgeR")
        or method.startswith("limma-trend")
    )

    # =====================================================
    # counts 自动检测与处理策略
    # =====================================================
    counts_strategy = "不处理"
    n_cpus = 8

    if is_counts_method:
        diag = _counts_diagnostics(plot_base)
        with st.expander("Counts 数据质量检测（RNA-seq counts 建议）", expanded=True):
            st.write(
                {
                    "near-integer 比例": f'{diag["near_int_ratio"]*100:.1f}%',
                    "负值比例": f'{diag["neg_ratio"]*100:.2f}%',
                    "最小值": diag["min"],
                    "最大值": diag["max"],
                    "均值": diag["mean"],
                }
            )

            if diag["neg_count"] > 0:
                st.warning("检测到负值：counts 方法不接受负值，建议 clip 到 0。")
            if diag["near_int_ratio"] < 0.98:
                st.warning("数据不是典型整数 counts（near-integer 比例较低）。如果是蛋白组/芯片/已 log2 强度，请选 limma（连续值）。")

            counts_strategy = st.selectbox(
                "counts 预处理策略（仅对 counts 方法生效）",
                ["不处理", "clip负值并四舍五入", "仅clip负值", "仅四舍五入"],
                index=1 if (diag["neg_count"] > 0 or diag["near_int_ratio"] < 0.98) else 0,
                key="diff_counts_strategy",
            )

            st.caption(
                "过滤提示：edgeR 的 filterByExpr 是自适应低表达过滤；Python-only 这里保留“提示逻辑”，实际过滤建议在上游明确阈值或按项目规范执行。"
            )

            # 多用户并发时不建议默认占满 64 核，给一个可控入口
            max_cpu = int(min(32, max(1, (os.cpu_count() or 8))))
            n_cpus = int(st.number_input("DESeq2 线程数（建议 4~16）", 1, max_cpu, 8, 1, key="diff_n_cpus"))

    # =====================================================
    # 阈值（筛显著：优先 padj/FDR）
    # =====================================================
    fc_cut = st.slider("Fold Change 阈值（|log2FC|）", 0.0, 3.0, 1.0, key="diff_fc_cut")
    p_cut = st.slider("显著性阈值（padj/FDR 优先；无则用 pvalue）", 0.0001, 0.1, 0.05, key="diff_p_cut")

    # =====================================================
    # Python 依赖检查（UI 友好提示）
    # =====================================================
    with st.expander("依赖检查（Python-only）", expanded=False):
        if method.startswith("DESeq2"):
            _need_python_pkgs(["pydeseq2"], title="DESeq2 (PyDESeq2) 依赖")
        elif method.startswith("edgeR"):
            _need_python_pkgs(["inmoose", "patsy"], title="edgeR (InMoose edgepy) 依赖")
        elif method.startswith("limma-trend") or method.startswith("limma "):
            _need_python_pkgs(["inmoose"], title="limma (InMoose limma) 依赖")

    # =====================================================
    # 执行差异分析
    # =====================================================
    if st.button("🚀 开始差异分析", key="diff_run_btn"):
        for label in selected_combos:
            g1, g2 = label.split("_vs_")

            run_mat = plot_base

            # counts 方法：应用预处理策略（保证不因负值/小数崩溃）
            if is_counts_method:
                run_mat = _prepare_counts_matrix(plot_base, counts_strategy)

            try:
                filt_info: Dict[str, Any] = {}

                if method in ["t-test", "wilcoxon"]:
                    res = diff_test(run_mat, annotation_col, group_col, g1, g2, method)
                    filt_info = {"filter_method": "none", "note": "Python 简单检验（用于探索/小样本）"}

                elif method.startswith("DESeq2"):
                    res, filt_info = _run_pydeseq2(
                        run_mat,
                        annotation_col,
                        group_col,
                        g1,
                        g2,
                        alpha=p_cut,
                        n_cpus=n_cpus,
                    )

                elif method.startswith("edgeR"):
                    res, filt_info = _run_inmoose_edger_like(
                        run_mat,
                        annotation_col,
                        group_col,
                        g1,
                        g2,
                    )

                elif method.startswith("limma-trend"):
                    # counts -> logCPM，然后 limma-trend
                    logcpm = _counts_to_logcpm(run_mat)
                    res, filt_info = _run_inmoose_limma(
                        logcpm,
                        annotation_col,
                        group_col,
                        g1,
                        g2,
                        trend=True,
                    )

                elif method.startswith("limma "):
                    res, filt_info = _run_inmoose_limma(
                        run_mat,
                        annotation_col,
                        group_col,
                        g1,
                        g2,
                        trend=False,
                    )

                else:
                    st.error(f"{label}：未知方法 {method}")
                    continue

            except Exception as e:
                st.error(f"{label} 运行失败：{e}")
                continue

            # -----------------------------
            # 列标准化识别
            # -----------------------------
            fc_col = "log2FC" if "log2FC" in res.columns else None

            # padj 优先
            if "padj" in res.columns:
                p_col = "padj"
            elif "FDR" in res.columns:
                p_col = "FDR"
            elif "adj.P.Val" in res.columns:
                p_col = "adj.P.Val"
            elif "pvalue" in res.columns:
                p_col = "pvalue"
            elif "P.Value" in res.columns:
                p_col = "P.Value"
            else:
                p_col = None

            if fc_col is None or p_col is None:
                st.error(f"{label}：无法识别 log2FC / (padj/pvalue) 列")
                continue

            sig_df = res[(res[fc_col].abs() >= fc_cut) & (res[p_col] <= p_cut)]

            st.session_state["diff_results"][label] = {
                "meta": {
                    "group_col": group_col,
                    "group1": g1,
                    "group2": g2,
                    "method": method,
                    "fc_cut": fc_cut,
                    "p_cut": p_cut,
                    "source": source,
                    "counts_strategy": counts_strategy if is_counts_method else "NA",
                },
                "filter": filt_info,
                "table": res,
                "sig_df": sig_df,
                "sig_genes": set(sig_df.index),
                "fc_col": fc_col,
                "p_col": p_col,
            }

            st.success(f"{label} 完成：显著基因 {len(sig_df)} 个")

    # =====================================================
    # 📦 差异结果管理中心
    # =====================================================
    if st.session_state["diff_results"]:
        st.markdown("---")
        st.markdown("## 📦 差异结果管理中心")

        summary = []
        for k, v in st.session_state["diff_results"].items():
            filt = v.get("filter", {})
            summary.append(
                {
                    "对比": k,
                    "分组": f'{v["meta"]["group1"]} vs {v["meta"]["group2"]}',
                    "方法": v["meta"]["method"],
                    "来源": v["meta"]["source"],
                    "counts处理": v["meta"].get("counts_strategy", "NA"),
                    "|log2FC| ≥": v["meta"]["fc_cut"],
                    "FDR/P ≤": v["meta"]["p_cut"],
                    "显著基因数": len(v["sig_genes"]),
                    "过滤": filt.get("filter_method", ""),
                    "过滤前": filt.get("genes_before", ""),
                    "过滤后": filt.get("genes_after", ""),
                }
            )

        summary_df = pd.DataFrame(summary)
        st.dataframe(summary_df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            del_key = st.selectbox(
                "删除指定差异结果",
                ["不删除"] + list(st.session_state["diff_results"].keys()),
                key="diff_del_key",
            )
            if st.button("🗑 删除该差异结果", key="diff_del_btn") and del_key != "不删除":
                del st.session_state["diff_results"][del_key]
                st.experimental_rerun()

        with col2:
            if st.button("⚠️ 清空全部差异结果", key="diff_clear_btn"):
                st.session_state["diff_results"].clear()
                st.experimental_rerun()

    # =====================================================
    # 📊 差异结果可视化
    # =====================================================
    if not st.session_state["diff_results"]:
        return

    st.markdown("---")
    st.markdown("## 📊 差异结果可视化")

    current = st.selectbox(
        "选择差异分析组合",
        list(st.session_state["diff_results"].keys()),
        key="diff_current",
    )
    r = st.session_state["diff_results"][current]
    res, sig_df = r["table"], r["sig_df"]
    fc_col, p_col = r["fc_col"], r["p_col"]
    g1 = r["meta"]["group1"]
    g2 = r["meta"]["group2"]

    # 过滤提示（counts 方法下更强调）
    filt_info = r.get("filter", {})
    if filt_info:
        with st.expander("过滤/预处理信息（counts 方法尤其重要）", expanded=False):
            st.write(f"过滤方法：{filt_info.get('filter_method', 'NA')}")
            if "genes_before" in filt_info and "genes_after" in filt_info:
                st.write(f"过滤前基因数：{filt_info.get('genes_before')}，过滤后基因数：{filt_info.get('genes_after')}")
            if "note" in filt_info and filt_info["note"]:
                st.info(filt_info["note"])
            st.write(f"counts 预处理策略：{r['meta'].get('counts_strategy', 'NA')}")

    viz_type = st.radio(
        "展示方式",
        ["火山图", "差异基因热图", "基因排名图"],
        horizontal=True,
        key="diff_viz_type",
    )

    if viz_type == "火山图":
        y = -np.log10(res[p_col].astype(float) + 1e-300)
        fig = px.scatter(
            res,
            x=fc_col,
            y=y,
            color=res[p_col] <= r["meta"]["p_cut"],
            hover_name=res.index,
        )
        fig.update_layout(xaxis_title=fc_col, yaxis_title=f"-log10({p_col})")
        _plotly_show(fig)

    elif viz_type == "差异基因热图":
        if sig_df.empty:
            st.info("无显著差异基因")
        else:
            # 仅上调/仅下调筛选
            direction = st.selectbox(
                "显著基因方向筛选",
                ["全部显著", "仅上调（log2FC>0）", "仅下调（log2FC<0）"],
                key="diff_heatmap_direction",
            )

            heat_sig = sig_df.copy()
            if direction.startswith("仅上调"):
                heat_sig = heat_sig[heat_sig[fc_col] > 0]
            elif direction.startswith("仅下调"):
                heat_sig = heat_sig[heat_sig[fc_col] < 0]

            if heat_sig.empty:
                st.info("该方向下无显著基因")
            else:
                genes = heat_sig.index.intersection(df.index)

                # 仅展示 g1/g2 两组样本，并按 g1 在前 g2 在后排序
                try:
                    mat_full, coldata = _align_two_groups(df, annotation_col, r["meta"]["group_col"], g1, g2)
                    sample_order = _ordered_samples_g1_g2(coldata, r["meta"]["group_col"], g1, g2)

                    df_sub = df.loc[genes, sample_order]
                    # df_show（搜索结果）也要同样子集化（但允许为空）
                    df_show_safe = df_show if (df_show is not None and not df_show.empty) else df.loc[[]]
                    df_show_sub = df_show_safe.loc[df_show_safe.index.intersection(genes), sample_order]

                    anno_sub = annotation_col.loc[sample_order]

                    heatmap_block(
                        df_sub,
                        df_show_sub,
                        anno_sub,
                    )
                except Exception as e:
                    st.error(f"热图子集化失败：{e}")

    elif viz_type == "基因排名图":
        ranked = res.sort_values(fc_col, ascending=False)
        topn = ranked.head(50)
        fig = px.bar(topn, x=topn.index, y=fc_col)
        fig.update_layout(xaxis_title="Gene", yaxis_title=fc_col)
        _plotly_show(fig)

    # =====================================================
    # 📤 导出
    # =====================================================
    st.markdown("---")
    st.markdown("## 📤 差异分析报告导出")

    st.download_button(
        "⬇ 下载当前对比完整差异结果",
        res.to_csv(),
        file_name=f"{current}_diff_full.csv",
        mime="text/csv",
        key="diff_dl_full",
    )

    st.download_button(
        "⬇ 下载当前对比显著差异基因",
        sig_df.to_csv(),
        file_name=f"{current}_diff_sig.csv",
        mime="text/csv",
        key="diff_dl_sig",
    )

    if st.checkbox("合并导出全部差异结果（CSV）", key="diff_dl_all_ck"):
        merged = []
        for k, v in st.session_state["diff_results"].items():
            tmp = v["table"].copy()
            tmp["comparison"] = k
            merged.append(tmp)

        merged_df = pd.concat(merged, axis=0)
        st.download_button(
            "⬇ 下载全部差异分析合并结果",
            merged_df.to_csv(),
            file_name="all_diff_results.csv",
            mime="text/csv",
            key="diff_dl_all",
        )

    # =====================================================
    # 📈 UpSet Plot
    # =====================================================
    st.markdown("---")
    st.markdown("## 📈 多差异显著基因交集分析（UpSet Plot）")

    diff_results = st.session_state["diff_results"]

    if len(diff_results) < 2:
        st.info("至少需要 2 个差异分析结果才能绘制 UpSet Plot")
        return

    selected_sets = st.multiselect(
        "选择用于 UpSet Plot 的差异分析结果",
        list(diff_results.keys()),
        default=list(diff_results.keys()),
        key="diff_upset_selected",
    )

    if len(selected_sets) < 2:
        st.warning("至少选择 2 组")
        return

    gene_sets = {
        k: diff_results[k]["sig_genes"]
        for k in selected_sets
        if len(diff_results[k]["sig_genes"]) > 0
    }

    if len(gene_sets) < 2:
        st.warning("所选差异结果中显著基因数量不足")
        return

    upset_data = from_contents(gene_sets)

    fig = plt.figure(figsize=(10, 6))
    UpSet(upset_data, show_counts=True, sort_categories_by="cardinality").plot(fig=fig)
    st.pyplot(fig)
