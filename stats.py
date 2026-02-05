# utils/stats.py

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests


def diff_test(
    expr_df: pd.DataFrame,
    annotation_col: pd.DataFrame,
    group_col: str,
    group1: str,
    group2: str,
    method: str = "t-test"
) -> pd.DataFrame:
    """
    差异分析主函数
    """

    # 样本分组
    samples1 = annotation_col.index[annotation_col[group_col] == group1]
    samples2 = annotation_col.index[annotation_col[group_col] == group2]

    common1 = expr_df.columns.intersection(samples1)
    common2 = expr_df.columns.intersection(samples2)

    if len(common1) < 2 or len(common2) < 2:
        raise ValueError("每组至少需要 2 个样本")

    log2fc = []
    pvals = []

    for gene, row in expr_df.iterrows():
        x1 = row[common1].astype(float)
        x2 = row[common2].astype(float)

        # log2 Fold Change
        fc = (x2.mean() + 1e-9) / (x1.mean() + 1e-9)
        log2fc.append(np.log2(fc))

        # 统计检验
        if method == "t-test":
            _, p = ttest_ind(x1, x2, equal_var=False)
        else:
            _, p = mannwhitneyu(
                x1,
                x2,
                alternative="two-sided"
            )

        pvals.append(p)

    res = pd.DataFrame(
        {
            "log2FC": log2fc,
            "pvalue": pvals
        },
        index=expr_df.index
    )

    # FDR 校正
    res["padj"] = multipletests(res["pvalue"], method="fdr_bh")[1]
    res["-log10P"] = -np.log10(res["pvalue"] + 1e-300)

    # 显著性标签
    res["significant"] = "NS"
    res.loc[(res["log2FC"] > 1) & (res["padj"] < 0.05), "significant"] = "Up"
    res.loc[(res["log2FC"] < -1) & (res["padj"] < 0.05), "significant"] = "Down"

    return res.sort_values("padj")
