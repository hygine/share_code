# modules/venn_plot.py
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3


def venn_block(gene_sets: dict):
    """
    gene_sets: dict[str, set]
    支持 2~4 组
    """
    n = len(gene_sets)
    labels = list(gene_sets.keys())
    sets = list(gene_sets.values())

    if n < 2 or n > 4:
        st.error("Venn 图仅支持 2~4 组")
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    if n == 2:
        venn2(sets, set_labels=labels, ax=ax)
    elif n == 3:
        venn3(sets, set_labels=labels, ax=ax)
    else:
        # 4 组退化成 pairwise 展示（matplotlib_venn 不原生支持 4）
        st.warning("4 组暂以两两交集方式展示")
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                fig, ax = plt.subplots(figsize=(5, 5))
                venn2(
                    [sets[i], sets[j]],
                    set_labels=[labels[i], labels[j]],
                    ax=ax
                )
                st.pyplot(fig)
        return

    st.pyplot(fig)

    # =========================
    # 交集基因展示
    # =========================
    st.markdown("### 🧬 交集基因列表")

    if n == 2:
        inter = sets[0] & sets[1]
    elif n == 3:
        inter = sets[0] & sets[1] & sets[2]

    if not inter:
        st.info("无交集基因")
        return

    inter_genes = sorted(inter)
    st.write(f"交集基因数：{len(inter_genes)}")
    st.dataframe(inter_genes)

    # 下载
    csv = "\n".join(inter_genes)
    st.download_button(
        "📥 下载交集基因列表",
        csv,
        file_name="venn_intersection_genes.txt",
        mime="text/plain"
    )
