import streamlit as st
import re

def gene_search_block(df):
    st.write("### 🔍 基因搜索")

    search = st.text_input("支持多个基因（逗号）")
    df_show = df.copy()

    if search:
        genes = [g.strip() for g in re.split("[,\n]", search) if g.strip()]
        pattern = "|".join(re.escape(g) for g in genes)
        mask = df.index.to_series().str.contains(pattern, case=False, na=False)
        df_show = df[mask]

    page_size = st.selectbox("每页行数", [10, 20, 50], index=1)
    total = df_show.shape[0]
    pages = max((total - 1) // page_size + 1, 1)
    page = st.number_input("页码", 1, pages, 1)

    start = (page - 1) * page_size
    end = start + page_size

    st.dataframe(df_show.iloc[start:end], use_container_width=True)
    st.caption(f"{total} genes | Page {page}/{pages}")
    st.download_button(
       "下载搜索结果",
        df_show.to_csv(),
        file_name="gene_search_result.csv"
    )
    return df_show
