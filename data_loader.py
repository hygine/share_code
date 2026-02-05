import streamlit as st
import pandas as pd

def load_expression():
    st.sidebar.header("数据上传")
    file = st.sidebar.file_uploader(
        "上传表达矩阵 (CSV/TSV)",
        type=["csv", "tsv"]
    )
    if not file:
        return None

    sep = "\t" if file.name.endswith(".tsv") else ","
    df = pd.read_csv(file, sep=sep, index_col=0)
    df.index = df.index.astype(str).str.strip()
    return df


def load_annotation():
    st.sidebar.header("样本注释（annotation bar）")
    file = st.sidebar.file_uploader(
        "上传 annotation 文件",
        type=["csv", "tsv"]
    )
    if not file:
        return None

    sep = "\t" if file.name.endswith(".tsv") else ","
    return pd.read_csv(file, sep=sep, index_col=0)
