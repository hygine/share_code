import streamlit as st
from sklearn.cross_decomposition import PLSRegression

def plsda_block(df, annotation_col):
    st.subheader("🧭 PLS-DA")

    group_col = st.selectbox("分组", annotation_col.columns)
    y = annotation_col[group_col].astype("category").cat.codes

    pls = PLSRegression(n_components=2)
    X = df.T.values
    pls.fit(X, y)

    scores = pls.x_scores_
