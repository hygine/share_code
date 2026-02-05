import streamlit as st
from sklearn.cluster import KMeans

def kmeans_time_block(df):
    st.subheader("⏱ 时间序列聚类")

    k = st.slider("聚类数", 2, 10, 4)
    model = KMeans(n_clusters=k)
    labels = model.fit_predict(df)

    df["cluster"] = labels
    st.dataframe(df)
