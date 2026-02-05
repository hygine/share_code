import streamlit as st
import networkx as nx
import plotly.graph_objects as go

def network_block(edge_df):
    G = nx.from_pandas_edgelist(edge_df, "gene1", "gene2")

    pos = nx.spring_layout(G)
