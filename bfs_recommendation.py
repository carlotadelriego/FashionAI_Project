import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def construir_grafo_similitud(df, features, top_k=5):
    G = nx.Graph()

    for idx, row in df.iterrows():
        G.add_node(idx, clase=row["clase"], estilo=row.get("estilo", None))

    similarities = cosine_similarity(features)

    for i in range(len(df)):
        sim_indices = np.argsort(similarities[i])[::-1][1:top_k+1]
        for j in sim_indices:
            G.add_edge(i, j, weight=similarities[i][j])

    return G


def mostrar_grafo_streamlit(G, df):
    st.subheader("Visualizing the similarity graph")

    plt.figure(figsize=(12, 8))

    # Posiciones para layout
    pos = nx.spring_layout(G, seed=42)

    # Colores únicos por clase
    clases = df["clase"].unique()
    color_map = {clase: plt.cm.tab20(i / len(clases)) for i, clase in enumerate(clases)}
    node_colors = [color_map[df.loc[n]["clase"]] for n in G.nodes]

    # Tamaño de los nodos según el grado
    node_sizes = [300 + 100 * G.degree(n) for n in G.nodes]

    # Pesos para las aristas
    edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges]

    nx.draw(
        G, pos,
        with_labels=False,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color="gray",
        width=edge_weights,
        alpha=0.85
    )

    # Agregar leyenda
    for clase, color in color_map.items():
        plt.plot([], [], marker='o', color=color, linestyle='', label=clase)
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='upper right')

    st.pyplot(plt)
