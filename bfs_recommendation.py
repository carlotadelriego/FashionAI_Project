import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
import pickle

def construir_grafo_similitud(df, features, top_k=5):
    G = nx.Graph()

    for idx, row in df.iterrows():
        G.add_node(idx, clase=row["clase"], estilo=row.get("estilo", None))

    similarities = cosine_similarity(features)

    # Añadir aristas basadas en la similitud entre los elementos
    for i in range(len(df)):
        sim_indices = np.argsort(similarities[i])[::-1][1:top_k+1]  # top_k elementos más similares
        for j in sim_indices:
            G.add_edge(i, j, weight=similarities[i][j])

    return G

def bfs_recomendaciones(grafo, nodo_inicio, top_k=5, profundidad_max=2):
    visitados = set()
    cola = deque([(nodo_inicio, 0)])
    recomendaciones = []

    while cola:
        nodo, profundidad = cola.popleft()
        if nodo not in visitados and profundidad <= profundidad_max:
            visitados.add(nodo)
            recomendaciones.append(nodo)
            # Asegúrate de que no vayas más allá de `top_k` recomendaciones
            if len(recomendaciones) >= top_k:
                break
            for vecino in grafo.neighbors(nodo):
                if vecino not in visitados:
                    cola.append((vecino, profundidad + 1))

    return recomendaciones[:top_k]  # Asegura que solo haya `top_k` recomendaciones

def mostrar_grafo_streamlit(G, df):
    st.subheader("Visualizing the similarity graph")

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    clases = df["clase"].unique()
    color_map = {clase: plt.cm.tab20(i / len(clases)) for i, clase in enumerate(clases)}
    
    clase_dict = df["clase"].to_dict()
    node_colors = [color_map[clase_dict[n]] for n in G.nodes]
    node_sizes = [300 + 100 * G.degree(n) for n in G.nodes]
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

    for clase, color in color_map.items():
        plt.plot([], [], marker='o', color=color, linestyle='', label=clase)
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='upper right')

    st.pyplot(plt)

def guardar_grafo(grafo, ruta):
    with open(ruta, 'wb') as f:
        pickle.dump(grafo, f)

def cargar_grafo(ruta):
    with open(ruta, 'rb') as f:
        return pickle.load(f)
