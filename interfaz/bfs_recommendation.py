import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from collections import deque
import pickle

def construir_grafo_similitud(df, features, top_k=5, min_sim=0.5):
    G = nx.Graph()
    for idx, row in df.iterrows():
        G.add_node(idx, clase=row["clase"], estilo=row.get("estilo", None))

    features = normalize(features)
    similarities = cosine_similarity(features)

    for i in range(len(df)):
        sim_indices = np.argsort(similarities[i])[::-1][1:top_k+1]
        for j in sim_indices:
            sim_val = similarities[i][j]
            if sim_val >= min_sim:
                G.add_edge(i, j, weight=sim_val)

    return G

def bfs_recomendaciones(grafo, nodo_inicio, profundidad_max=2):
    visitados = set()
    cola = deque([(nodo_inicio, 0)])

    while cola:
        nodo, profundidad = cola.popleft()
        if nodo not in visitados and profundidad <= profundidad_max:
            visitados.add(nodo)
            for vecino in grafo.neighbors(nodo):
                if vecino not in visitados:
                    cola.append((vecino, profundidad + 1))

    return list(visitados)

def mostrar_grafo_streamlit(G, df, nodo_inicio=None, profundidad_max=2):
    st.subheader("Visualizing the similarity graph")

    if nodo_inicio is not None and len(G) > 30:
        sub_nodos = bfs_recomendaciones(G, nodo_inicio=nodo_inicio, profundidad_max=profundidad_max)
        G = G.subgraph(sub_nodos)
        df = df.loc[sub_nodos]

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=0.6)

    clases = df["clase"].unique()
    color_map = {clase: plt.cm.get_cmap('hsv', len(clases))(i) for i, clase in enumerate(clases)}

    clase_dict = df["clase"].to_dict()
    node_colors = [color_map[clase_dict[n]] for n in G.nodes]
    node_sizes = [300 + 100 * G.degree(n) for n in G.nodes]
    edge_weights = [G[u][v]['weight'] * 10 for u, v in G.edges]

    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color="gray",
        width=edge_weights,
        alpha=0.85,
        font_size=8
    )

    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    for clase, color in color_map.items():
        plt.plot([], [], marker='o', color=color, linestyle='', label=clase)
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='upper right')

    st.pyplot(plt)
    plt.clf()
    plt.close()

def guardar_grafo(grafo, ruta):
    with open(ruta, 'wb') as f:
        pickle.dump(grafo, f)

def cargar_grafo(ruta):
    with open(ruta, 'rb') as f:
        return pickle.load(f)
