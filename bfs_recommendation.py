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

    for i in range(len(df)):
        sim_indices = np.argsort(similarities[i])[::-1][1:top_k+1]
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
            if len(recomendaciones) >= top_k:
                break
            for vecino in grafo.neighbors(nodo):
                if vecino not in visitados:
                    cola.append((vecino, profundidad + 1))

    return recomendaciones[:top_k]

def mostrar_grafo_streamlit(G, df, nodo_inicio=None):
    st.subheader("Visualizing the similarity graph")

    # Mostrar solo un subgrafo si el grafo es muy grande
    if nodo_inicio is not None and len(G) > 30:
        sub_nodos = bfs_recomendaciones(G, nodo_inicio=nodo_inicio, top_k=20, profundidad_max=2)
        G = G.subgraph(sub_nodos)
        df = df.loc[sub_nodos]

    plt.figure(figsize=(12, 8))

    # Layout con más espacio entre nodos
    pos = nx.spring_layout(G, seed=42, k=0.6)

    clases = df["clase"].unique()
    color_map = {clase: plt.cm.get_cmap('hsv', len(clases))(i) for i, clase in enumerate(clases)}

    clase_dict = df["clase"].to_dict()
    node_colors = [color_map[clase_dict[n]] for n in G.nodes]
    node_sizes = [300 + 100 * G.degree(n) for n in G.nodes]
    edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges]

    nx.draw(
        G, pos,
        with_labels=True,  # Puedes cambiar a False si no quieres etiquetas
        node_color=node_colors,
        node_size=node_sizes,
        edge_color="gray",
        width=edge_weights,
        alpha=0.85,
        font_size=8
    )

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

import base64
import plotly.graph_objects as go
import networkx as nx

def mostrar_nube_plotly(df, G, start_node=0, depth=2):
    """
    Visualiza el grafo de similitud usando Plotly con información al pasar el cursor
    """
    # ---------- Subgrafo (BFS) ----------
    if start_node is not None:
        nodes_to_include = {start_node}
        frontier = [start_node]
        for _ in range(depth):
            next_frontier = []
            for node in frontier:
                neighbors = list(G.neighbors(node))
                next_frontier.extend(neighbors)
                nodes_to_include.update(neighbors)
            frontier = next_frontier
        subgraph = G.subgraph(nodes_to_include)
    else:
        subgraph = G

    # ---------- Layout ----------
    pos = nx.spring_layout(subgraph, seed=42)

    # ---------- Aristas ----------
    edge_x, edge_y = [], []
    for u, v in subgraph.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                            mode="lines",
                            line=dict(width=0.5, color="#888"),
                            hoverinfo="none")

    # ---------- Nodos ----------
    node_x, node_y = [], []
    node_color, node_text = [], []
    node_classes, node_imgs = [], []

    # mapa clase→color (Viridis)
    color_map = {c: i for i, c in enumerate(df["clase"].unique())}

    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        clase = df.iloc[node]["clase"]
        node_classes.append(clase)
        node_color.append(color_map[clase])
        node_text.append(clase)
        node_imgs.append(df.iloc[node]["ruta"])

    # encode imágenes
    encoded_imgs = []
    for p in node_imgs:
        try:
            with open(p, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()
        except Exception:
            encoded = ""
        encoded_imgs.append(encoded)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers",
        marker=dict(size=10,
                    colorscale="Viridis",
                    color=node_color,
                    showscale=True,
                    colorbar=dict(title="Categoría", thickness=15,
                                  xanchor="left", titleside="right"),
                    line_width=2),
        text=node_text,
        hoverinfo="text",
        hovertemplate=(
            '<img src="data:image/png;base64,%{customdata[2]}" height="100px"><br>'
            '<b>Clase:</b> %{customdata[0]}<br>'
            '<b>ID:</b> %{customdata[1]}<extra></extra>'),
        customdata=list(zip(node_classes,
                            list(range(len(node_classes))),
                            encoded_imgs))
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Grafo de Similitud entre Prendas",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    return fig
