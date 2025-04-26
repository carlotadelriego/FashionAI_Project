import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
import pickle
import base64
import plotly.graph_objects as go

def construir_grafo_similitud(df, features, top_k=5):
    G = nx.Graph()
    for i in range(len(features)):
        G.add_node(i, label=df.iloc[i]["class"], ruta=df.iloc[i]["ruta"])

    similarities = cosine_similarity(features)
    for i in range(len(features)):
        sim_indices = np.argsort(similarities[i])[::-1][1:]  # Excluimos el propio nodo (self-similarity)
        sim_indices = sim_indices[:min(top_k, len(sim_indices))]  # CORREGIDO aquí: top_k no más grande que posibles nodos

        for idx in sim_indices:
            G.add_edge(i, idx, weight=similarities[i][idx])
    return G


def bfs_recomendaciones(grafo, nodo_inicio, top_k=5, profundidad_max=2):
    """Devuelve una lista de nodos recomendados mediante BFS acotada en profundidad."""
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

def guardar_grafo(grafo, ruta):
    """Guarda el grafo en un archivo usando pickle."""
    with open(ruta, 'wb') as f:
        pickle.dump(grafo, f)

def cargar_grafo(ruta):
    """Carga un grafo desde un archivo usando pickle."""
    with open(ruta, 'rb') as f:
        return pickle.load(f)

def mostrar_nube_plotly(df, G, start_node=0, depth=2, nodo_destacado=None):
    """
    Visualiza el grafo de similitud usando Plotly con información al pasar el cursor
    """
    # Subgrafo (BFS) para limitar la visualización
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

    # Layout de los nodos
    pos = nx.spring_layout(subgraph, seed=42)

    # Aristas
    edge_x, edge_y = [], []
    for u, v in subgraph.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=0.5, color="#888"),
                            hoverinfo="none")

    # Nodos
    node_x, node_y = [], []
    node_color, node_text = [], []
    node_classes, node_imgs = [], []
    node_sizes = []

    color_map = {c: i for i, c in enumerate(df["class"].unique())}

    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        clase = df.iloc[node]["class"]
        node_classes.append(clase)
        node_text.append(clase)
        node_imgs.append(df.iloc[node]["ruta"])

        # Destacar nodo
        if node == nodo_destacado:
            node_color.append('red')  # sobrescribe el colorscale para ese nodo
            node_sizes.append(18)
        else:
            node_color.append(color_map[clase])
            node_sizes.append(10)

    # Codificar imágenes
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
        marker=dict(size=node_sizes, color=node_color, colorscale="Viridis", line_width=2),
        text=node_text,
        hoverinfo="text",
        hovertemplate=(
            '<img src="data:image/png;base64,%{customdata[2]}" height="100px"><br>'
            '<b>Clase:</b> %{customdata[0]}<br>'
            '<b>ID:</b> %{customdata[1]}<extra></extra>'),
        customdata=list(zip(node_classes, list(subgraph.nodes()), encoded_imgs))
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(text="Similarity Graph between Garments", font=dict(size=16)),
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    return fig
