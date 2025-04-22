import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import plotly.graph_objects as go
from collections import deque


# Función para construir el grafo de similitud
def construir_grafo_similitud(df, features, top_k=5, min_sim=0.4):
    G = nx.Graph()

    # Añadir nodos con atributos
    for idx, row in df.iterrows():
        G.add_node(idx, clase=row["clase"], estilo=row.get("estilo", None))

    # Normalizar las características
    features = normalize(features)

    # Calcular similitud del coseno entre las características
    similarities = cosine_similarity(features)

    # Añadir aristas entre los nodos basados en la similitud
    for i in range(len(df)):
        sim_indices = np.argsort(similarities[i])[::-1][1:top_k+1]
        for j in sim_indices:
            sim_val = similarities[i][j]
            if sim_val >= min_sim:
                G.add_edge(i, j, weight=sim_val)
    
    return G


# Función BFS para obtener recomendaciones de prendas similares
def bfs_recommendations(G, start_node, max_depth=2):
    visited = set()  # Conjunto de nodos visitados
    if isinstance(start_node, np.ndarray):
        start_node = start_node.item()  # Extrae el único valor si es un array de tamaño 1
    queue = deque([(start_node, 0)])  # Cola de BFS (nodo, profundidad)
    bfs_result = []

    while queue:
        node, depth = queue.popleft()

        # Si el nodo ya fue visitado o hemos alcanzado la profundidad máxima, lo ignoramos
        if node in visited or depth > max_depth:
            continue

        visited.add(node)
        bfs_result.append(node)

        # Añadir los nodos vecinos a la cola con profundidad aumentada
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                queue.append((neighbor, depth + 1))

    return bfs_result


# Función para mostrar la nube de puntos interactiva utilizando Plotly
import plotly.graph_objects as go
import networkx as nx

def mostrar_nube_plotly(df, G, start_node):
    # Posiciones para los nodos generadas con el layout de NetworkX
    pos = nx.spring_layout(G, seed=42)

    # Asignamos las posiciones a los nodos
    for node in G.nodes:
        G.nodes[node]['x'] = pos[node][0]
        G.nodes[node]['y'] = pos[node][1]

    # Creamos las listas con las posiciones, clases y rutas de las imágenes
    x_values = [G.nodes[n]['x'] for n in G.nodes]
    y_values = [G.nodes[n]['y'] for n in G.nodes]
    clases = [df.iloc[n]['clase'] for n in G.nodes]
    rutas = [df.iloc[n]['ruta'] for n in G.nodes]  # Aquí tomamos la ruta de la imagen de cada nodo

    # Definimos los nodos en el gráfico
    node_trace = go.Scatter(
        x=x_values,
        y=y_values,
        mode='markers',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            color=[1 if n == start_node else 0 for n in G.nodes],  # Marca los nodos recomendados
            colorbar=dict(title="Recomendaciones", thickness=15, xanchor="left")
        ),
        hoverinfo='text',
        text=[f"{clase}" for clase in clases],  # Mostramos solo la clase al hacer hover
        customdata=list(zip(clases, rutas)),  # Guardamos clase y ruta como datos extra
    )

    # Creamos las aristas del gráfico
    edge_x = []
    edge_y = []
    for edge in G.edges:
        x0, y0 = G.nodes[edge[0]]['x'], G.nodes[edge[0]]['y']
        x1, y1 = G.nodes[edge[1]]['x'], G.nodes[edge[1]]['y']
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=0.5, color='gray'),
        hoverinfo='none'
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, t=0, l=0, r=0),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        clickmode='event+select'  # Necesario para capturar clics
    )

    return fig
