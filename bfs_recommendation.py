# bfs_recommendation.py
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque

def construir_grafo_similitud(vectores, umbral=0.85):
    grafo = {i: [] for i in range(len(vectores))}
    for i in range(len(vectores)):
        for j in range(i + 1, len(vectores)):
            sim = cosine_similarity([vectores[i]], [vectores[j]])[0][0]
            if sim >= umbral:
                grafo[i].append(j)
                grafo[j].append(i)
    return grafo

def guardar_grafo(grafo, ruta='grafo_similitud.pkl'):
    with open(ruta, 'wb') as f:
        pickle.dump(grafo, f)

def cargar_grafo(ruta='grafo_similitud.pkl'):
    with open(ruta, 'rb') as f:
        return pickle.load(f)

def bfs_recomendaciones(grafo, nodo_inicio, profundidad_max=2):
    visitados = set()
    cola = deque([(nodo_inicio, 0)])
    resultado = []

    while cola:
        actual, profundidad = cola.popleft()
        if actual in visitados or profundidad > profundidad_max:
            continue
        visitados.add(actual)
        resultado.append(actual)
        for vecino in grafo[actual]:
            cola.append((vecino, profundidad + 1))

    return resultado[1:6]  
