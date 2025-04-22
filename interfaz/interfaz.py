import streamlit as st
import os, tempfile, requests
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------
# ⚙️ CONFIGURACIÓN DE LA APP
# -----------------------------
st.set_page_config(page_title="Fashion Virtual Assistant", layout="wide")

# -----------------------------
# 🎨 MENÚ LATERAL DE NAVEGACIÓN MEJORADO
# -----------------------------
st.markdown("""
    <style>
    /* Estilos globales para la barra lateral */
    .sidebar .sidebar-content {
        background-color: #f7f7f7;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 2px 2px 20px rgba(0, 0, 0, 0.1);
        font-family: 'Poppins', sans-serif;
    }

    /* Título de la barra lateral */
    .sidebar .sidebar-title {
        color: #6b4e58;
        font-size: 1.8rem;
        font-weight: bold;
        padding-bottom: 15px;
    }

    /* Estilo de las opciones del menú lateral */
    .sidebar .stRadio>div>label {
        font-size: 1.2rem;
        color: #6b4e58;
        font-weight: bold;
        text-transform: uppercase;
    }
    .sidebar .stRadio div {
        background-color: rgba(255, 240, 250, 0.9);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 15px;
        transition: transform 0.2s ease, background-color 0.3s ease;
    }
    .sidebar .stRadio div:hover {
        background-color: rgba(255, 220, 240, 1);
        transform: translateY(-3px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }

    /* Estilos para el icono de la radio */
    .sidebar .stRadio input[type="radio"] {
        margin-right: 12px;
    }

    /* Añadir un fondo agradable para el menú lateral */
    .sidebar {
        background-image: url('https://www.transparenttextures.com/patterns/asfalt.png');
        background-size: 50%;
    }

    /* Botón de selección en la barra lateral */
    .sidebar .stRadio div label:hover {
        color: #ff6f61;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🧭 Navegación")

# Menú lateral de opciones con íconos
opcion = st.sidebar.radio("¿Qué quieres hacer?", (
    "💬 Chatear con el bot",
    "📸 Recomendación de prendas",
    "🔗 Grafos de similitud"
))

# Mostrar la opción seleccionada
st.write(f"Opción seleccionada: {opcion}")

# -----------------------------
# 🎨 CARGAR EL HTML
# -----------------------------
import streamlit as st
import streamlit.components.v1 as components

# Lee el contenido del archivo HTML
with open("/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/forma2.html", "r") as f:
    html_content = f.read()

# Renderiza el HTML usando components.html
components.html(html_content, height=600, scrolling=True)

# -----------------------------
# 📁 CARGA DE DATOS Y MODELOS
# -----------------------------
base_dir = '/Users/carlotafernandez/Desktop/Code/FashionAI_Project/data/zara_dataset'
modelos_dir = '/Users/carlotafernandez/Desktop/Code/FashionAI_Project/modelos'

@st.cache_resource
def cargar_modelos_y_datos():
    # Leer imágenes y clases
    data = []
    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if not filename.startswith("."):
                    file_path = os.path.join(class_path, filename)
                    data.append([file_path, class_name])
    df = pd.DataFrame(data, columns=["ruta", "clase"])

    # Procesar imágenes
    def preprocess_image(image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)) / 255.0
        return img

    processed_data = [[preprocess_image(row["ruta"]), row["clase"]] for _, row in df.iterrows()]
    X = np.array([x[0] for x in processed_data], dtype=np.float32)
    y = LabelEncoder().fit_transform([x[1] for x in processed_data])
    y = to_categorical(y)

    # Cargar modelos y features
    fashion_model = tf.keras.models.load_model(os.path.join(modelos_dir, 'fashion_model.h5'))
    style_model = tf.keras.models.load_model(os.path.join(modelos_dir, 'style_model.h5'))
    X_features = np.load(os.path.join(modelos_dir, 'X_features.npy'))

    return df, fashion_model, style_model, X_features

df, fashion_model, style_model, X_features = cargar_modelos_y_datos()

# -----------------------------
# 🧠 FUNCIONES PRINCIPALES
# -----------------------------
def get_similar_items(uploaded_file):
    if not uploaded_file:
        return pd.DataFrame(), None

    # Guardar imagen temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    # Procesar imagen
    img = cv2.imread(temp_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)) / 255.0
    input_img = np.expand_dims(img, axis=0)

    # Extraer features y calcular similitud
    extractor = models.Model(inputs=fashion_model.input, outputs=fashion_model.layers[-2].output)
    features = extractor.predict(input_img)
    similarities = cosine_similarity(features, X_features)
    indices = np.argsort(similarities[0])[::-1][:5]

    # Predecir estilo
    style_pred = style_model.predict(input_img)
    style_label = np.argmax(style_pred)

    os.remove(temp_path)
    return df.iloc[indices], style_label

# -----------------------------
# 🗨️ INTERFAZ DE CHAT
# -----------------------------
if opcion == "💬 Chatear con el bot":
    st.markdown("## 💬 Chat con el Asistente Virtual de Moda")
    user_input = st.text_input("Escribe tu mensaje:", key="chat_input")
    
    if st.button("Enviar"):
        if user_input:
            respuestas = send_message_to_rasa(user_input)
            for r in respuestas:
                st.markdown(f"**🤖:** {r['text']}")

# -----------------------------
# 📷 RECOMENDACIÓN CON IMAGEN
# -----------------------------
elif opcion == "📸 Recomendación de prendas":
    st.markdown("## 📸 Recomendaciones de moda con imágenes")
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Tu imagen", use_container_width=True)
        st.write("🔍 Analizando imagen...")
        resultados, estilo = get_similar_items(uploaded_file)

        estilos = {0: "Casual", 1: "Formal", 2: "Sportive", 3: "Elegant", 4: "Urban", 5: "Vintage"}
        st.success(f"✨ Estilo identificado: **{estilos.get(estilo, 'Desconocido')}**")

        st.write("### 🧥 Recomendaciones similares:")
        cols = st.columns(5)
        for i, (_, item) in enumerate(resultados.iterrows()):
            with cols[i]:
                st.image(item["ruta"], caption=item["clase"], use_container_width=True)

# -----------------------------
# 🔗 GRAFOS DE SIMILITUD
# -----------------------------
elif opcion == "🔗 Grafos de similitud":
    st.markdown("## 🔗 Grafo de Similitud")
    
    # Crear grafo de similitud
    G = nx.Graph()
    for i in range(len(X_features)):
        for j in range(i + 1, len(X_features)):
            similarity = cosine_similarity([X_features[i]], [X_features[j]])[0][0]
            if similarity > 0.5:  # Filtra las aristas con una similitud alta
                G.add_edge(i, j, weight=similarity)
    
    # Asigna etiquetas de clase a los nodos
    for i in range(len(X_features)):
        G.nodes[i]['clase'] = df.iloc[i]['clase']
    
    # Función para mostrar el grafo en Streamlit
    def mostrar_grafo_streamlit(G, df):
        st.subheader("Visualizando el grafo de similitud")

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

        # Añadir leyenda para las clases
        for clase, color in color_map.items():
            plt.plot([], [], marker='o', color=color, linestyle='', label=clase)
        plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='upper right')

        st.pyplot(plt)

    # Mostrar el grafo de similitud
    mostrar_grafo_streamlit(G, df)

