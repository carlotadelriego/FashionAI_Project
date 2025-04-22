# -----------------------------
# 📦 IMPORTACIONES NECESARIAS
# -----------------------------
import streamlit as st
import os
import tempfile
import requests
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
# 🎨 ESTILO PERSONALIZADO
# -----------------------------
st.markdown("""
    <style>
    .stButton > button {
        font-size: 24px;
        height: 60px;
        width: 100%;
        margin: 10px 0;
    }
    .stButton > button:hover {
        background-color: #f1f1f1;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 🎛️ MENÚ LATERAL
# -----------------------------
if 'opcion' not in st.session_state:
    st.session_state.opcion = "💬 Chatear con el bot"

if st.sidebar.button("💬 Chatear con el bot"):
    st.session_state.opcion = "💬 Chatear con el bot"

if st.sidebar.button("📸 Recomendación de prendas"):
    st.session_state.opcion = "📸 Recomendación de prendas"

if st.sidebar.button("🔗 Ver grafo de similitud"):
    st.session_state.opcion = "🔗 Ver grafo de similitud"

# -----------------------------
# 📁 CARGA DE DATOS Y MODELOS
# -----------------------------
base_dir = '/Users/carlotafernandez/Desktop/Code/FashionAI_Project/data/zara_dataset'
modelos_dir = '/Users/carlotafernandez/Desktop/Code/FashionAI_Project/modelos'

@st.cache_resource
def cargar_modelos_y_datos():
    data = []
    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if not filename.startswith("."):
                    file_path = os.path.join(class_path, filename)
                    data.append([file_path, class_name])
    df = pd.DataFrame(data, columns=["ruta", "clase"])

    def preprocess_image(image_path):
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224)) / 255.0
            return img
        except Exception as e:
            st.error(f"Error procesando la imagen {image_path}: {e}")
            return None

    processed_data = [[preprocess_image(row["ruta"]), row["clase"]] for _, row in df.iterrows()]
    processed_data = [item for item in processed_data if item[0] is not None]
    X = np.array([x[0] for x in processed_data], dtype=np.float32)
    y = LabelEncoder().fit_transform([x[1] for x in processed_data])
    y = to_categorical(y)

    fashion_model = tf.keras.models.load_model(os.path.join(modelos_dir, 'fashion_model.h5'))
    style_model = tf.keras.models.load_model(os.path.join(modelos_dir, 'style_model.h5'))
    X_features = np.load(os.path.join(modelos_dir, 'X_features.npy'))

    return df, fashion_model, style_model, X_features

df, fashion_model, style_model, X_features = cargar_modelos_y_datos()

# -----------------------------
# 🧠 FUNCIONES
# -----------------------------
def get_similar_items(uploaded_file):
    if not uploaded_file:
        return pd.DataFrame(), None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        img = cv2.imread(temp_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)) / 255.0
        input_img = np.expand_dims(img, axis=0)

        extractor = models.Model(inputs=fashion_model.input, outputs=fashion_model.layers[-2].output)
        features = extractor.predict(input_img)
        similarities = cosine_similarity(features, X_features)
        indices = np.argsort(similarities[0])[::-1][:5]

        style_pred = style_model.predict(input_img)
        style_label = np.argmax(style_pred)

        os.remove(temp_path)
        return df.iloc[indices], style_label

    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
        return pd.DataFrame(), None

def send_message_to_rasa(message):
    url = "http://localhost:5005/webhooks/rest/webhook"
    try:
        response = requests.post(url, json={"sender": "user", "message": message})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error de conexión con el chatbot: {e}")
        return [{"text": "❌ Error al conectar con el chatbot."}]

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

    for clase, color in color_map.items():
        plt.plot([], [], marker='o', color=color, linestyle='', label=clase)
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='upper right')

    st.pyplot(plt)

# -----------------------------
# 🧩 INTERFAZ PRINCIPAL
# -----------------------------
opcion = st.session_state.opcion

if opcion == "💬 Chatear con el bot":
    st.markdown("## 💬 Chat con el Asistente Virtual de Moda")
    user_input = st.text_input("Escribe tu mensaje:", key="chat_input")
    if st.button("Enviar"):
        if user_input:
            respuestas = send_message_to_rasa(user_input)
            for r in respuestas:
                st.markdown(f"**🤖:** {r['text']}")

elif opcion == "📸 Recomendación de prendas":
    st.markdown("## 📸 Recomendaciones de moda con imágenes")
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png"])
    if uploaded_file and uploaded_file.type in ["image/jpeg", "image/png"]:
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
    else:
        st.warning("Por favor, sube una imagen JPG o PNG válida.")

elif opcion == "🔗 Ver grafo de similitud":
    st.markdown("## 🔗 Grafo de Similitud")
    G = nx.Graph()
    for i in range(len(X_features)):
        for j in range(i + 1, len(X_features)):
            sim = cosine_similarity([X_features[i]], [X_features[j]])[0][0]
            if sim > 0.5:
                G.add_edge(i, j, weight=sim)
    for i in range(len(X_features)):
        G.nodes[i]['clase'] = df.iloc[i]['clase']
    mostrar_grafo_streamlit(G, df)
