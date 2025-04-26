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
import base64
import uuid
import sys
from pathlib import Path

# Añadir carpeta para imports personalizados
sys.path.append('/Users/carlotafernandez/Desktop/Code/FashionAI_Project')

from dataset_entrenado import get_similar_items
from bfs_recommendation import construir_grafo_similitud, bfs_recomendaciones, mostrar_nube_plotly
from info_prendas import info_prendas

# -----------------------------
# ⚙️ CONFIGURACIÓN DE SESIÓN
# -----------------------------
if "sender_id" not in st.session_state:
    st.session_state.sender_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        ("bot", "Hi! I'm Zairi, your fashion assistant. How can I help you today?")
    ]

if 'opcion' not in st.session_state:
    st.session_state.opcion = "🏠 Zairi"

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
st.sidebar.title("Fashion Virtual Assistant")

if st.sidebar.button("🏠 Zairi", key="sidebar_home"):
    st.session_state.opcion = "🏠 Zairi"
if st.sidebar.button("💬 Chat with the bot", key="sidebar_chat"):
    st.session_state.opcion = "💬 Chat with the bot"
if st.sidebar.button("📸 Clothing Recommendation", key="sidebar_recommendation"):
    st.session_state.opcion = "📸 Clothing Recommendation"
if st.sidebar.button("🔗 Similarity graph", key="sidebar_graph"):
    st.session_state.opcion = "🔗 Similarity graph"

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
                    data.append([file_path := os.path.join(class_path, filename), class_name])
    df = pd.DataFrame(data, columns=["ruta", "class"])

    def preprocess_image(image_path):
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224)) / 255.0
            return img
        except Exception as e:
            st.error(f"Error processing image {image_path}: {e}")
            return None

    processed_data = [[preprocess_image(row["ruta"]), row["class"]] for _, row in df.iterrows()]
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
# 📦 DATA PARA EL GRAFO
# -----------------------------
# 📦 DataFrame CORREGIDO para que coincida con info_prendas
data_info_prendas = {
    "class": ["boot", "shirt", "t-shirt", "jacket", "cap", "sweater", "pant", "sweatshirt", "heel", "dress", "sneakers"],
    "ruta": [
        "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/boot.png", 
        "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/shirt.png",
        "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/t-shirt.png",
        "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/jacket.png",
        "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/cap.png",
        "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/sweater.jpg",
        "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/pant.png",
        "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/hoodie.png",
        "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/heel.png",
        "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/dress.png",
        "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/sneaker.png"
    ]
}

df_info_prendas = pd.DataFrame(data_info_prendas)

# ⚡ Características aleatorias para el grafo
features_info_prendas = np.random.rand(len(df_info_prendas), 3)

# -----------------------------
# 🧠 FUNCIONES AUXILIARES
# -----------------------------
def send_message_to_rasa(message):
    url = "http://localhost:5005/webhooks/rest/webhook"
    try:
        response = requests.post(url, json={"sender": st.session_state.sender_id, "message": message.strip()})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Chatbot connection error: {e}")
        return [{"text": "❌ Error connecting to chatbot."}]

def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# -----------------------------
# 🖼️ CARGA DE FONDO SOLO EN "INICIO"
# -----------------------------
if st.session_state.opcion == "🏠 Zairi":
    background_path = "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/dios.png"
    background_image = get_base64_image(background_path)
    
    st.markdown(f"""
        <style>
        .background-img {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-image: url('data:image/png;base64,{background_image}');
            background-size: cover;
            background-position: center;
            z-index: -1;
        }}
        </style>
        <div class="background-img"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome-box" style="
        background-color: white;
        border-radius: 20px;
        padding: 3rem;
        max-width: 900px;
        margin: 8vh auto;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    ">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">Fashion Virtual Assistant</div>
        <div style="font-size: 1.3rem; color: #555; margin-bottom: 1.5rem;">AI meets your style.</div>
        <ul style="text-align: left; max-width: 600px; margin: auto; line-height: 1.6;">
            <li>🤖 Chat with Zairi, your AI stylist.</li>
            <li>📷 Upload your outfit and get style suggestions.</li>
            <li>🧠 Discover connections between pieces.</li>
        </ul>
        <br/>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💬 Chat with the bot", key="inicio_chat"):
            st.session_state.opcion = "💬 Chat with the bot"
    with col2:
        if st.button("📸 Clothing Recommendation", key="inicio_recomendacion"):
            st.session_state.opcion = "📸 Clothing Recommendation"
    with col3:
        if st.button("🔗 Similarity graph", key="inicio_grafo"):
            st.session_state.opcion = "🔗 Similarity graph"

# -----------------------------
# 💬 CHAT
# -----------------------------
elif st.session_state.opcion == "💬 Chat with the bot":
    st.markdown("## Chat with **Zairi** your virtual stylist 🤖✨")
    for speaker, text in st.session_state.chat_history:
        bubble_class = "chat-bubble-user" if speaker == "user" else "chat-bubble-bot"
        prefix = "" if speaker == "user" else "🤖 "
        st.markdown(f"<div class='chat-container'><div class='{bubble_class}'>{prefix}{text}</div></div>", unsafe_allow_html=True)

    input_text = st.text_input("Type your message:", key="chat_input_text", placeholder="What should I wear to a party?")
    send_button = st.button("Send")

    if send_button and input_text.strip():
        st.session_state.chat_history.append(("user", input_text.strip()))
        with st.spinner("Zairi is thinking..."):
            response = send_message_to_rasa(input_text.strip())
            for r in response:
                if "text" in r:
                    st.session_state.chat_history.append(("bot", r["text"]))
        st.session_state["chat_input_text"] = ""
        st.rerun()

# -----------------------------
# 📸 RECOMENDACIÓN POR IMAGEN
# -----------------------------
elif st.session_state.opcion == "📸 Clothing Recommendation":
    st.markdown("## 📸 Clothing Recommendation with images")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

    if uploaded_file and uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
        st.write("🔍 Analyzing image...")

        # Mostrar la imagen que el usuario sube
        st.image(uploaded_file, caption="🖼️ Your uploaded image", use_container_width=True)

        resultados, estilo_detectado = get_similar_items(uploaded_file)
        estilos = {0: "Casual", 1: "Formal", 2: "Sportive", 3: "Elegant", 4: "Urban", 5: "Vintage"}

        st.success(f"✨ Identified style: **{estilos.get(estilo_detectado, 'Unknown')}**")

        if not resultados.empty:
            # Detectar la prenda principal
            clase_detectada = resultados.iloc[0]["class"].strip().lower().rstrip("s")  # Normalizar nombre

            st.info(f"📌 Detected clothing item: **{clase_detectada}**")

            # Definir combinaciones (todo singular normalizado)
            combinaciones = {
                "jacket": ["pant", "heel", "cap"],
                "dress": ["heel", "sneaker", "cap"],
                "shirt": ["pant", "sneaker"],
                "t-shirt": ["sneaker", "pant"],
                "sweater": ["pant", "sneaker"],
                "pant": ["shirt", "t-shirt", "sweater"],
                "sneaker": ["pant", "t-shirt"],
                "heel": ["dress", "pant"],
                "boot": ["pant", "jacket"],
                "sweatshirt": ["pant", "sneaker"],
                "cap": ["jacket", "t-shirt"],
            }

            # Buscar clases recomendadas
            clases_recomendadas = combinaciones.get(clase_detectada.lower(), [])

            if not clases_recomendadas:
                st.warning("No complementary classes found for this item.")
            else:
                # Normalizar tu dataframe grande
                df["clase_normalizada"] = df["class"].str.strip().str.lower().str.rstrip("s")

                # Buscar recomendaciones reales en el Zara_dataset
                recomendaciones = df[df["clase_normalizada"].isin(clases_recomendadas)]

                if not recomendaciones.empty:
                    n_recomendar = min(5, len(recomendaciones))
                    recomendaciones = recomendaciones.sample(n=n_recomendar)

                    st.write("### 🧥 Complementary recommendations for your outfit:")
                    cols = st.columns(min(5, len(recomendaciones)))
                    for i, (_, item) in enumerate(recomendaciones.iterrows()):
                        ruta_imagen = item["ruta"]
                        if os.path.exists(ruta_imagen):
                            with cols[i % len(cols)]:
                                st.image(ruta_imagen, caption=item["class"], use_container_width=True)
                        else:
                            with cols[i % len(cols)]:
                                st.warning("Image not found.")
                else:
                    st.warning("No complementary recommendations found in the dataset.")
        else:
            st.warning("No similar item detected to recommend.")



# -----------------------------
# 🔗 GRAFO DE SIMILITUD
# -----------------------------
elif st.session_state.opcion == "🔗 Similarity graph":
    st.markdown("## 🔗 Similarity Graph between Garments")

    try:
        top_k = st.slider("🔢 Number of connections per node (top_k)", 2, 10, 5)

        # Asegurar que solo se usen imágenes válidas
        df_info_prendas = df_info_prendas[df_info_prendas["ruta"].apply(os.path.exists)].reset_index(drop=True)
        features_valid = features_info_prendas[:len(df_info_prendas)]

        profundidad = st.slider("📏 Subgraph depth", 1, 3, 2)
        top_k = min(top_k, len(features_valid))

        st.write("🛠️ Building graph...")
        G = construir_grafo_similitud(df_info_prendas, features_valid, top_k=top_k)
        st.write("✅ Graph constructed with", len(G.nodes), "nodes and", len(G.edges), "edges")

        # --------------------------
        # 🔍 BÚSQUEDA DE PRENDA
        # --------------------------
        busqueda = st.text_input("🔍 Search for a garment by name (e.g., boot, dress, sneaker)", "")

        # Normalización de búsqueda
        busqueda_normalizada = busqueda.strip().lower().rstrip("s")
        nodo_inicio = None
        for idx, clase in enumerate(df_info_prendas["class"]):
            if clase.strip().lower().rstrip("s") == busqueda_normalizada:
                nodo_inicio = idx
                break

        if nodo_inicio is None:
            st.warning("❌ No garment with that name was found.")
            nodo_inicio = 0
        else:
            st.success(f"📌 Showing graph centered on: **{df_info_prendas.iloc[nodo_inicio]['class']}** (node {nodo_inicio})")

        # Mostrar el grafo
        fig = mostrar_nube_plotly(df_info_prendas, G, start_node=nodo_inicio, depth=profundidad, nodo_destacado=nodo_inicio)
        st.plotly_chart(fig, use_container_width=True)

        # --------------------------
        # 🛍️ DETALLE DE LA PRENDA
        # --------------------------
        st.markdown("### 👕 Details of a garment from the graph")
        nodo_id = st.selectbox("Select a node from the subgraph:", options=list(G.nodes), index=0)

        if nodo_id is not None:
            clase_original = df_info_prendas.iloc[nodo_id]["class"]
            ruta = df_info_prendas.iloc[nodo_id]["ruta"]

            # Buscar información de la prenda en info_prendas
            info = info_prendas.get(clase_original, {
                "name": clase_original.capitalize(),
                "price": "Price not available",
                "color": "",
                "reference": "",
                "description": "Description not available.",
                "url": "https://www.zara.com/"
            })

            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(ruta, width=300)
            with col2:
                st.markdown(f"### {info['name']}")
                st.markdown(f"**{info['price']}**")
                if info.get("color"):
                    st.markdown(f"**Color:** {info['color']}  \n**Ref:** {info['reference']}")
                st.markdown("---")
                st.markdown(info['description'])
                st.markdown(f"[🛒 See in store]({info['url']})", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error generating graph: {e}")
