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
import sys
from pathlib import Path
sys.path.append('/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/bfs_recommendation.py')
from bfs_recommendation import construir_grafo_similitud, bfs_recomendaciones, mostrar_nube_plotly


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
    st.session_state.opcion = "🏠 Inicio"

st.sidebar.title("🛍️ Fashion Virtual Assistant")

if st.sidebar.button("🏠 Inicio"):
    st.session_state.opcion = "🏠 Inicio"

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


# Ejemplo de DataFrame con información de las prendas
data = {
    "clase": ["Accesorios", "Botas", "Vestido", "Tacón", "Sudadera", "Chaqueta", "Pantalones", "Camisetas", "Zapatillas", "Jersey", "Camiseta"],
    "ruta": ["/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/bota.png", 
             "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/bota.png",
             "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/vestido.png",
             "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/tacon.png",
             "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/sudadera.png",
             "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/chaqueta.jpg",
             "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/pant.png",
             "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/caisa.png",
             "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/teni.png",
             "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/jersey.png",
             "/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/static/img/cami.png"]
}
df = pd.DataFrame(data)

# Generar las características de las prendas (deberían ser características reales)
features = np.array([[0.2, 0.3, 0.5], [0.4, 0.6, 0.7], [0.5, 0.2, 0.4], [0.6, 0.8, 0.9], [0.1, 0.2, 0.3], 
                     [0.7, 0.6, 0.9], [0.8, 0.8, 0.5], [0.3, 0.4, 0.6], [0.5, 0.7, 0.6], [0.2, 0.1, 0.4], 
                     [0.4, 0.5, 0.8]])



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
    
# Función auxiliar para convertir imagen a base64
def bg_image_to_base64(image):
        from io import BytesIO
        import base64
        
        if image is None:
            return ""
            
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()


# -----------------------------
# 🧩 INTERFAZ PRINCIPAL
# -----------------------------
if "opcion" not in st.session_state:
    st.session_state.opcion = "🏠 Inicio"

if st.session_state.opcion == "🏠 Inicio":
    # Mostrar la imagen de fondo ocupando toda la pantalla
    st.image('/Users/carlotafernandez/Desktop/Code/FashionAI_Project/interfaz/revistas.png', use_container_width=True, output_format="PNG")

    # Cargar imagen (mejor práctica)
    try:
        bg_image = Image.open("interfaz/revistas.png")
    except:
        st.error("No se pudo cargar la imagen de fondo")
        bg_image = None

    # CSS personalizado
    st.markdown("""
    <style>
        /* Reset completo para contenedores de Streamlit */
        .main .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        
        /* Contenedor de imagen personalizado */
        .fullscreen-img-container {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            right: 0 !important;
            bottom: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            z-index: -2 !important;
            overflow: hidden !important;
        }
        
        /* Estilo para la imagen de fondo */
        .fullscreen-img {
            width: 100vw !important;
            height: 100vh !important;
            object-fit: cover !important;
            display: block !important;
        }
        
        /* Tarjeta de contenido */
        .content-card {
            background: rgba(255, 255, 255, 0.85) !important;
            border-radius: 15px !important;
            padding: 2.5rem !important;
            margin: 3rem auto !important;
            max-width: 1100px !important;
            position: relative !important;
            z-index: 1 !important;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1) !important;
        }
        
        /* Mejorar legibilidad del texto */
        .content-card h2 {
            color: #333 !important;
            margin-top: 0 !important;
        }
        
        .content-card ul {
            line-height: 1.8 !important;
        }
        
        .content-card code {
            background: rgba(0,0,0,0.05) !important;
            padding: 0.2em 0.4em !important;
            border-radius: 3px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="fullscreen-img-container">
        <img class="fullscreen-img" src="data:image/png;base64,{bg_image_to_base64(bg_image)}" alt="Fashion Background">
    </div>

    <div class="content-card">
        <h2>🛍️ Fashion Virtual Assistant</h2>
        <p>Bienvenido/a al <strong>Asistente Virtual de Moda</strong>. Este proyecto combina inteligencia artificial con visión por computador y procesamiento del lenguaje natural para ofrecerte una experiencia interactiva en el mundo de la moda.</p>

        <p>Aquí podrás:</p>

        <ul>
            <li>👗 <strong>Chatear</strong> con un asistente virtual entrenado para hablar sobre estilos, prendas y recomendaciones personalizadas.</li>
            <li>📸 <strong>Subir imágenes</strong> de ropa para recibir sugerencias de prendas similares.</li>
            <li>🔍 <strong>Visualizar un grafo de similitud</strong> que relaciona prendas según sus características visuales.</li>
        </ul>

        <hr>

        <p><strong>¿Qué tecnologías usamos?</strong></p>

        <ul>
            <li><code>Streamlit</code>: para crear esta interfaz web interactiva.</li>
            <li><code>TensorFlow</code>: para los modelos de clasificación y estilo.</li>
            <li><code>Rasa</code>: para el chatbot conversacional.</li>
            <li><code>OpenCV</code> y <code>scikit-learn</code>: para procesamiento de imágenes y similitud.</li>
            <li><code>NetworkX</code>: para construir y visualizar relaciones entre prendas.</li>
        </ul>

        <p>¡Explora las secciones del menú lateral y descubre cómo la inteligencia artificial puede transformar tu experiencia de moda!</p>
    </div>
    """, unsafe_allow_html=True)


elif st.session_state.opcion == "💬 Chatear con el bot":
    st.markdown(
        """
        <style>
        .pink-box
        {
        background:#ffe6f2;
        padding:1.4rem 1.6rem;
        border-radius:12px;
        box-shadow:0 3px 8px rgba(0,0,0,0.08);
        margin-top:1rem;
        }
        .pink-box p{margin:.35rem 0;font-weight:500;}
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown("## 💬 Chat con el Asistente Virtual de Moda")

    # Abrimos la caja antes de TODOS los elementos
    st.markdown('<div class="pink-box">', unsafe_allow_html=True)

    user_input = st.text_input("Escribe tu mensaje:", key="chat_input")
    if st.button("Enviar") and user_input:
        respuestas = send_message_to_rasa(user_input)
        for r in respuestas:
            st.markdown(f"<p>**🤖:** {r['text']}</p>", unsafe_allow_html=True)

    # Cerramos la caja
    st.markdown('</div>', unsafe_allow_html=True)



elif st.session_state.opcion == "📸 Recomendación de prendas":
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



elif st.session_state.opcion == "🔗 Ver grafo de similitud":
    st.markdown("## 🔗 Grafo de Similitud entre Prendas")
    
    try:
        st.write("✅ Datos cargados:", df.shape)
        st.write("✅ Features shape:", features.shape)
        
        top_k = st.slider("🔢 Número de conexiones por nodo (top_k)", 2, 10, 5)
        nodo_inicio = st.number_input("🔍 Nodo inicial para subgrafo (opcional)", min_value=0, max_value=len(df)-1, step=1, value=0)
        profundidad = st.slider("📏 Profundidad del subgrafo", 1, 3, 2)

        top_k = min(top_k, len(features))
        
        st.write("🛠️ Construyendo grafo...")
        G = construir_grafo_similitud(df, features, top_k=top_k)
        st.write("✅ Grafo construido con", len(G.nodes), "nodos y", len(G.edges), "aristas")

        st.write("📊 Mostrando grafo Plotly...")
        fig = mostrar_nube_plotly(df, G, start_node=nodo_inicio, depth=profundidad)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 👕 Detalles de una prenda del grafo")
        nodo_id = st.selectbox("Selecciona un nodo del subgrafo:", options=list(G.nodes), index=0)

        if nodo_id is not None:
            clase = df.iloc[nodo_id]["clase"]
            ruta = df.iloc[nodo_id]["ruta"]
            st.image(ruta, caption=f"Clase: {clase}", use_container_width=True)
            st.write(f"🧵 Clase: **{clase}**")

    except Exception as e:
        st.error(f"❌ Error al generar el grafo: {e}")

