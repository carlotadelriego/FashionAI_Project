# -----------------------------
# 📦 IMPORTACIONES NECESARIAS
# -----------------------------
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

# -----------------------------
# ⚙️ CONFIGURACIÓN DE LA APP
# -----------------------------
st.set_page_config(page_title="Fashion Virtual Assistant", layout="wide")

# -----------------------------
# 🎨 MENÚ LATERAL DE NAVEGACIÓN
# -----------------------------
st.sidebar.title("🧭 Navegación")
opcion = st.sidebar.radio("¿Qué deseas hacer?", (
    "💬 Chatear con el bot",
    "📸 Recomendación de prendas",
    "🔗 Grafos de similitud"
))

# -----------------------------
# 💅 ESTILOS HTML Y COMPONENTES
# -----------------------------
st.markdown("""
<style>
    body {
        background: linear-gradient(to bottom, #ffffff, #fdfdfd);
        font-family: 'Poppins', sans-serif;
    }
    .container-style {
        background: rgba(254, 243, 253, 0.978); 
        border-radius: 20px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .chat-header {
        background: rgba(255, 220, 235, 0.8);
        color: #6b4e58;
        text-align: center;
        padding: 15px;
        border-radius: 15px;
        font-size: 2rem;
        margin-bottom: 10px;
    }
    .chat-description, .instructions {
        padding: 10px 20px;
        background-color: rgba(255, 240, 250, 0.7);
        border-radius: 10px;
        text-align: center;
    }
    .features-wrapper {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 10px;
    }
    .feature {
        background-color: rgba(255, 245, 230, 0.8);
        padding: 15px;
        border-radius: 15px;
        width: 30%;
        min-width: 200px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
    }
    .feature:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    }
</style>

<div class="container-style">
    <div class="chat-header">Fashion Virtual Assistant</div>
    <div class="chat-description">
        Your personal assistant to explore the latest trends and find the perfect style for every occasion.
    </div>

    <div class="features-wrapper">
        <div class="feature">
            <h3><a href="https://www.elle.com/es/moda/tendencias/a63952675/tendencias-de-moda-2025-que-mas-rejuvenecen/">👗 Personalized Recommendations</a></h3>
            <p>Receive fashion suggestions tailored to your personal style and preferences.</p>
        </div>
        <div class="feature">
            <h3><a href="https://www.vogue.es/articulos/tendencias-primavera-verano-2025-que-se-lleva">🔥 Current Trends</a></h3>
            <p>Stay up-to-date with the latest trends and styles in the fashion world.</p>
        </div>
        <div class="feature">
            <h3><a href="https://www.instyle.es/moda/como-vestir-bien_60185">💡 Styling Tips</a></h3>
            <p>Get professional advice on combining clothes and creating stunning looks.</p>
        </div>
    </div>

    <div class="instructions">
        Use the chat or image upload to explore your ideal style with the assistant.
    </div>
</div>
""", unsafe_allow_html=True)

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

def send_message_to_rasa(message):
    url = "http://localhost:5005/webhooks/rest/webhook"
    response = requests.post(url, json={"sender": "user", "message": message})
    if response.status_code == 200:
        return response.json()
    else:
        return [{"text": "❌ Error al conectar con el chatbot."}]

# -----------------------------
# 🗨️ INTERFAZ DE CHAT
# -----------------------------
if opcion == "💬 Chatear con el bot":
    st.markdown("## 💬 Chat with Fashion Virtual Assistant")
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
    st.markdown("## 📸 Fashion recommendations with images")
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
