import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
import numpy as np
import cv2
import tempfile
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import requests  # Para hacer peticiones HTTP al servidor de Rasa

# --- INTERFAZ STREAMLIT ---
st.set_page_config(page_title="Fashion Assistant 👗", layout="wide")

# --- CARGA MODELOS Y DATOS ---
base_dir = '/Users/carlotafernandez/Desktop/Code/FashionAI_Project/data/zara_dataset'
modelos_dir = '/Users/carlotafernandez/Desktop/Code/FashionAI_Project/modelos'

@st.cache_resource
def cargar_modelos_y_datos():
    data = []
    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                file_path = os.path.join(class_path, filename)
                if not filename.startswith("."):
                    data.append([file_path, class_name])
    df = pd.DataFrame(data, columns=["ruta", "clase"])

    def preprocess_image(image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        return img

    processed_data = []
    for _, row in df.iterrows():
        img = preprocess_image(row["ruta"])
        if img is not None:
            processed_data.append([img, row["clase"]])

    X = np.array([x[0] for x in processed_data], dtype=np.float32)
    y = np.array([x[1] for x in processed_data])
    y = LabelEncoder().fit_transform(y)
    y = to_categorical(y)

    fashion_model = tf.keras.models.load_model(os.path.join(modelos_dir, 'fashion_model.h5'))
    style_model = tf.keras.models.load_model(os.path.join(modelos_dir, 'style_model.h5'))
    X_features = np.load(os.path.join(modelos_dir, 'X_features.npy'))

    return df, fashion_model, style_model, X_features

df, fashion_model, style_model, X_features = cargar_modelos_y_datos()

# --- FUNCIONES ---
def get_similar_items(uploaded_file):
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        img = cv2.imread(temp_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)) / 255.0
        input_img = np.expand_dims(img, axis=0)

        feature_extractor = models.Model(inputs=fashion_model.input, outputs=fashion_model.layers[-2].output)
        features = feature_extractor.predict(input_img)
        similarities = cosine_similarity(features, X_features)
        indices = np.argsort(similarities[0])[::-1][:5]

        style_pred = style_model.predict(input_img)
        style_label = np.argmax(style_pred)

        os.remove(temp_path)
        return df.iloc[indices], style_label
    return pd.DataFrame(), None

def send_message_to_rasa(message):
    """Función para enviar un mensaje al chatbot de Rasa y obtener una respuesta."""
    url = "http://localhost:5005/webhooks/rest/webhook"  # URL del servidor de Rasa
    headers = {"Content-Type": "application/json"}
    payload = {"sender": "user", "message": message}
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        return [{"text": "Hubo un error en la comunicación con el chatbot."}]

# Mostrar HTML visual
html_path = "forma.html"
with open(html_path, 'r', encoding='utf-8') as f:
    html = f.read()
components.html(html, height=700, scrolling=True)

# Chatbot con Rasa
st.markdown("## 💬 Chat con el Asistente de Moda")
user_input = st.text_input("Escribe tu mensaje:", key="chat_input")

if user_input:
    response = send_message_to_rasa(user_input)
    st.write("🤖 Rasa dice:")
    for message in response:
        st.write(message["text"])


# Subida de imagen
st.markdown("## 📸 Sube una imagen para recomendaciones de moda")
uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Tu imagen", use_container_width=True)

    st.write("🔍 Analizando imagen...")
    similar_items, style_label = get_similar_items(uploaded_file)

    style_dict = {0: "Casual", 1: "Formal", 2: "Sportive", 3: "Elegant", 4: "Urban"}
    st.success(f"✨ Estilo identificado: **{style_dict.get(style_label, 'Desconocido')}**")

    st.write("### 🧥 Recomendaciones similares:")
    cols = st.columns(5)
    for i, (_, item) in enumerate(similar_items.iterrows()):
        with cols[i]:
            st.image(item['ruta'], caption=item['clase'], use_container_width=True)
