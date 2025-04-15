import streamlit as st
import streamlit.components.v1 as components
import requests
import random
from PIL import Image
import tempfile
import os
import sys
import numpy as np

# === Añadir ruta para importar dataset_entrenado desde carpeta "modelos" ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
modelos_path = os.path.join(project_root, "modelos")
sys.path.append(modelos_path)

# === Importar función desde dataset_entrenado ===
from dataset_entrenado import get_similar_items

# URL de la API de Rasa
RASA_API_URL = "http://0.0.0.0:5005/webhooks/rest/webhook"

def send_message_to_rasa(message, sender_id="default"):
    """Función para enviar un mensaje a la API de Rasa y obtener la respuesta"""
    payload = {
        "sender": sender_id,
        "message": message
    }
    try:
        response = requests.post(RASA_API_URL, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error in request to Rasa: {e}")
        return [{"text": "Sorry, there seems to be a problem connecting to the assistant. Please try again later."}]

def load_html_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def recommend_similar_items(uploaded_file):
    similar_items, style_label = get_similar_items(uploaded_file)
    
    style_dict = {0: "Casual", 1: "Formal", 2: "Sportive", 3: "Elegant", 4: "Urban"}
    style_name = style_dict.get(style_label, "Unknown")

    st.write(f"🧠 Predicted style: {style_name}")
    st.write("🔎 Looking for similar clothes...")

    for _, item in similar_items.iterrows():
        st.image(item['ruta'], caption=f"Recommended: {item['clase']}", use_container_width=True)

def main():
    st.set_page_config(
        page_title="Fashion Assistant 👗",
        page_icon="👗",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"user_{random.randint(10000, 99999)}"

    try:
        html_code = load_html_file(os.path.join(current_dir, "forma.html"))
        components.html(html_code, height=600, scrolling=True)

        st.write("---")
        st.subheader("Conversation with Fashion Assistant")

        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"**You**: {message['content']}")
            else:
                st.markdown(f"**Fashion Assistant**: {message['content']}")

        user_message = st.text_input("What do you need today?")

        if st.button("Enviar") and user_message:
            st.session_state.messages.append({"role": "user", "content": user_message})
            rasa_responses = send_message_to_rasa(user_message, st.session_state.session_id)
            for response in rasa_responses:
                bot_message = response.get('text', "I didn't understand that. Could you rephrase that?")
                st.session_state.messages.append({"role": "assistant", "content": bot_message})
            st.experimental_rerun()

        st.write("Or upload an image of a garment to get recommendations:")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded image", use_container_width=True)
            recommend_similar_items(uploaded_file)
            
    except Exception as e:
        st.error(f"Error al cargar el archivo HTML: {e}")
        st.info("Asegúrate de que el archivo 'forma.html' esté en el mismo directorio que este script.")

if __name__ == "__main__":
    main()
