import streamlit as st
import streamlit.components.v1 as components
import requests
import random
from PIL import Image
import sys

# Añadir la ruta del archivo bfs_recommendation.py
sys.path.append('/Users/carlotafernandez/Desktop/Code/FashionAI_Project')

# Ahora puedes importar el módulo sin problemas
from bfs_recommendation import construir_grafo_similitud, mostrar_grafo_streamlit
from dataset_entrenado import get_similar_items  # Asegúrate de que esta función esté correctamente importada

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
        response.raise_for_status()  # Si la respuesta es 4xx o 5xx, se lanza una excepción
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error en la solicitud a Rasa: {e}")
        return [{"text": "Lo siento, parece que hay un problema de conexión con el asistente. Por favor, inténtalo de nuevo más tarde."}]

# Leer el contenido del archivo HTML
def load_html_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Función para manejar la recomendación de prendas basadas en la imagen
def recommend_similar_items(uploaded_file):
    similar_items, style_label = get_similar_items(uploaded_file)
    
    # Diccionario de estilos
    style_dict = {0: "Casual", 1: "Formal", 2: "Sportive", 3: "Elegant", 4: "Urban"}
    style_name = style_dict.get(style_label, "Unknown")

    st.write(f"🧠 Estilo predicho: {style_name}")

    # Mostrar las prendas recomendadas
    st.write("🔎 Buscando prendas similares...")
    for _, item in similar_items.iterrows():
        st.image(item['ruta'], caption=f"Recomendado: {item['clase']}", use_container_width=True)

# Función principal para manejar la interfaz de usuario
def main():
    # Configurar la página de Streamlit
    st.set_page_config(
        page_title="Fashion Assistant 👗",
        page_icon="👗",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Inicializar estado de sesión para el chat
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"user_{random.randint(10000, 99999)}"

    # Cargar y renderizar el HTML
    try:
        html_code = load_html_file("forma.html")
        
        # Incluir el HTML en la página
        components.html(html_code, height=600, scrolling=True)
        
        # Área de chat bajo el componente HTML
        st.write("---")
        st.subheader("Conversación con Fashion Assistant")
        
        # Mostrar mensajes previos
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"**Tú**: {message['content']}")
            else:
                st.markdown(f"**Fashion Assistant**: {message['content']}")
        
        # Campo para ingresar el mensaje del usuario
        user_message = st.text_input("¿Qué necesitas hoy?")
        
        if st.button("Enviar") and user_message:
            # Agregar mensaje del usuario al historial
            st.session_state.messages.append({"role": "user", "content": user_message})
            
            # Enviar el mensaje a Rasa y obtener la respuesta
            rasa_responses = send_message_to_rasa(user_message, st.session_state.session_id)
            
            for response in rasa_responses:
                bot_message = response.get('text', "No entendí eso. ¿Podrías reformularlo?")
                # Agregar mensaje del bot al historial
                st.session_state.messages.append({"role": "assistant", "content": bot_message})
            
            # Recargar la página para mostrar los nuevos mensajes
            st.experimental_rerun()

        # **Nuevo**: Subir una imagen para obtener recomendaciones de prendas
        st.write("O bien, sube una imagen de una prenda para obtener recomendaciones:")

        uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            # Mostrar la imagen cargada
            img = Image.open(uploaded_file)
            st.image(img, caption="Imagen subida", use_container_width=True)
            
            # Obtener y mostrar recomendaciones basadas en la imagen
            recommend_similar_items(uploaded_file)
            
    except Exception as e:
        st.error(f"Error al cargar el archivo HTML: {e}")
        st.info("Asegúrate de que el archivo 'forma.html' esté en el mismo directorio que este script.")

if __name__ == "__main__":
    main()
