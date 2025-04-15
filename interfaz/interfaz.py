import streamlit as st
import streamlit.components.v1 as components
import requests
import os
import random

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
        print(f"Error in request to Rasa: {e}")
        return [{"text": "Sorry, there seems to be a problem connecting to the assistant. Please try again later."}]

# Leer el contenido del archivo HTML
def load_html_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Función para manejar el chat
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
        st.subheader("Conversation with Fashion Assistant")
        
        # Mostrar mensajes previos
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"**You**: {message['content']}")
            else:
                st.markdown(f"**Fashion Assistant**: {message['content']}")
        
        # Campo para ingresar el mensaje del usuario
        user_message = st.text_input("What do you need today?")
        
        if st.button("Enviar") and user_message:
            # Agregar mensaje del usuario al historial
            st.session_state.messages.append({"role": "user", "content": user_message})
            
            # Enviar el mensaje a Rasa y obtener la respuesta
            rasa_responses = send_message_to_rasa(user_message, st.session_state.session_id)
            
            for response in rasa_responses:
                bot_message = response.get('text', "I didn't understand that. Could you rephrase that?")
                # Agregar mensaje del bot al historial
                st.session_state.messages.append({"role": "assistant", "content": bot_message})
            
            # Recargar la página para mostrar los nuevos mensajes
            st.experimental_rerun()
            
    except Exception as e:
        st.error(f"Error al cargar el archivo HTML: {e}")
        st.info("Asegúrate de que el archivo 'forma.html' esté en el mismo directorio que este script.")

if __name__ == "__main__":
    main()