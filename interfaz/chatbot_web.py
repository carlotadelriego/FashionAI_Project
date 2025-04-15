from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

RASA_API_URL = "http://localhost:5005/webhooks/rest/webhook"

def send_message_to_rasa(message):
    """Función para enviar un mensaje a la API de Rasa y obtener la respuesta"""
    payload = {"message": message}
    try:
        response = requests.post(RASA_API_URL, json=payload)
        response.raise_for_status()  # Si la respuesta es 4xx o 5xx, se lanza una excepción
        print(f"Respuesta de Rasa: {response.json()}")  # Imprimir la respuesta de Rasa para diagnosticar
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error en la solicitud a Rasa: {e}")
        return {"error": "Hubo un problema al conectarse con el servidor de Rasa."}

@app.route("/")
def home():
    return render_template("index.html")  # Página principal

@app.route("/send_message", methods=["POST"])
def send_message():
    """Ruta para recibir el mensaje del usuario y obtener la respuesta de Rasa"""
    user_message = request.form["message"]
    # Llamar a la función para enviar el mensaje a Rasa
    bot_response = send_message_to_rasa(user_message)
    
    print("Response from Rasa:", bot_response)  # Imprimir la respuesta de Rasa en consola

    # Verificar si la respuesta es válida y contiene la clave "text"
    if "error" in bot_response:
        return jsonify({"response": bot_response["error"]})

    if bot_response and isinstance(bot_response, list) and len(bot_response) > 0:
        text = bot_response[0].get("text", "No response")
        return jsonify({"response": text})
    else:
        return jsonify({"response": "No response received from the bot."})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
