from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

RASA_API_URL = "http://localhost:5005/webhooks/rest/webhook"

def send_message_to_rasa(message):
    """Función para enviar un mensaje a la API de Rasa y obtener la respuesta"""
    payload = {"message": message}
    response = requests.post(RASA_API_URL, json=payload)
    return response.json()

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

    if bot_response:
        return jsonify({"response": bot_response[0].get("text", "No response")})
    else:
        return jsonify({"response": "No response received from the bot."})

if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Cambia el puerto de 5000 a 5001 o cualquier otro puerto disponible
