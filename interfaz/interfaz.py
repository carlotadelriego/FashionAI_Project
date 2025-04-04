# chatbot_web.py
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

RASA_API_URL = "http://localhost:5005/webhooks/rest/webhook"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/send_message", methods=["POST"])
def send_message():
    user_message = request.form["message"]
    response = requests.post(
        RASA_API_URL,
        json={"sender": "user", "message": user_message}
    )
    bot_response = response.json()
    if bot_response:
        return jsonify({"response": bot_response[0].get("text", "No response")})
    return jsonify({"response": "No response received."})

if __name__ == "__main__":
    app.run(debug=True)
