######## CREACIÓN DE LA INTERFAZ GRÁFICA (API RASA) ##############
# Autor: Carlota Fernández del Riego

import { useState } from "react";
import { Send } from "lucide-react";

export default function ChatbotUI() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const sendMessage = async () => {
    if (!input.trim()) return;
    
    const userMessage = { sender: "user", text: input };
    setMessages([...messages, userMessage]);
    setInput("");

    try {
      const response = await fetch("http://localhost:5005/webhooks/rest/webhook", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sender: "user", message: input })
      });
      
      const data = await response.json();
      const botMessages = data.map((msg) => ({ sender: "bot", text: msg.text }));
      setMessages([...messages, userMessage, ...botMessages]);
    } catch (error) {
      console.error("Error fetching response from Rasa:", error);
      setMessages([...messages, userMessage, { sender: "bot", text: "Error: Unable to connect to chatbot." }]);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-gray-100">
      <div className="w-full max-w-md bg-white shadow-lg rounded-lg p-4">
        <div className="h-80 overflow-y-auto border-b mb-4 p-2">
          {messages.map((msg, index) => (
            <div key={index} className={`p-2 ${msg.sender === "user" ? "text-right" : "text-left"}`}>
              <span className={`inline-block px-3 py-1 rounded-lg ${msg.sender === "user" ? "bg-blue-500 text-white" : "bg-gray-300 text-black"}`}>
                {msg.text}
              </span>
            </div>
          ))}
        </div>
        <div className="flex">
          <input
            type="text"
            className="flex-1 border rounded-l-lg p-2 focus:outline-none"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && sendMessage()}
          />
          <button
            className="bg-blue-500 text-white px-4 py-2 rounded-r-lg flex items-center justify-center"
            onClick={sendMessage}
          >
            <Send size={20} />
          </button>
        </div>
      </div>
    </div>
  );
}

