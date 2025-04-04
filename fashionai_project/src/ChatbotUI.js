import React, { useState } from 'react';
import { Send } from 'lucide-react';

const ChatbotUI = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { sender: 'user', text: input };
    setMessages([...messages, userMessage]);
    setInput('');

    try {
      const response = await fetch('http://localhost:5005/webhooks/rest/webhook', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sender: 'user', message: input }),
      });

      const data = await response.json();
      const botMessages = data.map((msg) => ({ sender: 'bot', text: msg.text }));
      setMessages([...messages, userMessage, ...botMessages]);
    } catch (error) {
      console.error('Error fetching response from Rasa:', error);
      setMessages([...messages, userMessage, { sender: 'bot', text: 'Error: Unable to connect to chatbot.' }]);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-gradient-to-r from-pink-200 to-purple-300"> {/* Fondo degradado */}
      <div className="w-full max-w-lg bg-white shadow-xl rounded-3xl p-6"> {/* Ventana del chat */}
        <div className="h-80 overflow-y-auto border-b mb-4 p-2">
          {messages.map((msg, index) => (
            <div key={index} className={`p-3 ${msg.sender === 'user' ? 'text-right' : 'text-left'}`}>
              <span className={`inline-block px-4 py-2 rounded-lg ${msg.sender === 'user' ? 'bg-pink-500 text-white' : 'bg-gray-100 text-black'}`}>
                {msg.text}
              </span>
            </div>
          ))}
        </div>
        <div className="flex items-center space-x-2">
          <input
            type="text"
            className="flex-1 border rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-pink-400"
            placeholder="Escribe algo... (¿Te gustaría encontrar algo de moda?)"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          />
          <button
            className="bg-pink-500 text-white px-6 py-3 rounded-lg flex items-center justify-center hover:bg-pink-600"
            onClick={sendMessage}
          >
            <Send size={24} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatbotUI;
