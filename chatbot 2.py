import os
import spacy
import requests
from bs4 import BeautifulSoup
from rasa.core.agent import Agent
from transformers import pipeline, BertTokenizer, BertModel

# Cargar modelos
nlp = spacy.load("en_core_web_sm")

# Cargar modelo en inglés
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

text_generator = pipeline("text-generation", model="gpt2", framework="pt")

# Cargar el modelo de Rasa
agent = Agent.load("models")

def get_message_rasa(message):
    """Procesa el mensaje con Rasa y devuelve una respuesta."""
    response = agent.handle_text(message)
    return response[0]['text']

def web_scraping(urls):
    """Extrae el texto de las URLs dadas."""
    texts = [
    'https://www.vogue.com/article/9-outdated-fashion-pieces-vogue-editors-still-love',
    'https://www.vogue.com/article/metropolitan-museum-lorna-simpson',
    'https://www.elle.com/culture/music/a63324386/lady-gaga-mayhem-interview-2025/'
    ]

    for link in urls:
        response = requests.get(link)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        texts.append(text)
    return texts

def analyze_trends(texts):
    """Analiza tendencias en los textos con BERT."""
    trend_info = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        relevant_words = [tokenizer.decode([torch.argmax(embeddings[0]).item()])]
        trend_info.append(f"Tendencias encontradas: {', '.join(relevant_words)}")
    return trend_info
