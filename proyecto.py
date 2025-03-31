###### PROYECTO CHATBOT MODA ######
# Autora: Carlota Fernández del Riego

# Importamos las librerías necesarias
import os
import PyPDF2
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from spacy.lang.es.stop_words import STOP_WORDS
import requests
from bs4 import BeautifulSoup
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer  # Solo necesario para el TF-IDF
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from rasa.core.agent import Agent  # Rasa para el chatbot
from transformers import pipeline, BertTokenizer, BertModel  # Para BERT y GPT
import torch
from torchvision import models, transforms

# Cargar modelos de Transformers
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
text_generator = pipeline("text-generation", model="gpt2", framework="pt")
vision_model = models.resnet50(weights='IMAGENET1K_V1')
vision_model.eval()

# Cargar el modelo entrenado de Rasa
try:
    agent = Agent.load("models")  # asegurarse de que el directorio 'models' contiene el modelo entrenado
except Exception as e:
    print(f"Error loading Rasa model: {e}")
    agent = None  # Manejo de errores en caso de que no se pueda cargar el modelo


# Función para procesar el mensaje y obtener la respuesta de Rasa
def get_message_rasa(message):
    if agent:
        response = agent.handle_text(message)
        return response[0]['text']
    else:
        return "No Rasa model loaded."

# Web scraping
urls = [
    'https://www.vogue.com/article/9-outdated-fashion-pieces-vogue-editors-still-love',
    'https://www.vogue.com/article/metropolitan-museum-lorna-simpson',
    'https://www.elle.com/culture/music/a63324386/lady-gaga-mayhem-interview-2025/'
]

# Función para hacer web scraping y extraer los textos de los enlaces
def web_scraping(urls):
    texts = []
    for link in urls:
        try:
            response = requests.get(link)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            texts.append(text)
        except requests.exceptions.RequestException as e:
            print(f"Error scraping {link}: {e}")
    return texts

scraped_texts = web_scraping(urls)

# Cargar los pdf recolectados (verifica si las rutas son correctas)
files_data = [
    '/Users/carlotafernandez/Desktop/Code/FASHION/Fashion_Portfolio-1/Mode_system.pdf',
    '/Users/carlotafernandez/Desktop/Code/FASHION/Fashion_Portfolio-1/christianDior.pdf'
]


###### PREPROCESAMIENTO DEL TEXTO ######
# Cargar el modelo de spaCy
nlp = spacy.load("en_core_web_sm")

# 1. TOKENIZATION - SPACY
documents = files_data + scraped_texts
tokenized_documents = [nlp(document) for document in documents]

# 2. ELIMINAR STOPWORDS
def remove_stopwords(tokenized_docs):
    filtered_docs = []
    for doc in tokenized_docs:
        filtered_tokens = [token.text for token in doc if token.text.lower() not in STOP_WORDS and token.is_alpha]
        filtered_docs.append(" ".join(filtered_tokens))
    return filtered_docs
filtered_documents = remove_stopwords(tokenized_documents)

# 3. NORMALIZACIÓN (LEMATIZACIÓN)
def normalize_tokens(tokenized_docs):
    normalized_docs = []
    for doc in tokenized_docs:
        normalized_tokens = [token.lemma_.lower() for token in doc if token.lemma_ != '-PRON-' and token.is_alpha]
        normalized_docs.append(" ".join(normalized_tokens))  # Convertir a string
    return normalized_docs
normalized_documents = normalize_tokens(tokenized_documents)

# 4. PROCESAR DOCUMENTOS PARA TF-IDF
processed_documents = filtered_documents + normalized_documents  # Debe ser una lista de strings

# 5. EXTRACCIÓN DE PALABRAS CLAVE CON TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=0.1)
X = vectorizer.fit_transform(processed_documents)
keywords = vectorizer.get_feature_names_out()
tfidf_scores = X.toarray().mean(axis=0)  # Promedio de los scores por documento
df_keywords = pd.DataFrame({'keyword': keywords, 'score': tfidf_scores})
fashion_keywords = {"fashion", "style", "dress", "clothing", "trend", "outfit", "wear", "designer", "runway", "chic", "elegant"}
df_keywords = df_keywords[df_keywords['keyword'].isin(fashion_keywords)]
df_keywords = df_keywords.sort_values(by='score', ascending=False)

# 6. APLICAR EMBEDDINGS PARA ENTENDER EL SIGNIFICADO
def get_similar_fashion_terms(word, n=5):
    """ Encuentra palabras similares a un término de moda usando embeddings de spaCy """
    token = nlp.vocab[word]
    similar_words = sorted(nlp.vocab, key=lambda w: token.similarity(w), reverse=True)
    # Filtrar palabras con sentido y devolver las más cercanas
    return [w.text for w in similar_words if w.is_alpha and w.has_vector][:n]

# 7. APLICAR BERT PARA ANALIZAR TENDENCIAS
####### aplicar huggingface transformers en vez de BERT ####### 
def analyze_trends(texts):
    """ Analiza textos de moda usando BERT para extraer información de tendencias """
    all_trend_info = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Obtener embeddings promedio
        # Aquí iría la lógica para extraer información de tendencias a partir de los embeddings
        relevant_words = []
        for i in range(5):
            word_idx = torch.argmax(embeddings[0]).item()
            word = tokenizer.decode([word_idx])
            relevant_words.append(word)
            embeddings[0][word_idx] = -1  # Evitar seleccionar la misma palabra nuevamente
        
        trend_info = f"Tendencias encontradas: {', '.join(relevant_words)}"
        all_trend_info.append(trend_info)
    return all_trend_info

trend_info = analyze_trends(processed_documents)
print(trend_info) # Imprimir la información de tendencias extraídas

# CREATE A DATAFRAME WITH THE PROCESSED TEXT
df = pd.DataFrame({
    'Original': [' '.join([token.text for token in doc]) for doc in tokenized_documents],
    'Filtered': [' '.join(doc) for doc in filtered_documents],
    'Normalized': [' '.join(doc) for doc in normalized_documents]
})

# SAVE RESULTS ON A CSV
df.to_csv('preprocessed_texts.csv', index=False)



###### GENERAR DESCRIPCIONES AUTOMÁTICAS DE OUTFITS ######
# Función para generar descripciones basadas en imágenes
def generate_description_from_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = vision_model(image)
    description = text_generator("This outfit features", max_length=50, num_return_sequences=1)
    return description[0]['generated_text']


# Función para generar descripciones basadas en preferencias
def generate_description_from_preferences(preferences):
    color = preferences.get('color', 'neutral')
    style = preferences.get('style', 'casual')
    occasion = preferences.get('occasion', 'everyday')
    prompt = f"A {color} {style} outfit suitable for {occasion} occasions. This outfit includes"
    description = text_generator(prompt, max_length=50, num_return_sequences=1)
    return description[0]['generated_text']



###### INTERFAZ GRÁFICA ######
# Función para crear la interfaz gráfica del chatbot
def chatbot_interface():
    # Crear una ventana
    window = tk.Tk()
    window.title("Fashion Assistant Chatbot")
    window.geometry("700x600")  # Ajustar tamaño de la ventana

    # Cargar y mostrar la imagen
    try:
        image = Image.open('/Users/carlotafernandez/Desktop/Code/FashionAI_Project/data/LOGO_UIE_CUADRADO-01.jpg')
        image = image.resize((200, 150), Image.Resampling.LANCZOS)  # Redimensionar la imagen
        img = ImageTk.PhotoImage(image)
    except Exception as e:
        print(f"Error loading image: {e}")
        img = None

    # Crear un widget Label para mostrar la imagen
    image_label = tk.Label(window, image=img)
    image_label.grid(row=0, column=1, padx=10, pady=10)  # Posicionar la imagen al lado

    # Crear un área de texto con scroll para mostrar la conversación
    conversation_area = scrolledtext.ScrolledText(window, width=80, height=25, wrap=tk.WORD, state=tk.DISABLED, font=("Arial", 12))
    conversation_area.grid(row=0, column=0, padx=10, pady=10, columnspan=2)  # Posicionar en una fila ancha

    # Título en la parte superior
    welcome_label = tk.Label(window, text="Welcome to Fashion Assistant Chatbot!", font=("Arial", 14))
    welcome_label.grid(row=1, column=0, columnspan=2, pady=10)

    # Crear una caja de entrada para que el usuario escriba su mensaje
    user_input_box = tk.Entry(window, width=60, font=("Arial", 12))
    user_input_box.grid(row=2, column=0, padx=10, pady=10)

    # Función para manejar el envío de mensajes
    def send_message():
        user_input = user_input_box.get()
        if user_input.strip():  # Evitar mensajes vacíos
            conversation_area.config(state=tk.NORMAL)
            conversation_area.insert(tk.END, f"You: {user_input}\n")  # Mostrar la entrada del usuario
            conversation_area.yview(tk.END)
            
            # Obtener la respuesta del chatbot usando Rasa
            rasa_response = get_message_rasa(user_input)
            
            conversation_area.insert(tk.END, f"Chatbot: {rasa_response}\n")  # Mostrar la respuesta del chatbot
            conversation_area.yview(tk.END)
            
            user_input_box.delete(0, tk.END)  # Limpiar la caja de entrada

    # Crear un botón para enviar el mensaje
    send_button = tk.Button(window, text="Send", width=10, command=send_message, font=("Arial", 12))
    send_button.grid(row=2, column=1, padx=10, pady=10)

    # Ejecutar el bucle principal de la GUI
    window.mainloop()

# Ejecutar la interfaz del chatbot
chatbot_interface()
