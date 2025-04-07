import os
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import streamlit as st
from PIL import Image
import random
import tempfile
from bfs_recommendation import cargar_grafo, bfs_recomendaciones  # 🆕 NUEVO

# Ruta del dataset
base_dir = '/Users/carlotafernandez/Desktop/Code/FashionAI_Project/data/zara_dataset'

data = []
for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    if os.path.isdir(class_path):
        for filename in os.listdir(class_path):
            file_path = os.path.join(class_path, filename)
            if not filename.startswith("."):
                data.append([file_path, class_name])

df = pd.DataFrame(data, columns=["ruta", "clase"])
df.to_csv("dataset.csv", index=False)
print("✅ CSV dataset created successfully.")

# Preprocesar imágenes
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img

processed_data = []
for _, row in df.iterrows():
    img = preprocess_image(row["ruta"])
    if img is not None:
        processed_data.append([img, row["clase"]])

print("✅ Images processed correctly.")

# Convertir los datos a numpy arrays
X = np.array([x[0] for x in processed_data], dtype=np.float32)  
y = np.array([x[1] for x in processed_data])

# Codificar etiquetas
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y)  

# Cargar el label_encoder previamente guardado
label_encoder_classes = np.load("label_encoder_classes.npy", allow_pickle=True)

# Cargar el modelo preentrenado de VGG16
fashion_model = models.load_model("fashion_model.h5")
print("✅ Fashion model loaded successfully.")

# Cargar las características previamente extraídas
X_features = np.load("X_features.npy")
print("✅ Features loaded successfully.")

# Cargar el modelo de clasificación de estilo (ResNet50)
style_model = models.load_model("style_model.h5")
print("✅ Style model loaded successfully.")

# 🆕 Cargar el grafo de similitud
grafo_sim = cargar_grafo()

# Sistema de recomendación con SVD
user_ratings = pd.DataFrame({
    "user_id": [random.randint(1, 100) for _ in range(100)],
    "item_id": [random.randint(1, len(df)) for _ in range(100)],
    "rating": [random.randint(1, 5) for _ in range(100)]
})

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(user_ratings[["user_id", "item_id", "rating"]], reader)
svd_model = SVD()
cross_validate(svd_model, data, cv=5)
print("✅ Trained recommendation model.")

# 🆕 Función para encontrar elementos similares usando BFS
def get_similar_items(uploaded_file):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        input_img = preprocess_image(temp_path)
        input_img = np.expand_dims(input_img, axis=0)

        # Extraer características usando el modelo base
        feature_extractor = models.Model(inputs=fashion_model.input, outputs=fashion_model.layers[-2].output)
        features = feature_extractor.predict(input_img)
        features_flattened = features.flatten().reshape(1, -1)

        # 🧠 Calcular nodo más similar y ejecutar BFS
        similarities = cosine_similarity(features_flattened, X_features)
        indice_inicio = np.argmax(similarities[0])
        bfs_indices = bfs_recomendaciones(grafo_sim, indice_inicio, profundidad_max=2)

        # Clasificación de estilo
        style_prediction = style_model.predict(input_img)
        style_label = np.argmax(style_prediction)

        os.remove(temp_path)
        return df.iloc[bfs_indices], style_label
    return pd.DataFrame(), None

# Interfaz con Streamlit
st.title("Fashion Recommendation System")
uploaded_file = st.file_uploader("Upload an image of a garment", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded image", use_container_width=True)

    st.write("Looking for similar clothes...")
    similar_items, style_label = get_similar_items(uploaded_file)

    style_dict = {0: "Casual", 1: "Formal", 2: "Sportive", 3: "Elegant", 4: "Urban"}
    style_name = style_dict.get(style_label, "Unknown")
    st.write(f"Predicted style: {style_name}")
    
    for _, item in similar_items.iterrows():
        st.image(item['ruta'], caption=f"Recommended: {item['clase']}", use_container_width=True)
