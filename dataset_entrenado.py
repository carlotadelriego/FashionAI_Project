import os
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import streamlit as st
from PIL import Image
import random
import tempfile

# Importar funciones del grafo
from bfs_recommendation import construir_grafo_similitud, mostrar_grafo_streamlit, bfs_recomendaciones

# Ruta del dataset y modelos
base_dir = '/Users/carlotafernandez/Desktop/Code/FashionAI_Project/data/zara_dataset'
modelos_dir = '/Users/carlotafernandez/Desktop/Code/FashionAI_Project/modelos'  # Ruta a la carpeta 'modelos'

# Cargar datos
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

# Convertir los datos a numpy arrays
X = np.array([x[0] for x in processed_data], dtype=np.float32)  
y = np.array([x[1] for x in processed_data])

# Codificar etiquetas
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y)  

# Cargar el modelo preentrenado de VGG16 desde la carpeta 'modelos'
fashion_model_path = os.path.join(modelos_dir, 'fashion_model.h5')
fashion_model = tf.keras.models.load_model(fashion_model_path)
print("✅ Fashion model loaded successfully.")

# Cargar las características previamente extraídas desde la carpeta 'modelos'
X_features_path = os.path.join(modelos_dir, 'X_features.npy')
X_features = np.load(X_features_path)
print("✅ Features loaded successfully.")

# Cargar el modelo de clasificación de estilo (ResNet50) desde la carpeta 'modelos'
style_model_path = os.path.join(modelos_dir, 'style_model.h5')
style_model = tf.keras.models.load_model(style_model_path)
print("✅ Style model loaded successfully.")

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

# Función para encontrar elementos similares
def get_similar_items(uploaded_file):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        input_img = preprocess_image(temp_path)
        input_img = np.expand_dims(input_img, axis=0)

        # Extraer características de la imagen cargada usando el modelo de características
        feature_extractor = models.Model(inputs=fashion_model.input, outputs=fashion_model.layers[-2].output)
        features = feature_extractor.predict(input_img)

        # Aplanar las características y calcular similitudes
        features_flattened = features.flatten().reshape(1, -1)

        similarities = cosine_similarity(features_flattened, X_features)
        similar_indices = np.argsort(similarities[0])[::-1][:5]

        # Clasificación de estilo
        style_prediction = style_model.predict(input_img)
        style_label = np.argmax(style_prediction)

        os.remove(temp_path)
        return df.iloc[similar_indices], style_label
    return pd.DataFrame(), None

# Interfaz con Streamlit
st.title("Fashion Recommendation System")
uploaded_file = st.file_uploader("Upload an image of a garment", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded image", use_container_width=True)

    st.write("🔎 Looking for similar clothes...")
    similar_items, style_label = get_similar_items(uploaded_file)

    style_dict = {0: "Casual", 1: "Formal", 2: "Sportive", 3: "Elegant", 4: "Urban"}
    style_name = style_dict.get(style_label, "Unknown")
    st.write(f"🧠 Predicted style: {style_name}")
    
    for _, item in similar_items.iterrows():
        st.image(item['ruta'], caption=f"Recommended: {item['clase']}", use_container_width=True)

# Visualización del grafo
st.subheader("📊 Visualization of Similarity Graph")

# Mostrar grafo de similitud
if st.checkbox("Show similarity graph"):
    G = construir_grafo_similitud(df, X_features, top_k=5)
    mostrar_grafo_streamlit(G, df)

# Recomendación mediante BFS
st.subheader("🧠 BFS Recommendations")

if st.checkbox("Get BFS recommendations"):
    # Selecciona un nodo aleatorio del grafo para empezar el BFS
    start_node = random.choice(list(df.index))
    recommendations = bfs_recomendaciones(G, start_node, top_k=5)
    
    st.write("🧠 BFS recommendations based on selected item:")
    for node in recommendations:
        item = df.iloc[node]
        st.image(item['ruta'], caption=f"Recommended: {item['clase']}", use_container_width=True)
