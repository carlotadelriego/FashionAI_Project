import os
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
import streamlit as st
from PIL import Image
import random
import tempfile
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'modelo')))

# Importaciones desde módulos propios
from bfs_recommendation import construir_grafo_similitud, guardar_grafo, cargar_grafo, bfs_recomendaciones

# Definir la ruta donde se guardarán todos los archivos
modelos_dir = '/Users/carlotafernandez/Desktop/Code/FashionAI_Project/modelos'

if not os.path.exists(modelos_dir):
    os.makedirs(modelos_dir)

# Ruta del dataset
base_dir = '/Users/carlotafernandez/Desktop/Code/FashionAI_Project/data/zara_dataset'

data_img = []
for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    if os.path.isdir(class_path):
        for filename in os.listdir(class_path):
            file_path = os.path.join(class_path, filename)
            if not filename.startswith("."):
                data_img.append([file_path, class_name])

df = pd.DataFrame(data_img, columns=["ruta", "clase"])
df.to_csv(os.path.join(modelos_dir, "dataset.csv"), index=False)
print("✅ CSV dataset created successfully.")

# Preprocesar imágenes
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
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

np.save(os.path.join(modelos_dir, "label_encoder_classes.npy"), label_encoder.classes_)
print("✅ Label encoder classes saved.")

# Modelo CNN con VGG16
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(128, activation="relu")(x)
output_layer = layers.Dense(len(label_encoder.classes_), activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X, y, epochs=7, batch_size=32)
model.save(os.path.join(modelos_dir, "fashion_model.h5"))
print("✅ Modelo CNN con VGG16 entrenado y guardado.")

# Extraer características para clustering
feature_extractor = models.Model(inputs=base_model.input, outputs=x)
X_features = feature_extractor.predict(X)
np.save(os.path.join(modelos_dir, "X_features.npy"), X_features)
print("✅ Características extraídas y guardadas.")

# Construir y guardar el grafo de similitud
grafo_path = os.path.join(modelos_dir, "grafo_similitud.pkl")
if not os.path.exists(grafo_path):
    grafo = construir_grafo_similitud(df, X_features, top_k=5)
    guardar_grafo(grafo, grafo_path)
    print("✅ Grafo de similitud creado y guardado.")
else:
    print("Grafo ya existe. Saltando construcción.")

# Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_features)
df["cluster"] = labels
df.to_csv(os.path.join(modelos_dir, "clustered_dataset.csv"), index=False)
print("✅ Clustering completed y guardado.")

# Modelo de clasificación de estilos con ResNet50
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet_model.trainable = False

x_style = layers.GlobalAveragePooling2D()(resnet_model.output)
x_style = layers.Dense(128, activation='relu')(x_style)
output_layer_style = layers.Dense(5, activation='softmax')(x_style)

style_model = models.Model(inputs=resnet_model.input, outputs=output_layer_style)
style_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

style_model.save(os.path.join(modelos_dir, "style_model.h5"))
print("✅ Modelo de clasificación de estilos guardado.")

# Sistema de recomendación con SVD
user_ratings = pd.DataFrame({
    "user_id": [random.randint(1, 100) for _ in range(100)],
    "item_id": [random.randint(1, len(df)) for _ in range(100)],
    "rating": [random.randint(1, 5) for _ in range(100)]
})

reader = Reader(rating_scale=(1, 5))
surprise_data = Dataset.load_from_df(user_ratings[["user_id", "item_id", "rating"]], reader)
svd_model = SVD()
cross_validate(svd_model, surprise_data, cv=5)
print("✅ Trained recommendation model.")

# Cargar grafo
grafo_sim = cargar_grafo(grafo_path)

# Función para encontrar elementos similares
def get_similar_items(uploaded_file):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        input_img = preprocess_image(temp_path)
        if input_img is None:
            return pd.DataFrame(), None

        input_img = np.expand_dims(input_img, axis=0)
        features = feature_extractor.predict(input_img)
        features_flattened = features.flatten().reshape(1, -1)

        similarities = cosine_similarity(features_flattened, X_features)
        indice_inicio = np.argmax(similarities[0])

        bfs_indices = bfs_recomendaciones(grafo_sim, indice_inicio, profundidad_max=2)

        style_prediction = style_model.predict(input_img)
        style_label = np.argmax(style_prediction)

        os.remove(temp_path)
        return df.iloc[bfs_indices], style_label
    return pd.DataFrame(), None
