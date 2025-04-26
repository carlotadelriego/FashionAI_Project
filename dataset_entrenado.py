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
from PIL import Image
import random
import tempfile

# Importar funciones necesarias
from bfs_recommendation import construir_grafo_similitud, bfs_recomendaciones

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

# OJO aquí: ahora columna se llama "class"
df = pd.DataFrame(data, columns=["ruta", "class"])
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
        processed_data.append([img, row["class"]])

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

# Cargar las características previamente extraídas
X_features_path = os.path.join(modelos_dir, 'X_features.npy')
X_features = np.load(X_features_path)
print("✅ Features loaded successfully.")

# Cargar el modelo de clasificación de estilo (ResNet50)
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
    if not uploaded_file:
        return pd.DataFrame(), None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        img = cv2.imread(temp_path)
        if img is None:
            raise ValueError("The image could not be processed.")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)) / 255.0
        input_img = np.expand_dims(img, axis=0)

        extractor = models.Model(inputs=fashion_model.input, outputs=fashion_model.layers[-2].output)
        features = extractor.predict(input_img)

        similarities = cosine_similarity(features, X_features)
        indices = np.argsort(similarities[0])[::-1]
        indices = [idx for idx in indices if idx < len(df)]
        indices = indices[:5]

        style_pred = style_model.predict(input_img)
        style_label = np.argmax(style_pred)

        os.remove(temp_path)

        if indices:
            return df.iloc[indices], style_label
        else:
            return pd.DataFrame(), None

    except Exception as e:
        raise Exception(f"The image could not be processed: {e}")
        return pd.DataFrame(), None
