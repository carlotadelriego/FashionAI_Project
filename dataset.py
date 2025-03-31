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

# Guardar etiquetas para usarlas luego
np.save("label_encoder_classes.npy", label_encoder.classes_)

# Modelo CNN con VGG16
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(128, activation="relu")(x)
output_layer = layers.Dense(len(label_encoder.classes_), activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Entrenar y guardar el modelo
model.fit(X, y, epochs=7, batch_size=32)
model.save("fashion_model.h5")
print("✅ Modelo CNN con VGG16 entrenado y guardado.")

# Extraer características para clustering
feature_extractor = models.Model(inputs=base_model.input, outputs=x)
X_features = feature_extractor.predict(X)
np.save("X_features.npy", X_features)  # Guardar características
print("✅ Características extraídas y guardadas.")

# Aplicar K-Means y guardar clusters
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_features)
df["cluster"] = labels
df.to_csv("clustered_dataset.csv", index=False)
print("✅ Clustering completed y guardado.")

# Modelo de clasificación de estilos con ResNet50
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet_model.trainable = False  

x = layers.GlobalAveragePooling2D()(resnet_model.output)
x = layers.Dense(128, activation='relu')(x)
output_layer_style = layers.Dense(5, activation='softmax')(x)  

style_model = models.Model(inputs=resnet_model.input, outputs=output_layer_style)
style_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

style_model.save("style_model.h5")
print("✅ Modelo de clasificación de estilos guardado.")

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

        # Extraer características de la imagen cargada
        features = feature_extractor.predict(input_img)

        # Asegurarse de que las características tienen la misma dimensión
        features_flattened = features.flatten().reshape(1, -1)  # Aplanar las características

        # Calcular similitudes
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

    st.write("Looking for similar clothes...")
    similar_items, style_label = get_similar_items(uploaded_file)

    style_dict = {0: "Casual", 1: "Formal", 2: "Deportivo", 3: "Elegante", 4: "Urbano"}
    style_name = style_dict.get(style_label, "Desconocido")
    st.write(f"Predicted style: {style_name}")
    
    for _, item in similar_items.iterrows():
        st.image(item['ruta'], caption=f"Recommended: {item['clase']}", use_container_width=True)
