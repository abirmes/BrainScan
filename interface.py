import streamlit as st 
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# CONFIGURATION DE LA PAGE
# ---------------------------
st.set_page_config(page_title="🧠 Brain Tumor Detector", page_icon="🧠", layout="centered")

# ---------------------------
# CHARGEMENT DU MODÈLE
# ---------------------------
@st.cache_resource
def load_model_cached():
    model = load_model('last_model.h5')
    return model

model = load_model_cached()

# Classes cibles (dans le même ordre que ton entraînement)
class_names = ['notumor', 'glioma', 'meningioma', 'pituitary']

# ---------------------------
# FONCTION DE PRÉDICTION
# ---------------------------
def predict_image(img_path, model, class_names):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0  # Normalisation
        img_array = np.expand_dims(img_array, axis=0)  # Ajout de la dimension batch

        predictions = model.predict(img_array)[0]  # Tableau de probabilités
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = predictions[predicted_index]

        # Créer un DataFrame pour affichage clair
        results_df = pd.DataFrame({
            'Classe': class_names,
            'Confiance (%)': [round(p * 100, 2) for p in predictions]
        }).sort_values(by='Confiance (%)', ascending=False)

        return predicted_class, confidence, results_df
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        return None, None, None

# ---------------------------
# INTERFACE STREAMLIT
# ---------------------------
st.title("🧠 Brain Tumor Detector")
st.write("Importe une image IRM cérébrale pour détecter le type de tumeur.")

uploaded_file = st.file_uploader("📤 Choisir une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Affichage de l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Image chargée", width=300)

    # Sauvegarde temporaire
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Prédiction
    with st.spinner("🔍 Analyse de l'image..."):
        predicted_class, confidence, results_df = predict_image(temp_path, model, class_names)

    if predicted_class:
        st.success(f"✅ **Prédiction principale :** {predicted_class}")
        st.info(f"📊 **Confiance :** {confidence * 100:.2f}%")

        # Afficher toutes les classes avec leur confiance
        st.subheader("Résultats détaillés")
        st.dataframe(results_df, use_container_width=True)

        # Graphique en barres
        fig, ax = plt.subplots()
        ax.barh(results_df['Classe'], results_df['Confiance (%)'], color='skyblue')
        ax.set_xlabel("Confiance (%)")
        ax.set_ylabel("Classe")
        ax.set_title("Distribution des prédictions")
        ax.invert_yaxis()
        st.pyplot(fig)
