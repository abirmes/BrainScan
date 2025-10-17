import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# =============================
# TITRE ET INTRODUCTION
# =============================
st.set_page_config(page_title="Rapport CNN - Classification de Tumeurs Cérébrales", layout="wide")
st.title("🧠 Rapport explicatif - Classification de tumeurs cérébrales avec CNN")
st.write("""
Ce rapport présente les étapes principales du projet de **détection de tumeurs cérébrales** à l'aide d'un **réseau de neurones convolutif (CNN)**.
L'objectif était de classifier les images IRM en quatre catégories :
- **Notumor**
- **Glioma**
- **Meningioma**
- **Pituitary**
""")

# =============================
# 1. PRÉTRAITEMENT DES DONNÉES
# =============================
st.header("1️⃣ Prétraitement des données")
st.write("""
- Les images ont été chargées depuis les 4 dossiers (`Data/notumor`, `Data/glioma`, `Data/meningioma`, `Data/pituitary`).
- Chaque image a été redimensionnée à **224×224 pixels**.
- Les étiquettes numériques ont été associées aux classes correspondantes.
""")

try:
    img_nb = Image.open("rapport/nombre_images_par_classe.png")
    st.image(img_nb, caption="Distribution initiale des images par classe", width=600)
except FileNotFoundError:
    st.warning("Image non trouvée : nombre_images_par_classe.png")

st.write("""
Ensuite, une **visualisation d'un échantillon d'images** a permis de vérifier la qualité et la variété du dataset.
""")

try:
    img_ex = Image.open("rapport/echantillon_classes.png")
    st.image(img_ex, caption="Échantillon d'images par classe", width=800)
except FileNotFoundError:
    st.warning("Image non trouvée : echantillon_classes.png")

# =============================
# 2. AUGMENTATION DES DONNÉES
# =============================
st.header("2️⃣ Data Augmentation")
st.write("""
Pour éviter le déséquilibre entre les classes, une **augmentation de données** a été appliquée :
- Rotation, zoom, décalage, cisaillement, flip horizontal.
- Objectif : atteindre **2000 images par classe**.
""")

try:
    img_aug = Image.open("rapport/nombre_images_apres_augmentation.png")
    st.image(img_aug, caption="Distribution après augmentation", width=600)
except FileNotFoundError:
    st.warning("Image non trouvée : nombre_images_apres_augmentation.png")

# =============================
# 3. CONSTRUCTION DU MODÈLE CNN
# =============================
st.header("3️⃣ Construction du modèle CNN")
st.write("""
Le modèle contient :
- **3 couches Convolution + MaxPooling**
- **2 Dropout** pour éviter l'overfitting
- **1 couche Dense cachée (128 neurones)** avec activation ReLU
- **1 couche de sortie Softmax (4 classes)**

Optimiseur : **Adam** (learning_rate = 0.01)  
Fonction de perte : **categorical_crossentropy**
""")

try:
    img_arch = Image.open("rapport/cnn_architecture.png")
    st.image(img_arch, caption="Architecture du CNN", width=700)
except FileNotFoundError:
    st.warning("Image non trouvée : cnn_architecture.png")

# =============================
# 4. ENTRAÎNEMENT DU MODÈLE
# =============================
st.header("4️⃣ Entraînement du modèle")
st.write("""
- **80%** des données utilisées pour l'entraînement, **20%** pour le test.
- Utilisation de **EarlyStopping** et **ModelCheckpoint**.
- Entraînement sur **35 epochs**, batch size = 64.
""")

col1, col2 = st.columns(2)
with col1:
    try:
        st.image("rapport/courbe_accuracy.png", caption="Accuracy au fil des époques", width=400)
    except FileNotFoundError:
        st.warning("Image non trouvée : courbe_accuracy.png")

with col2:
    try:
        st.image("rapport/courbe_loss.png", caption="Loss au fil des époques", width=400)
    except FileNotFoundError:
        st.warning("Image non trouvée : courbe_loss.png")

st.write("""
> Ces courbes montrent la convergence du modèle et la bonne généralisation sur le jeu de validation.
""")

# =============================
# 5. ÉVALUATION ET RÉSULTATS
# =============================
st.header("5️⃣ Évaluation du modèle")

st.write("""
Les performances finales sur le jeu de test :
- **Accuracy** ≈ `0.9206249713897705`
- **Loss** ≈ `0.2861359417438507`

Les **matrices de confusion** par classe permettent d'évaluer les erreurs de classification.
""")

# Affichage des matrices de confusion en 2x2
cols = st.columns(2)
images_matrices = [
    ("rapport/matrice_confusion_glioma.png", "Classe : Glioma"),
    ("rapport/matrice_confusion_meningioma.png", "Classe : Meningioma"),
    ("rapport/matrice_confusion_pituitary.png", "Classe : Pituitary"),
    ("rapport/matrice_confusion_notumor.png", "Classe : Notumor"),
]

for idx, (img_path, caption) in enumerate(images_matrices):
    col = cols[idx % 2]
    try:
        img = Image.open(img_path)
        col.image(img, caption=caption, width=350)
    except FileNotFoundError:
        col.warning(f"Image non trouvée : {img_path}")

# =============================
# 6. EXEMPLES DE PRÉDICTIONS
# =============================
st.header("6️⃣ Exemples de prédictions")

col1, col2 = st.columns(2)
with col1:
    try:
        st.image("rapport/predictions_correctes.png", caption="Exemples de prédictions correctes", width=450)
    except FileNotFoundError:
        st.warning("Image non trouvée : predictions_correctes.png")

with col2:
    try:
        st.image("rapport/predictions_incorrectes.png", caption="Exemples de prédictions incorrectes", width=450)
    except FileNotFoundError:
        st.warning("Image non trouvée : predictions_incorrectes.png")

st.write("""
Ces exemples permettent d'analyser les réussites et les confusions du modèle.
""")

# =============================
# 7. CONCLUSION
# =============================
st.header("✅ Conclusion")
st.write("""
Le modèle CNN a permis d'obtenir de bons résultats pour la classification des tumeurs cérébrales à partir d'IRM.
Grâce à la **data augmentation** et à la **régularisation (Dropout)**, le surapprentissage a été limité.

🔹 **Forces :**
- Bonne généralisation
- Précision correcte
- Structure CNN simple et efficace

🔹 **Axes d'amélioration :**
- Tester un modèle pré-entraîné (Transfer Learning)
- Ajuster le taux d'apprentissage
- Ajouter une meilleure gestion du déséquilibre

---
👩‍💻 Réalisé par : **Abir Meskini**
""")