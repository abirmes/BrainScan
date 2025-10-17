# BrainScan

## 📌 Objectif
Ce projet vise à **classifier les images IRM du cerveau** en quatre catégories à l’aide d’un **réseau de neurones convolutif (CNN)** :
- **notumor**
- **glioma**
- **meningioma**
- **pituitary**

Le modèle est entraîné sur un dataset d’images, puis évalué pour mesurer sa capacité à distinguer les différents types de tumeurs.

---

## ⚙️ Étapes du projet

### 1️⃣ Prétraitement
- Chargement des images depuis le dossier `Data/`
- Redimensionnement à **224×224 pixels**
- Encodage des labels (0 à 3)
- Visualisation du nombre d’images par classe

### 2️⃣ Data Augmentation
- Utilisation de `ImageDataGenerator` pour équilibrer les classes :
  - rotation, zoom, translation, cisaillement, flip horizontal
- Chaque classe atteint environ **2000 images** après augmentation

### 3️⃣ Construction du modèle CNN
- 3 couches **Convolution + MaxPooling**
- 2 couches **Dropout**
- 1 couche **Dense (128 neurones)**
- 1 couche de sortie **Softmax (4 classes)**
- Optimiseur : **Adam (lr = 0.01)**
- Fonction de perte : **categorical_crossentropy**

### 4️⃣ Entraînement
- Division du dataset : 80% entraînement / 20% test
- **EarlyStopping** et **ModelCheckpoint** pour éviter le surapprentissage
- Entraînement sur 35 époques, batch size = 64

### 5️⃣ Évaluation et Résultats
- Courbes d’évolution de la **précision** et de la **perte**
- **Matrices de confusion** par classe
- **Classification report** (Précision, Recall, F1-score)
- Exemples de **prédictions correctes** et **incorrectes**

---

## 📊 Rapport Streamlit

Une application Streamlit simple permet de visualiser le rapport complet du projet.

### ▶️ Lancer l’application

```bash
streamlit run rapport/rapport.py
