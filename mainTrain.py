import os 
from PIL import Image
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D , MaxPooling2D, Activation , Dropout, Flatten , Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


image_directory = 'Data/'
folders = {
    'notumor': 0,
    'glioma': 1,
    'meningioma': 2,
    'pituitary': 3
}
dataset = []
label = []

for folder_name, folder_label in folders.items():
    images = os.listdir(image_directory + folder_name + '/')
    for image_name in images:
        ext = image_name.split('.')[-1].lower()
        if ext in ['jpg', 'jpeg', 'bmp', 'png']:
            try:
                image = cv2.imread(image_directory + folder_name + '/' + image_name) # lit l’image depuis le disque en tableau NumPy
                image = Image.fromarray(image, 'RGB') # convertit ce tableau en objet PIL
                image = image.resize((224, 224)) # Redimensionner les images à une taille fixe 224×224 à l’aide de la bibliothèque OpenCV.
                dataset.append(np.array(image)) # retransforme en tableau NumPy pour ML 
                label.append(folder_label) #Chaque image et son label doivent avoir le même indice dans les deux listes.
            except Exception as e:
                print(f"Erreur lors du traitement de l'image {image_name} dans le dossier {folder_name} : {e}")
#Une fois toutes les images et leurs labels chargés, convertir les listes images et étiquettes en tableaux NumPy pour les rendre exploitables par le modèle CNN.
dataset = np.array(dataset)
label = np.array(label)



#Afficher graphiquement le nombre d’images dans chaque classe.
classes, counts = np.unique(label, return_counts=True) # compter combien d’images pour chaque classe
class_names = list(folders.keys())
plt.figure(figsize=(8,5))
plt.bar(class_names, counts, color='skyblue')
plt.xlabel('Classes')
plt.ylabel('Nombre d’images')
plt.title('Nombre d’images par classe')
plt.show()





# Montrer un échantillon d’images pour chaque classe.
class_names = list(folders.keys()) # Récupérer les noms des classes dans l’ordre de leur label
plt.figure(figsize=(10, 6)) # Définir la taille du graphique (1 ligne, 4 colonnes)
for i, class_name in enumerate(class_names): # Afficher 1 image par classe
    class_indices = np.where(label == i)[0] # Trouver les indices des images appartenant à cette classe
    image = dataset[class_indices[0]] # Récupérer l'image correspondante
    # Créer un sous-graphique
    plt.subplot(1, len(class_names), i + 1)
    plt.imshow(image)
    plt.title(class_name)
    plt.axis('off')
plt.tight_layout()
plt.show()




#Vérifier l’équilibre entre les classes et appliquer un rééquilibrage si nécessaire.
# classes, counts = np.unique(label, return_counts=True)
inverse_folders = {v: k for k, v in folders.items()}
# for c, count in zip(classes, counts):
#     print(f"Classe {inverse_folders[c]}: {count} images")

# Classe notumor: 2000 images
# Classe glioma: 1621 images
# Classe meningioma: 1645 images
# Classe pituitary: 1757 images


# utilisation de data augmentation
train_datagen = ImageDataGenerator( # Crée un générateur d’images avec transformations aléatoires
    rotation_range=20, # Rotation aléatoire des images jusqu’à 20 degrés
    width_shift_range=0.2,# Décalage horizontal aléatoire jusqu’à 20% de la largeur
    height_shift_range=0.2,# Décalage vertical aléatoire jusqu’à 20% de la hauteur
    shear_range=0.2,# Application d’un cisaillement aléatoire
    zoom_range=0.2,# Zoom avant/arrière aléatoire jusqu’à 20%
    horizontal_flip=True,# Inversion horizontale aléatoire (miroir)
    fill_mode='nearest'# Remplit les pixels manquants avec les valeurs les plus proches
)

# Calculer le nombre d'images à générer pour chaque classe
max_count = 2000  # Nombre cible d’images par classe après augmentation

augmented_dataset = list(dataset) # Copie initiale du dataset original
augmented_labels = list(label) # Copie initiale des labels

# Pour chaque classe, générer des images augmentées
for class_label in classes:# Boucle sur chaque classe
    class_indices = np.where(label == class_label)[0]# Indices des images appartenant à cette classe
    class_images = dataset[class_indices]# Images de cette classe
    current_count = len(class_images)# Nombre actuel d’images
    images_to_generate = max_count - current_count# Nombre d’images manquantes pour atteindre max_count
    if images_to_generate > 0: # Si la classe a besoin d’augmentation
        generated = 0
        while generated < images_to_generate:# Tant qu’on n’a pas généré assez d’images
            random_image = class_images[np.random.randint(0, len(class_images))]# Choisir une image au hasard
            random_image = np.expand_dims(random_image, axis=0)# Ajouter une dimension batch
            augmented_image = train_datagen.random_transform(random_image[0])# Appliquer transformation aléatoire
            augmented_dataset.append(augmented_image)# Ajouter image augmentée au dataset
            augmented_labels.append(class_label)# Ajouter le label correspondant
            generated += 1
# Convertir en arrays NumPy
augmented_dataset = np.array(augmented_dataset)# Conversion en tableau NumPy
augmented_labels = np.array(augmented_labels) # Conversion des labels en tableau NumPy

classes, counts = np.unique(augmented_labels, return_counts=True)# Compte du nombre d’images par classe
class_names = list(folders.keys()) # Noms des classes à afficher
plt.figure(figsize=(8,5))# Taille du graphique
plt.bar(class_names, counts, color='skyblue')# Diagramme en barres
plt.xlabel('Classes')# Légende axe X
plt.ylabel('Nombre d’images')# Légende axe Y
plt.title('Nombre d’images par classe')# Titre
plt.show()# Affichage du graphique

x = augmented_dataset/255.0  # Normalisation des pixels (valeurs entre 0 et 1)
y = augmented_labels # Copie des labels

y = to_categorical(y, num_classes=4)  # Conversion des labels en one-hot encoding (4 classes)

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=1)  # Division en train/test

# model building
model = Sequential()  # Initialisation du modèle séquentiel

# couche 1
model.add(Conv2D(32 , (3 ,3) , input_shape = (224 , 224 ,3))) # Détection de motifs simples (bords, textures)
model.add(MaxPooling2D(pool_size=(2,2))) # Réduction de la taille de l’image (garde l’essentiel)

# couche 2
model.add(Conv2D(64 , (3 ,3) , activation='relu'))# Plus de filtres → caractéristiques plus complexes
model.add(MaxPooling2D(pool_size=(2,2)))# Pooling pour réduire la dimension spatiale

# couche 3
model.add(Conv2D(128 , (3 ,3) , activation='relu'))# Détection de motifs encore plus complexes
model.add(MaxPooling2D(pool_size=(2,2))) # Réduction dimensionnelle

model.add(Dropout(0.5))# Désactive 50% des neurones pour éviter le surapprentissage

model.add(Flatten()) # Passage de 2D → 1D
model.add(Dense(128, activation='relu')) # Couche cachée entièrement connectée
model.add(Dropout(0.5)) # Dropout supplémentaire pour régulariser

model.add(Dense(4, activation='softmax'))# Couche de sortie avec 4 classes (probabilités)

optimizer = Adam(learning_rate=0.01) # Optimiseur Adam avec un taux d’apprentissage fixé
model.compile(
    optimizer=optimizer,# Algorithme d’optimisation
    loss='categorical_crossentropy',# Fonction de perte adaptée à la classification
    metrics=['accuracy']# Suivi de la précision pendant l’entraînement
)

model.summary() # Affiche la structure du modèle
plot_model(model, to_file='cnn_architecture.png', show_shapes=True, show_layer_names=True)  # Sauvegarde du schéma

batch_size = 64# Nombre d’images par lot
epochs = 35# Nombre total d’époques d’entraînement

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # Arrêt si pas d’amélioration
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')# Sauvegarde du meilleur modèle

import time
start_time = time.time()# Début du chronométrage

history = model.fit(# Entraînement du modèle
    x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
    callbacks=[early_stop , checkpoint]
)
end_time = time.time()# Fin du chronométrage
training_time = end_time - start_time # Calcul de la durée
print("training time: " , training_time) # Affichage du temps d’entraînement

# évaluation du modèle sur le test set
test_loss, test_acc = model.evaluate(x_test, y_test)# Évaluation sur données test
print("Test Accuracy:", test_acc) # Précision sur test
print("Test Loss:", test_loss) # Perte sur test

# visualisation graphique
plt.plot(history.history['accuracy'], label='train acc')# Courbe précision d’entraînement
plt.plot(history.history['val_accuracy'], label='val acc') # Courbe précision validation
plt.title('Accuracy au fil des époques')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='train loss') # Courbe perte d’entraînement
plt.plot(history.history['val_loss'], label='val loss') # Courbe perte validation
plt.title('Loss au fil des époques')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

y_pred_probs = model.predict(x_test)# Prédictions du modèle (probabilités)
y_pred = np.argmax(y_pred_probs, axis=1)# Conversion en classes
y_true = np.argmax(y_test, axis=1)# Conversion des labels one-hot

for i, class_name in enumerate(class_names):# Boucle sur chaque classe
    y_true_binary = (y_true == i).astype(int)# Labels binaires (classe vs pas classe)
    y_pred_binary = (y_pred == i).astype(int) # Prédictions binaires
    cm = confusion_matrix(y_true_binary, y_pred_binary)# Matrice de confusion binaire
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f'Not {class_name}', class_name])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Matrice de Confusion binaire pour la classe "{class_name}"')
    plt.show()

from sklearn.metrics import classification_report
import numpy as np

y_pred = model.predict(x_test)# Prédictions du modèle
y_pred_classes = np.argmax(y_pred, axis=1)# Conversion en classes
y_true = np.argmax(y_test, axis=1)# Vraies classes

print(classification_report(y_true, y_pred_classes, target_names=class_names)) # Rapport complet (precision, recall, F1)

import numpy as np
correct_indices = np.where(y_pred_classes == y_true)[0]# Indices des prédictions correctes
incorrect_indices = np.where(y_pred_classes != y_true)[0]# Indices des erreurs

plt.figure(figsize=(10, 6))
for i, idx in enumerate(correct_indices[:6]):# Affiche 6 exemples corrects
    plt.subplot(2, 3, i+1)
    plt.imshow(x_test[idx])
    plt.title(f"Vrai: {class_names[y_true[idx]]} | Prédit: {class_names[y_pred_classes[idx]]}")
    plt.axis("off")
plt.suptitle("Exemples de prédictions correctes")
plt.show()

plt.figure(figsize=(10, 6))
for i, idx in enumerate(incorrect_indices[:6]):# Affiche 6 erreurs de prédiction
    plt.subplot(2, 3, i+1)
    plt.imshow(x_test[idx])
    plt.title(f"Vrai: {class_names[y_true[idx]]} | Prédit: {class_names[y_pred_classes[idx]]}")
    plt.axis("off")
plt.suptitle("Exemples de prédictions incorrectes")
plt.show()

