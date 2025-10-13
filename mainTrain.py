import os 
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


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
            image = cv2.imread(image_directory + folder_name + '/' + image_name) # lit l’image depuis le disque en tableau NumPy
            image = Image.fromarray(image, 'RGB') # convertit ce tableau en objet PIL
            image = image.resize((224, 224)) # Redimensionner les images à une taille fixe 224×224 à l’aide de la bibliothèque OpenCV.
            dataset.append(np.array(image)) # retransforme en tableau NumPy pour ML 
            label.append(folder_label) #Chaque image et son label doivent avoir le même indice dans les deux listes.
            
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
classes, counts = np.unique(label, return_counts=True)
inverse_folders = {v: k for k, v in folders.items()}
for c, count in zip(classes, counts):
    print(f"Classe {inverse_folders[c]}: {count} images")
# Classe notumor: 2000 images
# Classe glioma: 1621 images
# Classe meningioma: 1645 images
# Classe pituitary: 1757 images


#utilisation de data augmentaion


 








