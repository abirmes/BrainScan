import os 
from PIL import Image
import cv2
import numpy as np

image_directory = 'Data/'
no_tumor_image = os.listdir(image_directory + 'notumor/')
glioma_image = os.listdir(image_directory + 'glioma/')
meningioma_image = os.listdir(image_directory + 'meningioma/')
pituitary_image = os.listdir(image_directory + 'pituitary/')
import matplotlib.pyplot as plt

# print(len(no_tumor_image))

# for i , image_name in enumerate(no_tumor_image):
#     if((image_name.split('.')[1] == 'jpg') or (image_name.split('.')[1] == 'jpeg') or (image_name.split('.')[1] == 'bmp') or (image_name.split('.')[1] == 'png') ):
#         image = cv2.imread(image_directory + 'notumor/' + image_name)
#         image = Image.fromarray(image , 'RGB')
#         image = image.resize((224 , 224))
#         dataset.append(np.array(image))
#         label.append(0)


# for i , image_name in enumerate(glioma_image):
#     if((image_name.split('.')[1] == 'jpg') or (image_name.split('.')[1] == 'jpeg') or (image_name.split('.')[1] == 'bmp') or (image_name.split('.')[1] == 'png') ):
#         image = cv2.imread(image_directory + 'glioma/' + image_name)
#         image = Image.fromarray(image , 'RGB')
#         image = image.resize((224 , 224))
#         dataset.append(np.array(image))
#         label.append(1)
        
        
        
# for i , image_name in enumerate(meningioma_image):
#     if((image_name.split('.')[1] == 'jpg') or (image_name.split('.')[1] == 'jpeg') or (image_name.split('.')[1] == 'bmp') or (image_name.split('.')[1] == 'png') ):
#         image = cv2.imread(image_directory + 'meningioma/' + image_name)
#         image = Image.fromarray(image , 'RGB')
#         image = image.resize((224 , 224))
#         dataset.append(np.array(image))
#         label.append(2)
        
        

# for i , image_name in enumerate(pituitary_image):
#     if((image_name.split('.')[1] == 'jpg') or (image_name.split('.')[1] == 'jpeg') or (image_name.split('.')[1] == 'bmp') or (image_name.split('.')[1] == 'png') ):
#         image = cv2.imread(image_directory + 'pituitary/' + image_name)
#         image = Image.fromarray(image , 'RGB')
#         image = image.resize((224 , 224))
#         dataset.append(np.array(image))
#         label.append(3)
        
        
        
        
        
        
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
            image = image.resize((224, 224)) # redimensionne facilement l’image
            dataset.append(np.array(image)) # retransforme en tableau NumPy pour ML
            label.append(folder_label)


dataset = np.array(dataset)
label = np.array(label)




classes, counts = np.unique(label, return_counts=True) # compter combien d’images pour chaque classe
class_names = list(folders.keys())
plt.figure(figsize=(8,5))
plt.bar(class_names, counts, color='skyblue')
plt.xlabel('Classes')
plt.ylabel('Nombre d’images')
plt.title('Nombre d’images par classe')
plt.show()











