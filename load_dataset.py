import glob
import numpy as np
import tifffile as tiff
from sklearn.model_selection import train_test_split
from keras.utils import normalize
import numpy as np

def filter_black_patches(images, masks, threshold=0.05):
    non_black_images = []
    non_black_masks = []
    for img, mask in zip(images, masks):
        if np.mean(mask) > threshold:  # Ajusta el umbral según necesites (0.05 o 5%)
            non_black_images.append(img)
            non_black_masks.append(mask)
    return np.array(non_black_images), np.array(non_black_masks)

def load_dataset(image_dir, mask_dir, test_size=0.15, random_state=42):
    image_names = glob.glob(image_dir)
    image_names.sort()
    print('\nimage_names:', image_names[:5])
    
    mask_names = glob.glob(mask_dir)
    mask_names.sort()
    print('mask_names:', mask_names[:5])

    mask_dataset = [tiff.imread(mask) for mask in mask_names]
    mask_dataset = np.array(mask_dataset)

    image_dataset = [tiff.imread(image) for image in image_names] 
    image_dataset = np.array(image_dataset)
    print('image_dataset shape:', image_dataset.shape)

    # Normalizar las imágenes
    image_dataset = normalize(image_dataset) # Normalizar las imágenes, ahora estan en float32

    # Normalizar máscaras y mantenerlas sin más preprocesamiento
    mask_dataset = np.expand_dims(mask_dataset, axis=-1)
    mask_dataset = mask_dataset.astype(np.float32) / 255.0
    print('mask_dataset shape:', mask_dataset.shape)
    
    # Filtrar mascaras completamente negras
    image_dataset, mask_dataset = filter_black_patches(image_dataset, mask_dataset)
    print('Filtered image_dataset shape:', image_dataset.shape)
    print('Filtered mask_dataset shape:', mask_dataset.shape)
    
    # Dividir inicialmente en entrenamiento+validación y prueba
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        image_dataset, mask_dataset, test_size=test_size, random_state=random_state)
    
    # Dividir el conjunto de entrenamiento+validación en entrenamiento y validación
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=0.1, random_state=random_state)
    
    # Imprimir las formas de los conjuntos divididos
    print('x_train shape:', x_train.shape, 'x_val shape:', x_val.shape, 'x_test shape:', x_test.shape)
    print('y_train shape:', y_train.shape, 'y_val shape:', y_val.shape, 'y_test shape:', y_test.shape)
    
    if len(y_test.shape) == 3:
        y_test = np.expand_dims(y_test, axis=-1)

    

    return x_train, x_val, x_test, y_train, y_val, y_test
