import glob
import tensorflow as tf
import numpy as np
import tifffile as tiff
from sklearn.model_selection import train_test_split
from keras.utils import normalize

def read_image(filename):
    filename = filename.decode('utf-8')
    image = tiff.imread(filename)
    image = normalize(image.astype(np.float32))
    return image

def read_mask(filename):
    filename = filename.decode('utf-8')
    mask = tiff.imread(filename).astype(np.float32)
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=-1)
    mask = mask / 255.0
    return mask

def load_dataset(image_pattern, mask_pattern, test_size=0.1, random_state=42):
    image_files = sorted(glob.glob(image_pattern))
    mask_files = sorted(glob.glob(mask_pattern))
    print('\nPrimeras rutas de imágenes:', image_files[:5])
    print('Primeras rutas de máscaras:', mask_files[:5])
    
    train_val_images, test_images, train_val_masks, test_masks = train_test_split(
        image_files, mask_files, test_size=test_size, random_state=random_state)
    
    # Ajustar el tamaño de val_size dentro de train+val para obtener el 20% de los datos
    val_size_adjusted = 0.2 / (1 - test_size)  # Ajuste para que val sea 20% del total
    train_images, val_images, train_masks, val_masks = train_test_split(
        train_val_images, train_val_masks, test_size=val_size_adjusted, random_state=random_state)
    
    print("Cantidad de imágenes - Train: {}, Val: {}, Test: {}".format(len(train_images), len(val_images), len(test_images)))
    
    # Guardar el número de ejemplos (antes del pipeline)
    train_count = len(train_images)
    val_count = len(val_images)
    test_count = len(test_images)
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_masks))
    
    def load_data(image_path, mask_path):
        image = tf.numpy_function(read_image, [image_path], tf.float32)
        mask = tf.numpy_function(read_mask, [mask_path], tf.float32)
        image.set_shape([512, 512, None])
        mask.set_shape([512, 512, 1])
        return image, mask
    
    train_ds = train_ds.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    # def filter_black(image, mask):
    #     return tf.reduce_mean(mask) > 0.05
    # train_ds = train_ds.filter(filter_black)
    # val_ds = val_ds.filter(filter_black)
    # test_ds = test_ds.filter(filter_black)
    
    return train_ds, val_ds, test_ds, train_count, val_count, test_count
