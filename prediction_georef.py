import numpy as np
import tensorflow as tf
from keras.models import load_model
from patchify import patchify, unpatchify
import rasterio
from rasterio.enums import Resampling
from keras.utils import normalize  # Para normalizar las imágenes
import os
import cv2

# Establecer la GPU a utilizar
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Verificar si TensorFlow reconoce la GPU especificada
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Usando GPU: {tf.config.experimental.get_visible_devices('GPU')}")
    except RuntimeError as e:
        print(e)
else:
    print("No se encontró GPU disponible.")

# Definición de funciones personalizadas
@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

@tf.keras.utils.register_keras_serializable()
def positive_precision(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_pred_pos = tf.round(tf.clip_by_value(y_pred_f, 0, 1))
    true_positives = tf.reduce_sum(y_true_f * y_pred_pos)
    predicted_positives = tf.reduce_sum(y_pred_pos)
    precision = (true_positives + smooth) / (predicted_positives + smooth)
    return precision

@tf.keras.utils.register_keras_serializable()
def positive_recall(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_pred_pos = tf.round(tf.clip_by_value(y_pred_f, 0, 1))
    true_positives = tf.reduce_sum(y_true_f * y_pred_pos)
    actual_positives = tf.reduce_sum(y_true_f)
    recall = (true_positives + smooth) / (actual_positives + smooth)
    return recall

@tf.keras.utils.register_keras_serializable()
def pixel_accuracy(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    correct_pixels = tf.equal(y_true_f, tf.round(y_pred_f))
    accuracy = tf.reduce_mean(tf.cast(correct_pixels, tf.float32))
    return accuracy
def precision_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    # Eliminamos tf.round para evitar problemas con gradientes
    true_positives = tf.reduce_sum(y_true_f * y_pred_f)
    predicted_positives = tf.reduce_sum(y_pred_f)
    precision = (true_positives + smooth) / (predicted_positives + smooth)
    return 1 - precision

def recall_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    true_positives = tf.reduce_sum(y_true_f * y_pred_f)
    actual_positives = tf.reduce_sum(y_true_f)
    recall = (true_positives + smooth) / (actual_positives + smooth)
    return 1 - recall

@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred, alpha=0.7, smooth=1e-6):
    precision = precision_loss(y_true, y_pred, smooth)
    recall = recall_loss(y_true, y_pred, smooth)
    return alpha * precision + (1 - alpha) * recall

# Carga del modelo con los objetos personalizados
model = load_model('trainingCopy_model1.1.keras', custom_objects={'dice_loss': combined_loss,
                                                  'positive_precision': positive_precision,
                                                  'positive_recall': positive_recall,
                                                  'pixel_accuracy': pixel_accuracy})

# Carga la imagen de alta resolución usando rasterio
input_path = 'Data/RESIZED/image_to_predict/RESIZED_20240410_VILLAVICIOSA_IZQ1.tif'
with rasterio.open(input_path) as src:
    imagen = src.read(out_shape=(src.count, src.height, src.width), resampling=Resampling.nearest)
    imagen = np.moveaxis(imagen, 0, -1)  # Cambiar el eje para tener (alto, ancho, canales)
    original_transform = src.transform  # Guardar la transformación original
    original_crs = src.crs  # Guardar el CRS original
    original_meta = src.meta  # Guardar los metadatos originales

# Ajusta la imagen a un tamaño divisible por 512
nuevo_alto = (imagen.shape[0] // 512) * 512
nuevo_ancho = (imagen.shape[1] // 512) * 512
imagen_ajustada = cv2.resize(imagen, (nuevo_ancho, nuevo_alto))

# Divide la imagen ajustada en patches de 512x512
patches = patchify(imagen_ajustada, (512, 512, imagen.shape[2]), step=512)

# Crea un array para almacenar las predicciones de los patches
predicted_patches = np.empty((patches.shape[0], patches.shape[1], 512, 512), dtype=np.uint8)

# Crear el directorio para almacenar las predicciones si no existe
output_dir = 'patches_prediction'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Realiza la inferencia en cada patch
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        patch = patches[i, j, 0]  # Extrae el patch de tamaño (512, 512, canales)
        
        # Normalizar y convertir a float32
        patch = normalize(patch)
        patch = patch.astype(np.float32)
        
        # Añadir batch dimension para el modelo
        patch = np.expand_dims(patch, axis=0)
        
        # Realizar la predicción
        pred = model.predict(patch)
        pred = tf.squeeze(pred, axis=(0, 3))  # Eliminar dimensiones extra
        pred = pred.numpy()  # Convertir tensor a NumPy
        predicted_patches[i, j] = (pred > 0.5).astype(np.uint8) * 255  # Escalar a 0-255
        
        # Guardar la máscara de cada parche
        patch_filename = os.path.join(output_dir, f'patch_{i}_{j}.tif')
        with rasterio.open(
            patch_filename,
            'w',
            driver='GTiff',
            height=512,
            width=512,
            count=1,
            dtype='uint8',
        ) as dst:
            dst.write(predicted_patches[i, j], 1)

# Reconstruir la máscara predicha completa
mascara_predicha = unpatchify(predicted_patches, (nuevo_alto, nuevo_ancho))

# Redimensiona la máscara predicha al tamaño original de la imagen si es necesario
mascara_predicha = cv2.resize(mascara_predicha, (imagen.shape[1], imagen.shape[0]), interpolation=cv2.INTER_NEAREST)

# Actualiza los metadatos originales para la máscara predicha
output_mask_path = '1.1_Copy_villaviciosa_IZQ1.tif'
output_meta = original_meta.copy()
output_meta.update({
    "driver": "GTiff",
    "height": mascara_predicha.shape[0],
    "width": mascara_predicha.shape[1],
    "count": 1,  # Solo una capa para la máscara
    "dtype": 'uint8',
    "transform": original_transform,  # Mantener la transformación original
    "crs": original_crs,  # Mantener el CRS original
})

# Guardar la máscara predicha completa con los metadatos originales
with rasterio.open(output_mask_path, 'w', **output_meta) as dst:
    dst.write(mascara_predicha, 1)
