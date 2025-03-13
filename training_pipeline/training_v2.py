import os
import tensorflow as tf
import numpy as np
import time
from training_pipeline.load_dataset_v2 import load_dataset
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
import sys
from keras.callbacks import TensorBoard
from model_script.keras_unet import get_model
import keras
from functools import partial
from training_pipeline.loss import positive_precision, positive_recall, pixel_accuracy, CombinedLoss, dice_loss, iou_loss
from tifffile import imwrite  # Usamos imwrite en lugar de imsave

# Rutas y patrón de archivos
image_dir = "Data/filtered_patches/filtered_images/*.tif"
mask_dir = "Data/filtered_patches/filtered_masks/*.tif"

# Cargar los datasets y contadores de ejemplos
train_ds, val_ds, test_ds, train_count, val_count, test_count = load_dataset(image_dir, mask_dir)

img_height = 512
img_width = 512
batch_size = 32

# Preparar el dataset de entrenamiento (se repite indefinidamente),
# y agrupar en batches con prefetch para rendimiento
train_dataset = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Configuración de la GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

# # Crear el modelo (asegúrate de que get_model acepte el tamaño de entrada adecuado)
model = get_model(img_size=(img_height, img_width))
model.summary()


# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=30, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-6)
tensorboard_callback = TensorBoard(log_dir='experiment_1/logs_filtrado', histogram_freq=1, write_graph=False, write_images=False)


# class SavePredictionsCallback(keras.callbacks.Callback):
#     def __init__(self, val_data, output_dir="epoch_predictions"):
#         super(SavePredictionsCallback, self).__init__()
#         self.val_data = val_data
#         self.output_dir = output_dir
#         os.makedirs(self.output_dir, exist_ok=True)

#     def on_epoch_end(self, epoch, logs=None):
#         # Para evitar que el iterador se agote, usamos .take(1)
#         val_batch = next(iter(self.val_data.take(1)))
#         val_images, val_masks = val_batch
#         predictions = self.model.predict(val_images)
#         predictions = (predictions * 255).astype(np.uint8)
#         epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch + 1}")
#         os.makedirs(epoch_dir, exist_ok=True)
#         for i in range(min(5, val_images.shape[0])):
#             imwrite(os.path.join(epoch_dir, f"input_{i}.tif"), (val_images[i].numpy() * 255).astype(np.uint8))
#             imwrite(os.path.join(epoch_dir, f"mask_{i}.tif"), (val_masks[i].numpy() * 255).astype(np.uint8))
#             imwrite(os.path.join(epoch_dir, f"pred_{i}.tif"), predictions[i])

class TimeLoggingCallback(keras.callbacks.Callback):
    def __init__(self):
        super(TimeLoggingCallback, self).__init__()
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        print(f"Epoch {epoch + 1}: {epoch_duration:.2f} seconds")
        
    def on_train_end(self, logs=None):
        total_training_time = time.time() - self.start_time
        print(f"Total training time: {total_training_time:.2f} seconds")


# Callback para guardar el mejor modelo basado en la pérdida de validación
checkpoint = ModelCheckpoint(
    "experiment_1/filtrado.keras",  # Nombre del archivo para guardar
    monitor="val_loss",  # Métrica que se supervisará
    save_best_only=True,  # Solo guarda el mejor modelo
    mode="min",  # Queremos minimizar la pérdida
    verbose=1  # Mensajes de guardado
)


# Define el valor de alpha que deseas
alpha_value = 0.5

# Compila el modelo usando la clase de pérdida personalizada
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
    loss=CombinedLoss(alpha=alpha_value),
    metrics=[positive_precision, positive_recall, pixel_accuracy]
)


# Calcular los pasos por época usando el número de ejemplos de entrenamiento
steps_per_epoch = train_count // batch_size
print("Steps per epoch:", steps_per_epoch)

model.fit(
    train_dataset,
    epochs=2000,
    validation_data=val_dataset,
    callbacks=[early_stopping, reduce_lr, TimeLoggingCallback(), checkpoint, tensorboard_callback],
    verbose=2
)


# Evaluar en el conjunto de test
test_loss, test_positive_precision, test_positive_recall, test_pixel_accuracy = model.evaluate(test_dataset, verbose=2)
print("\nResultados en el conjunto de test:")
print(f"Loss: {test_loss:.4f}")
print(f"Positive Precision: {test_positive_precision:.4f}")
print(f"Positive Recall: {test_positive_recall:.4f}")
print(f"Pixel Accuracy: {test_pixel_accuracy:.4f}")


