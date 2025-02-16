import os
import tensorflow as tf
import numpy as np
from load_dataset import load_dataset
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from model_script.keras_unet import get_model
import keras
from loss import  positive_precision, positive_recall, pixel_accuracy, combined_loss
from tifffile import imsave
import time


# Cargar el dataset
image_dir = "Data/filtered_patches/images/*.tif"
mask_dir = "Data/filtered_patches/masks/*.tif"
x_train, x_val, x_test, y_train, y_val, y_test = load_dataset(image_dir, mask_dir)

img_height = x_train.shape[1]
img_width = x_train.shape[2]
img_channels = x_train.shape[3]

batch_size = 8
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Establecer la GPU a utilizar
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Verificar si TensorFlow reconoce la GPU especificada
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Configurar memoria para crecimiento
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Usando GPU: {tf.config.experimental.get_visible_devices('GPU')}")
    except RuntimeError as e:
        print(e)
else:
    print("No se encontró GPU disponible.")
    
# Crear el modelo
model = get_model(img_size=(img_height, img_width))
model.summary()


# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=30, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.000001)

# Callback para guardar predicciones por época
class SavePredictionsCallback(keras.callbacks.Callback):
    def __init__(self, val_data, output_dir="epoch_predictions"):
        super(SavePredictionsCallback, self).__init__()
        self.val_data = val_data
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        val_images, val_masks = next(iter(self.val_data))
        predictions = self.model.predict(val_images)  # Ahora 'self.model' está accesible sin necesidad de establecerlo

        # Normalizar predicciones si es necesario (ajustar a uint8 para visualización)
        predictions = (predictions * 255).astype(np.uint8)

        # Guardar imágenes de predicción, imágenes de entrada y máscaras reales
        epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch + 1}")
        os.makedirs(epoch_dir, exist_ok=True)

        for i in range(min(5, val_images.shape[0])):  # Guardar las primeras 5 imágenes de cada época
            imsave(os.path.join(epoch_dir, f"input_{i}.tif"), (val_images[i].numpy() * 255).astype(np.uint8))
            imsave(os.path.join(epoch_dir, f"mask_{i}.tif"), (val_masks[i].numpy() * 255).astype(np.uint8))
            imsave(os.path.join(epoch_dir, f"pred_{i}.tif"), predictions[i])

class TimeLoggingCallback(keras.callbacks.Callback):
    def __init__(self):
        super(TimeLoggingCallback, self).__init__()
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - self.epoch_start_time
        print(f"Epoch {epoch + 1}: {epoch_duration:.2f} seconds")
        
    def on_train_end(self, logs=None):
        total_training_time = time.time() - self.start_time
        print(f"Total training time: {total_training_time:.2f} seconds")


# Compilar el modelo    
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=3e-4),
    loss=combined_loss,
    metrics=[positive_precision, positive_recall, pixel_accuracy]
)


# Configurar entrenamiento con el callback personalizado
model.fit(
    train_dataset,
    epochs=500,
    validation_data=val_dataset,
    callbacks=[early_stopping, reduce_lr, SavePredictionsCallback(val_dataset), TimeLoggingCallback()],
    verbose=2
)

# Evaluar el modelo en el conjunto de test

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_loss, test_positive_precision, test_positive_recall, test_pixel_accuracy = model.evaluate(test_dataset, verbose=2)


# Imprimir las métricas para el conjunto de test
print("\nResultados en el conjunto de test:")
print(f"Loss: {test_loss:.4f}")
print(f"Positive Precision: {test_positive_precision:.4f}")
print(f"Positive Recall: {test_positive_recall:.4f}")
print(f"Pixel Accuracy: {test_pixel_accuracy:.4f}")


# Guardar el modelo
model.save("3_model0.000001.keras")
