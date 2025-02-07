import os
import time
import tensorflow as tf
import numpy as np
import optuna
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras_unet_tune import get_model  # Asegúrate de que esta función esté definida para tu modelo UNET
from loss import positive_precision, positive_recall, pixel_accuracy, combined_loss, dice_loss
from tifffile import imwrite
from load_dataset_copy import load_dataset

# Rutas y patrón de archivos
image_dir = "Data/filtered_patches/images/*.tif"
mask_dir = "Data/filtered_patches/masks/*.tif"

# Cargar los datasets y contadores de ejemplos
train_ds, val_ds, test_ds, train_count, val_count, test_count = load_dataset(image_dir, mask_dir)

img_height = 512
img_width = 512

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

# Callbacks
class SavePredictionsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, output_dir="epoch_predictions"):
        super(SavePredictionsCallback, self).__init__()
        self.val_data = val_data
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        val_batch = next(iter(self.val_data.take(1)))
        val_images, val_masks = val_batch
        predictions = self.model.predict(val_images)
        predictions = (predictions * 255).astype(np.uint8)
        epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch + 1}")
        os.makedirs(epoch_dir, exist_ok=True)
        for i in range(min(5, val_images.shape[0])):
            imwrite(os.path.join(epoch_dir, f"input_{i}.tif"), (val_images[i].numpy() * 255).astype(np.uint8))
            imwrite(os.path.join(epoch_dir, f"mask_{i}.tif"), (val_masks[i].numpy() * 255).astype(np.uint8))
            imwrite(os.path.join(epoch_dir, f"pred_{i}.tif"), predictions[i])

class TimeLoggingCallback(tf.keras.callbacks.Callback):
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

def objective(trial):
    # Selección de hiperparámetros
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 4, 16, step=4)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    reduce_lr_factor = trial.suggest_float("reduce_lr_factor", 0.1, 0.5, step=0.05)
    early_stopping_patience = trial.suggest_int("early_stopping_patience", 10, 30)
    reduce_lr_patience = trial.suggest_int("reduce_lr_patience", 5, 15)
    
    # Configuración del dataset con batching dinámico
    train_dataset = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Crear el modelo con los hiperparámetros
    model = get_model(img_size=(img_height, img_width), dropout_rate=dropout_rate)

    # Callbacks dinámicos
    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor, patience=reduce_lr_patience, min_lr=1e-6)
    checkpoint = ModelCheckpoint(
        "best_model_1.keras", monitor="val_loss", save_best_only=True, mode="min", verbose=1
    )

    # Compilar el modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=dice_loss,
        metrics=[positive_precision, positive_recall, pixel_accuracy]
    )

    # Entrenar el modelo
    history = model.fit(
        train_dataset,
        epochs=100,
        validation_data=val_dataset,
        callbacks=[early_stopping, reduce_lr, TimeLoggingCallback(), checkpoint],
        verbose=2
    )

    # Evaluar en el conjunto de validación
    val_loss = min(history.history['val_loss'])
    
    # Liberar memoria de la GPU
    tf.keras.backend.clear_session()
    del model
    import gc
    gc.collect()
    
    return val_loss


# Configurar Optuna para guardar en SQLite
study = optuna.create_study(direction="minimize", storage="sqlite:///optuna_study_dice.db", load_if_exists=False)

# Optimizar
study.optimize(objective, n_trials=30)

# Imprimir resultados
print("Best trial:")
best_trial = study.best_trial
print(f"  Value: {best_trial.value}")
print("  Params:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Comando para iniciar el dashboard: optuna-dashboard sqlite:///optuna_study_1.db
