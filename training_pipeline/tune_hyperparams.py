import os
import time
import tensorflow as tf
import numpy as np
import optuna
from tensorflow import keras
import gc  # Para liberar memoria
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from model_script.keras_unet import get_model  
from training_pipeline.loss import positive_precision, positive_recall, pixel_accuracy, CombinedLoss
from training_pipeline.load_dataset_v2 import load_dataset

# Rutas y patrón de archivos
image_dir = "Data/filtered_patches/filtered_images/*.tif"
mask_dir = "Data/filtered_patches/filtered_masks/*.tif"

# Cargar los datasets (se cargan una vez y se reutilizan)
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

class TimeLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TimeLoggingCallback, self).__init__()
        self.start_time = None

    def on_train_begin(self, _logs=None):
        self.start_time = time.time()

    def on_epoch_begin(self, _epoch, _logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, _logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        print(f"Epoch {epoch + 1}: {epoch_duration:.2f} seconds")

    def on_train_end(self, _logs=None):
        total_training_time = time.time() - self.start_time
        print(f"Total training time: {total_training_time:.2f} seconds")

def objective(trial):
    try:
        # Selección de hiperparámetros
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_int("batch_size", 4, 32, step=4)
        reduce_lr_factor = trial.suggest_float("reduce_lr_factor", 0.05, 0.5, step=0.05)
        early_stopping_patience = trial.suggest_int("early_stopping_patience", 10, 50)
        reduce_lr_patience = trial.suggest_int("reduce_lr_patience", 5, 20)

        # Se reutilizan los datasets ya cargados
        train_dataset = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Crear el modelo con los hiperparámetros
        model = get_model(img_size=(img_height, img_width))

        # Callbacks dinámicos
        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, mode='min', min_delta=0.000001)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor, patience=reduce_lr_patience, min_lr=1e-6)
        checkpoint = ModelCheckpoint(
            "best_model_tuned_2.keras", monitor="val_loss", save_best_only=True, mode="min", verbose=1
        )

        # Compilar el modelo
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
            loss=CombinedLoss(alpha=0.5),
            metrics=[positive_precision, positive_recall, pixel_accuracy]
        )

        # Entrenar el modelo
        history = model.fit(
            train_dataset,
            epochs=500,
            validation_data=val_dataset,
            callbacks=[early_stopping, reduce_lr, TimeLoggingCallback(), checkpoint],
            verbose=2
        )

        # Evaluar en el conjunto de validación
        val_loss = min(history.history['val_loss'])

        return val_loss

    finally:
        # Liberar memoria correctamente después de cada trial
        tf.keras.backend.clear_session()
        gc.collect()

# Configurar Optuna para guardar en SQLite (sin sobrescribir la base de datos)
study = optuna.create_study(direction="minimize", storage="sqlite:///optuna_study_combined_2.db", load_if_exists=True)

# Optimizar
study.optimize(objective, n_trials=30, n_jobs=1)

# Imprimir resultados
print("Best trial:")
best_trial = study.best_trial
print(f"  Value: {best_trial.value}")
print("  Params:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")
# Comando para iniciar el dashboard: optuna-dashboard sqlite:///optuna_study_combined.db
