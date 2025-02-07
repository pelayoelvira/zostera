import tensorflow as tf
import keras.backend as K
from keras.saving import register_keras_serializable
from keras.metrics import MeanMetricWrapper

# ---------------------------
# Funciones de métricas personalizadas
# ---------------------------

def positive_precision(y_true, y_pred, smooth=1e-6):
    """
    Calcula la precisión positiva: fracción de píxeles predichos como positivos que son
    verdaderamente positivos.
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    # Redondeamos las predicciones al entero más cercano (0 o 1)
    y_pred_pos = tf.round(tf.clip_by_value(y_pred_f, 0, 1))
    true_positives = tf.reduce_sum(y_true_f * y_pred_pos)
    predicted_positives = tf.reduce_sum(y_pred_pos)
    
    precision = (true_positives + smooth) / (predicted_positives + smooth)
    return precision

def positive_recall(y_true, y_pred, smooth=1e-6):
    """
    Calcula el recall (sensibilidad) positivo: fracción de píxeles positivos reales
    que se han predicho correctamente.
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_pred_pos = tf.round(tf.clip_by_value(y_pred_f, 0, 1))
    true_positives = tf.reduce_sum(y_true_f * y_pred_pos)
    actual_positives = tf.reduce_sum(y_true_f)
    
    recall = (true_positives + smooth) / (actual_positives + smooth)
    return recall

def pixel_accuracy(y_true, y_pred):
    """
    Calcula la precisión de píxeles: fracción de píxeles en los que la predicción coincide
    con la etiqueta real.
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_pred_f = tf.cast(tf.greater(y_pred_f, 0.5), tf.float32)  # Umbral de 0.5 para decidir 0 o 1
    correct_pixels = tf.equal(y_true_f, y_pred_f)
    accuracy = tf.reduce_mean(tf.cast(correct_pixels, tf.float32))
    return accuracy

# ---------------------------
# Funciones de pérdidas basadas en precisión y recall
# ---------------------------

def precision_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
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

def combined_loss(y_true, y_pred, alpha=0.6, smooth=1e-6):
    """
    Pérdida combinada que pondera la pérdida de precisión y recall.  
    alpha controla el peso de la pérdida de precisión.
    """
    precision = precision_loss(y_true, y_pred, smooth)
    recall = recall_loss(y_true, y_pred, smooth)
    return alpha * precision + (1 - alpha) * recall

# ---------------------------
# Wrappers para usar las funciones anteriores como métricas en model.compile()
# ---------------------------

positive_precision_metric = MeanMetricWrapper(fn=positive_precision, name="positive_precision")
positive_recall_metric = MeanMetricWrapper(fn=positive_recall, name="positive_recall")
pixel_accuracy_metric = MeanMetricWrapper(fn=pixel_accuracy, name="pixel_accuracy")
