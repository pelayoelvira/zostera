from keras.models import load_model
from keras.utils import plot_model
import numpy as np
import tensorflow as tf
from keras.models import load_model
from patchify import patchify, unpatchify
import rasterio
from rasterio.enums import Resampling
from keras.utils import normalize  # Para normalizar las imágenes
import os
import cv2
from keras.layers import Activation, Add


# Definición de funciones personalizadas
@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, (tf.shape(y_true)[0], -1))  # Mantiene la dimensión del batch
    y_pred_f = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1))  # Mantiene la dimensión del batch
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)  # Suma por cada muestra
    dice_per_sample = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1) + smooth)
    
    return 1 - tf.reduce_mean(dice_per_sample)  # Promedia la pérdida en el batch

@tf.keras.utils.register_keras_serializable()
def iou_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, (-1,))
    y_pred_f = tf.reshape(y_pred, (-1,))

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou  # Como es una loss, se minimiza

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
class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.6, smooth=1e-6, name="combined_loss", **kwargs):
        super(CombinedLoss, self).__init__(name=name, **kwargs)
        self.alpha = alpha
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        true_positives = tf.reduce_sum(y_true_f * y_pred_f)
        predicted_positives = tf.reduce_sum(y_pred_f)
        actual_positives = tf.reduce_sum(y_true_f)
        precision = (true_positives + self.smooth) / (predicted_positives + self.smooth)
        recall = (true_positives + self.smooth) / (actual_positives + self.smooth)
        return self.alpha * (1 - precision) + (1 - self.alpha) * (1 - recall)

    def get_config(self):
        config = super(CombinedLoss, self).get_config()
        config.update({
            "alpha": self.alpha,
            "smooth": self.smooth
        })
        return config


# Carga del modelo con los objetos personalizados
model = load_model('experiment_2/res01/filtrado.keras', custom_objects={'CombinedLoss': CombinedLoss,
                                                  'positive_precision': positive_precision,
                                                  'positive_recall': positive_recall,
                                                  'pixel_accuracy': pixel_accuracy})


import matplotlib.pyplot as plt
import visualkeras

from PIL import ImageFont

font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)

visualkeras.layered_view(model, legend=True, font=font, to_file='Keras_Unet.png', type_ignore=[Activation, Add], spacing=50, draw_volume=False)  # font is optional!

