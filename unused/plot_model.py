from keras.models import load_model
from keras.utils import plot_model
import numpy as np
import tensorflow as tf
from keras.models import load_model
from patchify import patchify, unpatchify 

from keras.layers import Activation, Add
import matplotlib.pyplot as plt
import visualkeras

from PIL import ImageFont


@tf.keras.utils.register_keras_serializable()
def precision(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_pred_pos = tf.round(tf.clip_by_value(y_pred_f, 0, 1))
    true_positives = tf.reduce_sum(y_true_f * y_pred_pos)
    predicted_positives = tf.reduce_sum(y_pred_pos)
    precision = (true_positives + smooth) / (predicted_positives + smooth)
    return precision

@tf.keras.utils.register_keras_serializable()
def recall(y_true, y_pred, smooth=1e-6):
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


@tf.keras.utils.register_keras_serializable()
class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.5, smooth=1e-6, name="combined_loss", **kwargs):
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
                                                  'positive_precision': precision,
                                                  'positive_recall': recall,
                                                  'pixel_accuracy': pixel_accuracy})




font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)

visualkeras.layered_view(model, legend=True, font=font, to_file='Keras_Unet.png', type_ignore=[Activation, Add], spacing=50, draw_volume=False)  # font is optional!

