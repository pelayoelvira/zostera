import keras.backend as K
import tensorflow as tf
from keras.saving import register_keras_serializable


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])  # Aplanar
    y_pred_f = tf.reshape(y_pred, [-1])  # Aplanar
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

import tensorflow as tf

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




# Precision positiva
def positive_precision(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    # y_true_f = tf.cast(y_true_f, tf.float32)
    y_pred_f = tf.reshape(y_pred, [-1])

    y_pred_pos = tf.round(tf.clip_by_value(y_pred_f, 0, 1))
    true_positives = tf.reduce_sum(y_true_f * y_pred_pos)
    predicted_positives = tf.reduce_sum(y_pred_pos)
    
    precision = (true_positives + smooth) / (predicted_positives + smooth)
    return precision

# Recall positiva
def positive_recall(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    # y_true_f = tf.cast(y_true_f, tf.float32)
    y_pred_f = tf.reshape(y_pred, [-1])

    y_pred_pos = tf.round(tf.clip_by_value(y_pred_f, 0, 1))
    true_positives = tf.reduce_sum(y_true_f * y_pred_pos)
    actual_positives = tf.reduce_sum(y_true_f)
    
    recall = (true_positives + smooth) / (actual_positives + smooth)
    return recall

def pixel_accuracy(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    y_pred_f = tf.cast(tf.greater(y_pred_f, 0.5), tf.float32)  # Redondear predicciones
    correct_pixels = tf.equal(y_true_f, y_pred_f)
    accuracy = tf.reduce_mean(tf.cast(correct_pixels, tf.float32))
    return accuracy



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



