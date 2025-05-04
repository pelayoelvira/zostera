import keras.backend as K
import tensorflow as tf
from keras.saving import register_keras_serializable


def precision(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    y_pred_pos = tf.round(tf.clip_by_value(y_pred_f, 0, 1))
    true_positives = tf.reduce_sum(y_true_f * y_pred_pos)
    predicted_positives = tf.reduce_sum(y_pred_pos)
    
    precision = (true_positives + smooth) / (predicted_positives + smooth)
    return precision


def recall(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
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



