import keras.losses
import tensorflow as tf


def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - numerator / denominator


def sparse_cce_dice_combination_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    sparse_cce = keras.losses.SparseCategoricalCrossentropy()
    o = sparse_cce(y_true, y_pred) + dice_loss(y_true, y_pred)
    return tf.reduce_mean(o)


class SparseMeanIoU(tf.keras.metrics.MeanIoU):

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, tf.argmax(y_pred, axis=-1), sample_weight)


"""
    Other diceloss implementation
    def dice_loss(y_true, y_pred, smooth=1.):
        # y_true_f = tf.compat.v1.layers.flatten(y_true)  # y_true stretch to one dimension
        # y_pred_f = tf.compat.v1.layers.flatten(y_pred)
        y_true_f, y_pred_f = y_true, y_pred
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice_coef = (2. * intersection + smooth) / (
                    tf.reduce_sum(y_true_f * y_true_f) + tf.reduce_sum(y_pred_f * y_pred_f) + smooth)
        return 1 - dice_coef

"""
