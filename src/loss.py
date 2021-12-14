import keras.losses
import tensorflow as tf


def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - numerator / denominator


def tversky_loss(y_true, y_pred):
    beta = 0.3
    y_true = tf.cast(y_true, tf.float32)
    numerator = y_true * y_pred
    denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
    return 1 - tf.reduce_sum(numerator) / tf.reduce_sum(denominator)


def nerve_segmentation_loss(y_true, y_pred, multiplier=2):
    y_true = tf.cast(y_true, tf.float32)
    sparse_cce = keras.losses.SparseCategoricalCrossentropy()
    o = sparse_cce(y_true, y_pred) + multiplier * tversky_loss(y_true, y_pred)
    return tf.reduce_mean(o)


def sparse_cce_dice_combination_loss(y_true, y_pred, multiplier=2):
    y_true = tf.cast(y_true, tf.float32)
    sparse_cce = keras.losses.SparseCategoricalCrossentropy()
    o = sparse_cce(y_true, y_pred) + multiplier * tversky_loss(y_true, y_pred)
    return tf.reduce_mean(o)

class SparseMeanIoU(tf.keras.metrics.MeanIoU):

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, tf.argmax(y_pred, axis=-1), sample_weight)


if __name__ == '__main__':
    """ Test IoU values for images """
    import numpy as np
    from src.config import initialise_run, num_classes
    initialise_run()

    ann = np.load('data/vagus_dataset_11/train/annotations/vago DX - 27.06.18 - HH - vetrino 1 - prox - campione 0010.npy')
    img = np.load('data/vagus_dataset_11/train/images/vago DX - 27.06.18 - HH - vetrino 1 - prox - campione 0010.npy')
    print(f'Annotation values: {np.unique(ann, return_counts=True)}')
    model = keras.models.load_model('model_checkpoints/Adam_512_SCCE_dataset_11.h5', custom_objects={
        'nerve_segmentation_loss': nerve_segmentation_loss,
        'SparseMeanIoU': SparseMeanIoU,
        'dice_loss': dice_loss,
        'tversky_loss': tversky_loss
    })
    y_pred = model.predict(np.expand_dims(img, axis=0))
    iou = SparseMeanIoU(num_classes=num_classes)
    iou.update_state(ann, y_pred)
    print(iou.result())

    y4 = tf.concat([tf.ones([1, 512, 512, 1]), tf.zeros([1, 512, 512, 1])], axis=3)
    ann2 = tf.ones([512, 512])
    iou = SparseMeanIoU(num_classes=num_classes)
    iou.update_state(ann2, y4)
    print(iou.result())




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
