import keras.losses
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras import backend as K


def pre_process_vectors(y_true, y_pred, logits=True):
    y_true = tf.cast(y_true, tf.float32)
    if logits:
        y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return y_true, y_pred


def dice_loss(y_true, y_pred, logits=True):
    y_true, y_pred = pre_process_vectors(y_true, y_pred, logits)
    numerator = 2 * K.sum(y_true * y_pred)
    denominator = K.sum(y_true + y_pred)
    return 1 - numerator / denominator


def tversky_loss(y_true, y_pred, logits=True):
    y_true, y_pred = pre_process_vectors(y_true, y_pred, logits)
    # computed as the average ratio between the area of the fascicles and the area of the background
    beta = 0.154
    numerator = K.sum(y_true * y_pred)
    denominator = K.sum(y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred))
    return 1 - numerator / denominator


def focal_tversky_loss(y_true, y_pred, logits=True):
    gamma = 4 / 3
    return tversky_loss(y_true, y_pred, logits)**gamma

def diff_tversky_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    #y_pred = tf.cast(y_pred, tf.float32)
    #y_true = K.flatten(y_true)
    #y_pred = K.flatten(y_pred)
    beta = 0.154
    numerator = K.sum(y_true * y_pred)
    denominator = K.sum(y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred))
    return 1 - numerator / denominator
'''
def focal_loss(gamma=2., alpha=.5):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+K.epsilon())) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
	return focal_loss_fixed
'''
def focal_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    gamma = 1
    return -K.mean(K.pow(1 - y_pred, gamma) * K.log(y_pred + K.epsilon()) * y_true + K.pow(y_pred, gamma) * K.log(1 - y_pred + K.epsilon()) * (1 - y_true))

def custom_loss(y_true, y_pred):
    bce = keras.losses.SparseCategoricalCrossentropy()
    fl = tfa.losses.SigmoidFocalCrossEntropy()
    return bce(y_true, y_pred) + fl(y_true, y_pred)

def nerve_segmentation_loss(y_true, y_pred, logits=True):
    y_true = tf.cast(y_true, tf.float32)
    sparse_cce = keras.losses.SparseCategoricalCrossentropy()
    o = sparse_cce(y_true, y_pred)
    return tf.reduce_mean(o)


def iou_score(y_true, y_pred, logits=True):
    y_true, y_pred = pre_process_vectors(y_true, y_pred, logits)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return intersection / union


if __name__ == '__main__':
    """ Test IoU values for images """
    from config import initialise_run

    initialise_run()

    # all_ones = np.array([[1, 1], [1, 1]])
    # half_ones = np.array([[1, 1], [0, 0]])
    # all_zero = np.array([[0, 0], [0, 0]])
    # print(iou_score(half_ones, all_ones))

    ann = np.load(
        'data/vagus_dataset_11/train/annotations/vago DX - 27.06.18 - HH - vetrino 1 - prox - campione 0010.npy')
    img = np.load('data/vagus_dataset_11/train/images/vago DX - 27.06.18 - HH - vetrino 1 - prox - campione 0010.npy')
    print(f'Annotation values: {np.unique(ann, return_counts=True)}')
    # model = keras.models.load_model('model_checkpoints/Adam_512_sparse_cce_14_12.h5', custom_objects={
    #     'nerve_segmentation_loss': nerve_segmentation_loss,
    #     # 'SparseMeanIoU': SparseMeanIoU,
    #     'dice_loss': dice_loss,
    #     'tversky_loss': tversky_loss
    # })

    # y_pred = model.predict(np.expand_dims(img, axis=0))
    # iou = SparseMeanIoU(num_classes=num_classes)
    # iou.update_state(ann, y_pred)
    # print(iou.result())
    #
    # y4 = tf.concat([tf.ones([1, 512, 512, 1]), tf.zeros([1, 512, 512, 1])], axis=3)
    # ann2 = tf.ones([512, 512])
    # iou = SparseMeanIoU(num_classes=num_classes)
    # iou.update_state(ann2, y4)
    # print(iou.result())

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
