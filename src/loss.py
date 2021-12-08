import tensorflow as tf

def nerve_seg_custom_loss(y_true, y_pred):
  def dice_loss(y_true, y_pred):
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator

  print(y_true.shape, y_pred.shape, 'In loss function')
  y_true = tf.cast(y_true, tf.float32)
  o = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred) + dice_loss(y_true, y_pred)
  return tf.reduce_mean(o)