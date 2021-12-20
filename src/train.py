import pickle

import tensorflow_addons as tfa
from tensorflow import keras

from config import batch_size, img_size, epochs
from data_loader import VagusDataLoader
from loss import iou_score, dice_loss, tversky_loss, focal_tversky_loss, custom_loss


def train(model, model_id, train_img_target_pairs, val_img_target_pairs):
    """
        Trains an iteration of the model
        :param model - keras model to be trained
        :param model_id - unique identifier for current training run
        :param train_img_target_pairs - train img mask pairs
        :param val_img_target_pairs - validation img mask pairs
    """
    _optimizer = keras.optimizers.Adam()
    _loss = custom_loss  # change loss functions here

    model.compile(
        optimizer=_optimizer,
        loss=_loss,
        metrics=[
            iou_score,
            dice_loss,
            tversky_loss,
            focal_tversky_loss,
            keras.metrics.SparseCategoricalCrossentropy(),
            keras.metrics.SparseCategoricalAccuracy(),
            tfa.losses.SigmoidFocalCrossEntropy(),
            custom_loss
        ]
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(f'model_checkpoints/{model_id}.h5', save_best_only=True)
    ]

    (img_paths, target_paths) = train_img_target_pairs
    assert len(img_paths) == len(target_paths)

    print('Training')

    train_x, train_y = img_paths, target_paths
    val_x, val_y = val_img_target_pairs

    assert len(train_x) == len(train_y)
    assert len(val_x) == len(val_y)

    print(f'Create train dataset with batch_size={batch_size}, img_size={img_size}, n={len(train_x)}')
    train_data = VagusDataLoader(batch_size, img_size, train_x, train_y)
    print(f'Create validation dataset with batch_size={batch_size}, img_size={img_size}, n={len(val_x)}')
    val_data = VagusDataLoader(batch_size, img_size, val_x, val_y)

    # Fit to current train and validation split
    model_history = model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=callbacks)

    print('Finished training')

    with open(f'model_losses/{model_id}.pkl', 'wb') as results_file:
        pickle.dump(model_history.history, results_file)

    return model
