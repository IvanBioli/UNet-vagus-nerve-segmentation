import pickle

from tensorflow import keras

from config import num_classes, batch_size, img_size, fine_tune_epochs
from data_loader import VagusDataLoader
from loss import nerve_segmentation_loss, dice_loss, tversky_loss


def fine_tune(trained_model_path, model_id, num_blocks_fine_tune, encoder_layers, train_img_target_pairs, val_img_target_pairs):
    """
        Fine-tunes a model for transfer learning tasks

        Parameters
        ---------------        
        trained_model_path: str
            path to pre-trained model
        model_id: str
            unique identifier for current training run
        num_blocks_fine_tune: int
            number of blocks in encoder layers to fine-tune the weights
        encoder_layers: [int]
            starting layers of the encoder in the network
        train_img_target_pairs: ([str], [str])
            train img mask pairs
        val_img_target_pairs: ([str], [str])
            validation img mask pairs

        Returns
        ----------------
        the fine-tuned model
    """

    trained_model = keras.models.load_model(trained_model_path, custom_objects={
        'nerve_segmentation_loss': nerve_segmentation_loss,
        'dice_loss': dice_loss,
        'tversky_loss': tversky_loss
    })

    # check the number of fine-tuning blocks is less or equal to the number of blocks in the encoder
    assert num_blocks_fine_tune <= len(encoder_layers)

    # get the layer number to which we freeze every layers before it
    start_fine_tune_layer = encoder_layers[-num_blocks_fine_tune]

    # freeze all the layer before it
    for layer in trained_model.layers[:start_fine_tune_layer]:
        layer.trainable = False
    for layer in trained_model.layers[start_fine_tune_layer:]:
        layer.trainable = True

    _loss = nerve_segmentation_loss
    _optimizer = keras.optimizers.SGD(lr=1e-2)
    # _optimizer = keras.optimizers.Adam()
    trained_model.compile(
        optimizer=_optimizer,
        loss=_loss,
        metrics=[
            dice_loss,
            tversky_loss,
            keras.metrics.SparseCategoricalCrossentropy()
        ]
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(f'model_checkpoints/{model_id}.h5', save_best_only=True)
    ]

    (img_paths, target_paths) = train_img_target_pairs
    assert len(img_paths) == len(target_paths)

    print('Fine-tuning')

    train_x, train_y = img_paths, target_paths
    val_x, val_y = val_img_target_pairs

    assert len(train_x) == len(train_y)
    assert len(val_x) == len(val_y)

    print(f'Create train dataset with batch_size={batch_size}, img_size={img_size}, n={len(train_x)}')
    train_data = VagusDataLoader(batch_size, img_size, train_x, train_y)
    print(f'Create validation dataset with batch_size={batch_size}, img_size={img_size}, n={len(val_x)}')
    val_data = VagusDataLoader(batch_size, img_size, val_x, val_y)

    # Fit to current train and validation split
    model_history = trained_model.fit(train_data, epochs=fine_tune_epochs, validation_data=val_data, callbacks=callbacks)

    print('Finished training')

    with open(f'model_losses/{model_id}.pkl', 'wb') as results_file:
        pickle.dump(model_history.history, results_file)

    return trained_model
