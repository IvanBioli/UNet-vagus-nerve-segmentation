import numpy as np
from tensorflow import keras

from config import img_size, num_classes, batch_size, epochs, fine_tune_epochs, steps_per_epoch, validation_steps, initialise_run
from data_loader import VagusDataLoader
from data_utils import input_target_path_pairs
from eval import model_iou, one_prediction_iou
from model import get_model
from loss import nerve_segmentation_loss, SparseMeanIoU, dice_loss
import pickle


def train(model, model_id, train_img_target_pairs, val_img_target_pairs):

    _optimizer = keras.optimizers.Adam()
    _loss = nerve_segmentation_loss
    #_loss = keras.losses.SparseCategoricalCrossentropy()
    model.compile(
        optimizer=_optimizer,
        loss=_loss,
        metrics=[
            SparseMeanIoU(num_classes=num_classes, name='sparse_mean_iou'),
            dice_loss,
            keras.metrics.SparseCategoricalAccuracy(),
            keras.metrics.SparseCategoricalCrossentropy()
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

def fine_tune(trained_model_path, model_id, num_blocks_fine_tune, encoder_layers, train_img_target_pairs, val_img_target_pairs):

    trained_model = keras.models.load_model(trained_model_path, custom_objects={'sparse_cce_dice_combination_loss': nerve_segmentation_loss, 'SparseMeanIoU': SparseMeanIoU, 'dice_loss': dice_loss})

    # check the number of fine-tuning blocks is less or equal to the number of blocks in the encoder 
    assert num_blocks_fine_tune <= len(encoder_layers)

    # get the layer number to which we freeze every layers before it
    start_fine_tune_layer = encoder_layers[-num_blocks_fine_tune]

    # freeze all the layer before it
    for layer in trained_model.layers[:start_fine_tune_layer]:
        layer.trainable = False
    for layer in trained_model.layers[start_fine_tune_layer:]:
        layer.trainable = True

    #_optimizer = keras.optimizers.Adam(1e-5) # low learning rate
    _loss = nerve_segmentation_loss
    _optimizer = keras.optimizers.SGD(lr=1e-2)
    #_optimizer = keras.optimizers.Adam()
    #_loss = keras.losses.SparseCategoricalCrossentropy()
    trained_model.compile(
        optimizer=_optimizer,
        loss=_loss,
        metrics=[
            SparseMeanIoU(num_classes=num_classes, name='spare_mean_iou'),
            dice_loss,
            keras.metrics.SparseCategoricalAccuracy(),
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


def output_predictions(trained_model=None, trained_model_id=None):
    if trained_model is None and trained_model_id is None:
        raise ValueError('Must supply either model or model id to output predictions.')

    if trained_model_id is not None:
        trained_model = keras.models.load_model('model_checkpoints/trained_model_id.h5')

    # TODO call prediction code from notebook here as well
    print('Generating predictions')
    one_prediction_iou(
        trained_model,
        test_img=np.load(
            'data/vagus_dataset_10/validation/images/vago DX - 27.06.18 - HH - vetrino 1 - prox - campione 0002.npy'),
        test_anno=np.load(
            'data/vagus_dataset_10/validation/annotations/vago DX - 27.06.18 - HH - vetrino 1 - prox - campione 0002.npy')
    )


if __name__ == '__main__':
    initialise_run()
    model, decoder_layers, encoder_layers = get_model(img_size, num_classes)
    print("Decoder Layers: " + ' '.join(str(e) for e in decoder_layers))
    print("Encoder Layers: " + ' '.join(str(e) for e in encoder_layers))
    cur_model_id = 'Adam_512_SCCE'
    ft_model_id = 'Adam_512_SCCE_fine_tune'
    trained_model_path = 'model_checkpoints\Adam_512_SCCE.h5'
    m = train(
        model=model,
        model_id=cur_model_id,
        train_img_target_pairs=input_target_path_pairs('data/vagus_dataset_10/train'),
        val_img_target_pairs=input_target_path_pairs('data/vagus_dataset_10/validation')
    )
    # m = fine_tune(
    #     trained_model_path=trained_model_path,
    #     model_id=ft_model_id,
    #     num_blocks_fine_tune=4,
    #     encoder_layers=encoder_layers,
    #     train_img_target_pairs=input_target_path_pairs('data/transfer_learning_dataset/train'),
    #     val_img_target_pairs=input_target_path_pairs('data/transfer_learning_dataset/validation')
    # )

    # output_predictions(trained_model_id=cur_model_id)
    print('Done')
