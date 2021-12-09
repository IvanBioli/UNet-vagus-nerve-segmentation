import numpy as np
from tensorflow import keras

from config import img_size, num_classes, batch_size, epochs, steps_per_epoch, validation_steps, initialise_run
from data_loader import VagusDataLoader
from data_utils import input_target_path_pairs
from eval import model_iou, one_prediction_iou
from model import get_model
from loss import sparse_cce_dice_combination_loss

def train(model_id, train_img_target_pairs, val_img_target_pairs=None, cross_validation_folds=10):
    model_losses_and_metrics = []

    model = get_model(img_size, num_classes)

    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    #_optimizer = keras.optimizers.RMSprop()
    _optimizer = keras.optimizers.Adam()
    # _loss = sparse_cce_dice_combination_loss
    _loss = keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=_optimizer, loss=_loss,
                  # metrics=[keras.metrics.SparseCategoricalCrossentropy()]
                  metrics=[
                      sparse_cce_dice_combination_loss,
                      keras.metrics.SparseCategoricalCrossentropy()
                  ])

    callbacks = [
        keras.callbacks.ModelCheckpoint(f'model_checkpoints/{model_id}.h5', save_best_only=True)
    ]

    (img_paths, target_paths) = train_img_target_pairs
    assert len(img_paths) == len(target_paths)


    if val_img_target_pairs:
        print('Running without cross validation.')

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

        print(model_history.history.keys())

        losses = model_history.history['loss']
        validation_losses = model_history.history['val_loss']
    else:
        print('Running with cross validation.')
        num_examples_per_fold = len(img_paths) // cross_validation_folds

        for epoch in range(0, epochs, cross_validation_folds):
            for fold_num in range(cross_validation_folds):
                last_idx = (fold_num + 1) * num_examples_per_fold if fold_num < cross_validation_folds - 1 else len(img_paths)
                val_indices = [i for i in range(fold_num * num_examples_per_fold, last_idx)]
                train_x = [img_paths[i] for i in range(len(img_paths)) if i not in val_indices]
                train_y = [target_paths[i] for i in range(len(target_paths)) if i not in val_indices]
                val_x = [img_paths[i] for i in val_indices]
                val_y = [target_paths[i] for i in val_indices]

                assert len(train_x) == len(train_y)
                assert len(val_x) == len(val_y)

                print(f'Create train dataset with batch_size={batch_size}, img_size={img_size}, n={len(train_x)}')
                train_data = VagusDataLoader(batch_size, img_size, train_x, train_y)
                print(f'Create validation dataset with batch_size={batch_size}, img_size={img_size}, n={len(val_x)}')
                val_data = VagusDataLoader(batch_size, img_size, val_x, val_y)

                # Fit to current train and validation split
                model_history = model.fit(train_data, epochs=1, validation_data=val_data, callbacks=callbacks)

                # losses.append(model_history.history['loss'])
                # validation_losses.append(model_history.history['val_loss'])

    print('Finished training')

    losses_save_location = f'model_losses/{model_id}'

    np.save(losses_save_location, np.array(list(model_history.history.values())))

    return model


def output_predictions(trained_model=None, trained_model_checkpoint=None):
    if trained_model is None and trained_model_checkpoint is None:
        raise ValueError('Must supply either model or model checkpoint file to output predictions.')

    if trained_model_checkpoint is not None:
        trained_model = keras.models.load_model(trained_model_checkpoint)

    print('Generating predictions')

    one_prediction_iou(trained_model, test_img=np.load('data/vagus_dataset_6/images/img_1.npy'), test_anno=np.load('data/vagus_dataset_6/annotations/img_1.npy'))

    # val_imgs, val_annos = get_image_annotation_generators(subset='validation', image_directory='data/vagus_dataset_6/images', annotation_directory='data/vagus_dataset_6/annotations')
    # model_iou(trained_model, val_imgs, val_annos)

    # test_im = cv2.imread(val_input_img_paths[0])

    # for i in range(2):
    #     visualise_one_prediction(trained_model, val_imgs.next())

    # i1 = cv2.imread('data/vagus_dataset_3/images/Vago dx 21.02.19 DISTALE con elettrodo - vetrino 1 - fetta 0005.jpg')
    # visualise_one_prediction(trained_model, np.expand_dims(cv2.resize(i1, img_size), axis=0))

    # predictions = eval(trained_model, val_generator)
    # print(predictions)
    # print(predictions.shape)
    # display_predictions(val_input_img_paths, val_target_img_paths, predictions)
    print('Finished predictions')


if __name__ == '__main__':
    initialise_run()
    #m = train(model_id='Adam_SCC_512_default', train_img_target_pairs=input_target_path_pairs('data/training/520')) # With cross validation
    # Without cross validation
    m = train(
        model_id='test_custom_metric',
        train_img_target_pairs=input_target_path_pairs('data/vagus_dataset_10/train'), 
        val_img_target_pairs=input_target_path_pairs('data/vagus_dataset_10/validation')
    )
    # output_predictions(trained_model_checkpoint=model_save_file)
    # run_train_with_augmentation()
    print('Done')
