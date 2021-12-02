import numpy as np
from tensorflow import keras

from augmentation import get_image_annotation_generators
from config import img_size, num_classes, batch_size, epochs, steps_per_epoch, validation_steps, val_samples, initialise_run
from data_loader import VagusDataLoader
from data_utils import input_target_path_pairs
from eval import model_iou, one_prediction_iou
from model import get_model


def train(train_data, val_data, save_location):
    model = get_model(img_size, num_classes)

    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

    callbacks = [
        keras.callbacks.ModelCheckpoint(save_location, save_best_only=True)
    ]

    # Train the model, doing validation at the end of each epoch.
    if type(train_data) == VagusDataLoader:
        model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=callbacks)
    else:
        model.fit_generator(
            train_data, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=val_data,
            validation_steps=validation_steps, callbacks=callbacks
        )

    return model


def eval(model, test_data):
    test_predictions = model.predict(test_data)
    return test_predictions


def run_train_without_augmentation(dataset_folder, model_save_file='model_checkpoints/model_checkpoint0.h5'):
    input_img_paths, target_img_paths = input_target_path_pairs(dataset_folder)

    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    # Instantiate data Sequences for each split
    print(f'Create train dataset with batch_size={batch_size}, img_size={img_size}, n={len(train_input_img_paths)}')
    train_data = VagusDataLoader(batch_size, img_size, train_input_img_paths, train_target_img_paths)
    print(f'Create validation dataset with batch_size={batch_size}, img_size={img_size}, n={len(train_input_img_paths)}')
    print('Validation images: ', val_input_img_paths)
    val_data = VagusDataLoader(batch_size, img_size, val_input_img_paths, val_target_img_paths)

    print('Training model')
    trained_model = train(train_data, val_data, save_location=model_save_file)
    print('Training complete')
    return trained_model


def run_train_with_augmentation():
    train_x, train_y = get_image_annotation_generators(subset='training')
    val_x, val_y = get_image_annotation_generators(subset='validation')
    train_gen = zip(train_x, train_y)
    val_gen = zip(val_x, val_y)

    train(train_gen, val_gen, save_location='model_checkpoints/model_checkpoint7.h5')

    # x = train_x.next()
    # y = train_y.next()
    # print(x.shape, np.unique(x))
    # print(y.shape, np.unique(y))
    # import matplotlib.pyplot as plt
    # plt.imshow(x[0, :, :, :])
    # plt.show()


def output_predictions(trained_model=None, trained_model_checkpoint=None):
    if trained_model is None and trained_model_checkpoint is None:
        raise ValueError('Must supply either model or model checkpoint file to output predictions.')

    if trained_model_checkpoint is not None:
        trained_model = keras.models.load_model(trained_model_checkpoint)

    print('Generating predictions')

    one_prediction_iou(trained_model, test_img=np.load('data/vagus_dataset_7/images/1.npy'), test_anno=np.load('data/vagus_dataset_7/annotations/1.npy'))

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
    model_save_file = 'model_checkpoints/model_checkpoint11.h5'
    m = run_train_without_augmentation(dataset_folder='data/vagus_dataset_7', model_save_file=model_save_file)
    output_predictions(trained_model_checkpoint=model_save_file)
    # run_train_with_augmentation()
    print('Done')
