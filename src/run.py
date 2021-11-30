import os

from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras

from config import img_size, num_classes, batch_size, epochs, seed, steps_per_epoch, validation_steps, val_samples, initialise_run
from data_loader import VagusDataLoader
from data_utils import input_target_path_pairs
from model import get_model
from src.augmentation import get_image_annotation_generators
from src.eval import model_iou
from src.visualisation import visualise_one_prediction


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
    val_data = VagusDataLoader(batch_size, img_size, val_input_img_paths, val_target_img_paths)

    print('Training model')
    trained_model = train(train_data, val_data, save_location=model_save_file)
    print('Training complete')
    return trained_model


def output_predictions(trained_model=None, trained_model_checkpoint=None):
    if trained_model is None and trained_model_checkpoint is None:
        raise ValueError('Must supply either model or model checkpoint file to output predictions.')

    if trained_model_checkpoint is not None:
        trained_model = keras.models.load_model(trained_model_checkpoint)

    print('Generating predictions')

    val_imgs, val_annos = get_image_annotation_generators(subset='validation')

    model_iou(trained_model, val_imgs, val_annos)

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
    model_save_file = 'model_checkpoints/model_checkpoint6.h5'
    # m = run_train_without_augmentation(dataset_folder='data/vagus_dataset_4', model_save_file=model_save_file)
    output_predictions(trained_model_checkpoint=model_save_file)
    print('Done')
