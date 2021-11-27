import os

import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import img_size, num_classes, batch_size, epochs, seed, steps_per_epoch, validation_steps, val_samples
from data_loader import VagusDataLoader
from data_utils import annotation_preprocessor, input_target_path_pairs
from model import get_model
from src.visualisation import compare_augmented_image_annotations, visualise_one_prediction


def train(train_data: VagusDataLoader, val_data: VagusDataLoader):
    model = get_model(img_size, num_classes)

    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

    callbacks = [
        keras.callbacks.ModelCheckpoint("model_checkpoints/model_checkpoint1.h5", save_best_only=True)
    ]

    # Train the model, doing validation at the end of each epoch.
    model.fit_generator(train_data, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=val_data,
              validation_steps=validation_steps, callbacks=callbacks)
    # model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=callbacks)
    return model


def eval(model, test_data):
    test_predictions = model.predict(test_data)
    return test_predictions

if __name__ == '__main__':
    # devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(devices[0], True)

    os.chdir('/home/albion/code/EPFL/ml/nerve-segmentation')
    # os.chdir('D:/EPFL/ML/projects/nerve-segmentation/')
    # input_img_paths, target_img_paths = input_target_path_pairs('data/vagus_dataset_2')
    #
    # train_input_img_paths = input_img_paths[:-val_samples]
    # train_target_img_paths = target_img_paths[:-val_samples]
    # val_input_img_paths = input_img_paths[-val_samples:]
    # val_target_img_paths = target_img_paths[-val_samples:]
    #
    # # Instantiate data Sequences for each split
    # train_data = VagusDataLoader(batch_size, img_size, train_input_img_paths, train_target_img_paths)
    # val_data = VagusDataLoader(batch_size, img_size, val_input_img_paths, val_target_img_paths)
    #
    # trained_model = train(train_data, val_data)

    img_path = 'data/vagus_dataset_2/images'
    anno_path = 'data/vagus_dataset_2/annotations_old'
    # augmented_path = 'data/train/augmented'

    data_gen_arcs = dict(
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode='constant',
        cval=255,
        validation_split=0.2,
    )

    img_datagen = ImageDataGenerator(**data_gen_arcs)
    anno_datagen = ImageDataGenerator(**data_gen_arcs, preprocessing_function=annotation_preprocessor)

    img_dir_args = dict(
        directory=img_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=None,
        seed=seed,
    )

    anno_dir_args = dict(
        directory=anno_path,
        target_size=img_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode=None,
        seed=seed,
    )

    train_img_generator = img_datagen.flow_from_directory(subset='training', **img_dir_args)

    train_anno_generator = anno_datagen.flow_from_directory(subset='training', **anno_dir_args)

    val_img_generator = img_datagen.flow_from_directory(subset='validation', **img_dir_args)

    val_anno_generator = anno_datagen.flow_from_directory(subset='validation', **anno_dir_args)

    train_generator = zip(train_img_generator, train_anno_generator)
    val_generator = zip(val_img_generator, val_anno_generator)

    compare_augmented_image_annotations(val_img_generator, val_anno_generator)

    trained_model = train(train_generator, val_generator)

    print('Generating predictions')

    # test_im = cv2.imread(val_input_img_paths[0])

    visualise_one_prediction(trained_model, val_img_generator.next())

    # predictions = eval(trained_model, val_generator)

    # print(predictions)
    # print(predictions.shape)
    # display_predictions(val_input_img_paths, val_target_img_paths, predictions)

    print('Done')
