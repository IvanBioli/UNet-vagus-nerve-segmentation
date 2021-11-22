import os

import PIL
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

from config import img_size, num_classes, val_samples, batch_size, epochs, seed, steps_per_epoch
from data_loader import VagusDataLoader
from data_utils import input_target_path_pairs
from model import get_model

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from visualisation import display_predictions


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
    model.fit(train_data, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=val_data, callbacks=callbacks)

    return model


def eval(model, test_data):
    test_predictions = model.predict(test_data)
    return test_predictions


if __name__ == '__main__':

    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)

    os.chdir('D:/EPFL/ML/projects/nerve-segmentation/')
    # input_img_paths, target_img_paths = input_target_path_pairs('data/vagus_dataset_2')

    # train_input_img_paths = input_img_paths[:-val_samples]
    # train_target_img_paths = target_img_paths[:-val_samples]
    # val_input_img_paths = input_img_paths[-val_samples:]
    # val_target_img_paths = target_img_paths[-val_samples:]

    # # Instantiate data Sequences for each split
    # train_data = VagusDataLoader(batch_size, img_size, train_input_img_paths, train_target_img_paths)
    # val_data = VagusDataLoader(batch_size, img_size, val_input_img_paths, val_target_img_paths)

    # trained_model = train(train_data, val_data)

    img_path = 'data/vagus_dataset_2/images'
    anno_path = 'data/vagus_dataset_2/annotations'
    augmented_path = 'data/vagus_dataset_2/augmented'

    data_gen_arcs = dict(rotation_range=40,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='nearest',
                        validation_split=0.2)

    img_datagen = ImageDataGenerator(**data_gen_arcs)
    anno_datagen = ImageDataGenerator(**data_gen_arcs)

    train_img_generator = img_datagen.flow_from_directory(directory=img_path,
                                                          target_size=img_size,
                                                          batch_size=batch_size,
                                                          class_mode=None,
                                                          seed=seed,
                                                          save_to_dir=augmented_path,
                                                          save_prefix='augmented_train',
                                                          save_format='png',
                                                          subset='training')

    train_anno_generator = anno_datagen.flow_from_directory(directory=anno_path,
                                                            target_size=img_size,
                                                            batch_size=batch_size,
                                                            class_mode=None,
                                                            seed=seed,
                                                            save_to_dir=augmented_path,
                                                            save_prefix='augmented_anno',
                                                            save_format='png',
                                                            subset='training')

    val_img_generator = img_datagen.flow_from_directory(directory=img_path,
                                                        target_size=img_size,
                                                        batch_size=batch_size,
                                                        class_mode=None,
                                                        seed=seed,
                                                        subset='validation')

    val_anno_generator = anno_datagen.flow_from_directory(directory=anno_path,
                                                          target_size=img_size,
                                                          batch_size=batch_size,
                                                          class_mode=None,
                                                          seed=seed,
                                                          subset='validation')

    train_generator = zip(train_img_generator, train_anno_generator)
    val_generator = zip(val_img_generator, val_anno_generator)


    trained_model = train(train_generator, val_generator)

    predictions = eval(trained_model, val_data)

    display_predictions(val_input_img_paths, val_target_img_paths, predictions)