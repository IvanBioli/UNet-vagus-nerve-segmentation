import os

import PIL
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

from config import img_size, num_classes, val_samples, batch_size, epochs
from data_loader import VagusDataLoader
from data_utils import input_target_path_pairs
from model import get_model

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from src.visualisation import display_predictions


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
    model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=callbacks)

    return model


def eval(model, test_data):
    test_predictions = model.predict(test_data)
    return test_predictions


if __name__ == '__main__':
    os.chdir('/home/albion/code/EPFL/ml/nerve-segmentation')
    input_img_paths, target_img_paths = input_target_path_pairs('data/vagus_dataset_2')

    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    # Instantiate data Sequences for each split
    train_data = VagusDataLoader(batch_size, img_size, train_input_img_paths, train_target_img_paths)
    val_data = VagusDataLoader(batch_size, img_size, val_input_img_paths, val_target_img_paths)

    trained_model = train(train_data, val_data)

    predictions = eval(trained_model, val_data)

    display_predictions(val_input_img_paths, val_target_img_paths, predictions)