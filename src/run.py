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

    def get_prediction(i):
        """Quick utility to display a model's prediction."""
        mask = np.argmax(test_predictions[i], axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
        return img

    i = 0

    # Display input image
    original_image = mpimg.imread(val_input_img_paths[i])
    original_image_plot = plt.imshow(original_image)
    plt.show()

    # Display ground-truth target mask
    # original_mask = mpimg.imread(val_target_img_paths[i])
    # original_mask_plot = plt.imshow(original_mask)
    # plt.show()
    original_mask = PIL.ImageOps.autocontrast(load_img(val_target_img_paths[i]))

    # Display mask predicted by our model
    prediction = get_prediction(i)  # Note that the model only sees inputs at 150x150.

    return original_image, original_mask, prediction


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

    eval(trained_model, val_data)
