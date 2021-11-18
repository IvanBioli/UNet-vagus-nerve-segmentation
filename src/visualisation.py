import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageOps, Image
from tensorflow import keras


def display_annotation_high_contrast(fpath):
    pil_target_img = ImageOps.autocontrast(Image.fromarray(cv2.imread(fpath)))
    plt.imshow(pil_target_img)
    plt.show()


def display_predictions(input_image_paths, input_target_paths, predictions):
    """ Create a matplotlib figure showing predictions """

    rows = len(predictions)
    cols = 3

    axes = []
    fig = plt.figure()

    for row in range(0, rows, 3):
        axes.append(fig.add_subplot(rows, cols, row + 1))
        # input_image = PIL.Image.open(input_image_paths[row])
        plt.imshow(cv2.imread(input_image_paths[row]))

        axes.append(fig.add_subplot(rows, cols, row + 2))
        cv2_target_img = cv2.imread(input_target_paths[row])
        pil_target_img = ImageOps.autocontrast(Image.fromarray(cv2_target_img))
        plt.imshow(pil_target_img)

        axes.append(fig.add_subplot(rows, cols, row + 3))
        """Quick utility to display a model's prediction."""
        mask = np.argmax(predictions[row], axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        img = ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
        plt.imshow(img)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    display_annotation_high_contrast('data/vagus_dataset_2/annotations/Vago dx 21.02.19 DISTALE con elettrodo - vetrino 1 - fetta 0100.bmp')

"""
    def get_prediction(i):
        mask = np.argmax(test_predictions[i], axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
        return img

    i = 0

    # Display input image
    original_image = mpimg.imread(val_input_img_paths[i])
    original_image_plot = plt.imshow(original_image)

    # Display ground-truth target mask
    # original_mask = mpimg.imread(val_target_img_paths[i])
    # original_mask_plot = plt.imshow(original_mask)
    # plt.show()
    original_mask = PIL.ImageOps.autocontrast(load_img(val_target_img_paths[i]))

    # Display mask predicted by our model
    prediction = get_prediction(i)  # Note that the model only sees inputs at 150x150.

    plt.show()
"""
