import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageOps, Image
from skimage import color
from tensorflow import keras

from post_processing import identify_fasicle_regions


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

    for row in range(0, rows, 10):
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


def show_original_image(model_input, i=0):
    plt.imshow(model_input[i, :, :, :] / 255)
    plt.show()


def show_original_annotation(annotation, i=0):
    plt.imshow(annotation[i, :, :, 0])
    plt.imsave('results/anno_true.png', annotation[i, :, :, 0])
    plt.show()


def show_predicted_annotation(annotation, i=0):
    plt.imshow(annotation[i, :, :])
    plt.imsave('results/anno_pred.png', annotation[i, :, :])
    plt.show()


def show_combined_result(model_input, y_true, y_pred, i=0, iou_score=None, save_file=None):
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))

    for i, (ax, val) in enumerate(zip(axs, [model_input[i, :, :, :] / 255, y_true[i, :, :, 0], y_pred[i, :, :]])):
        ax.imshow(val)
        num_regions = identify_fasicle_regions(val, return_num_regions=True)
        if i == 0:
            ax.set_title(f'Input image', fontsize=10)
        elif i == 1:
            ax.set_title(f'True annotation\nNum regions: {num_regions}', fontsize=10)
        else:
            ax.set_title(f'Prediction\nIOU: {str(iou_score)}, Num regions: {num_regions}', fontsize=10)
    fig.tight_layout()

    if save_file:
        plt.savefig(save_file)

    plt.show()

def show_overlay_result(model_input, y_true, y_pred, i=0, iou_score=None, save_file=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.imshow(model_input[i, :, :, :]/255)
    ax1.set_title(f'Input image', fontsize=10)

    ax2.imshow(y_true[i, :, :, 0], cmap='gray', interpolation='none')
    ax2.imshow(y_pred[i, :, :], cmap='viridis', alpha=0.5, interpolation='none')

    ax2.set_title(f'True annotation: White      Prediction: Yellow\nIOU: {str(iou_score)}', fontsize=10)
    
    if save_file:
        plt.savefig(save_file)

    plt.show()


def show_result_test(image_resized, mask_resized):
    fig, axs = plt.subplots(1, 3, figsize=(12, 12))
    axs[0].imshow(image_resized)
    axs[1].imshow(mask_resized)
    overlaying_image = color.label2rgb(mask_resized, image_resized, bg_label=0)
    axs[2].imshow(overlaying_image)
    plt.show()
    plt.pause(1)


def visualise_one_prediction(model, input_image):
    print('Input image shape: ', input_image.shape)
    # plt.imshow(input_image)
    show_original_image(input_image)

    prediction = model.predict(input_image)
    print('Model output shape: ', prediction.shape)
    prediction = np.argmax(prediction, axis=-1)
    prediction = prediction[0, :, :]
    print('Prediction shape: ', prediction.shape)
    plt.imshow(prediction)
    plt.show()


def compare_augmented_image_annotations(img_generator, anno_generator):
    for i in range(1):
        x = img_generator.next()
        plt.imshow(x[0, :, :, :] / 255)
        plt.show()
        y = anno_generator.next()
        plt.imshow(y[0, :, :, 0])
        plt.show()


def compare_dataloader_image_annotations(train_dataloader, val_dataloader):
    for i in range(1):
        x, y = train_dataloader[i]
        plt.imshow(x[0, :, :, :] / 255)
        plt.show()
        plt.imshow(y[0, :, :, 0])
        plt.show()

"""
TODO modify dimensions, transparent overlay, 
"""

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


## TODO: distribution of metrics