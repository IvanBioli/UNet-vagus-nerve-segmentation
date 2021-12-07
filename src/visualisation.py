import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageOps, Image
from skimage import color
from tensorflow import keras

from src.config import initialise_run
from src.post_processing import identify_fasicle_regions


# from post_processing import identify_fasicle_regions


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

    ax1.imshow(model_input[i, :, :, :] / 255)
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


def plot_loss_graph(loss_file):
    losses = np.load(loss_file)
    plt.plot(losses[0], label='training loss')
    plt.plot(losses[1], label='validation loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    initialise_run()
    # display_annotation_high_contrast('data/vagus_dataset_2/annotations/Vago dx 21.02.19 DISTALE con elettrodo - vetrino 1 - fetta 0100.bmp')
    plot_loss_graph('model_losses/cv_aug_512.npy')
