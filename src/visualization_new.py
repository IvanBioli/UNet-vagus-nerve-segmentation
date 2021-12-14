import os
import pickle
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tensorflow import keras
from loss import SparseMeanIoU, dice_loss, nerve_segmentation_loss, tversky_loss
from eval import predict_mask
from stats import get_samples

from config import initialise_run, model_path, minimum_fascicle_area, watershed_coeff
custom = {'SparseMeanIoU': SparseMeanIoU, 'dice_loss': dice_loss, 'nerve_segmentation_loss': nerve_segmentation_loss, 'tversky_loss': tversky_loss}

def show_masks_vs_prediction(img_path_list, mask_path_list, trained_model_checkpoint=None, save=False, show=True):
    if trained_model_checkpoint is not None:
        trained_model = keras.models.load_model(trained_model_checkpoint, custom_objects=custom)

    fig, axs = plt.subplots(len(img_path_list), 4, figsize=(12, 9))

    if save:
        output_folder = os.path.join(os.getcwd(), 'results/visualisations/prediction')
        os.makedirs(output_folder, exist_ok=True)
        out_fname = output_folder

    for k, img_path in enumerate(img_path_list):
        img = np.load(img_path)
        mask = np.load(mask_path_list[k])
        pred = predict_mask(trained_model, img, threshold=minimum_fascicle_area, coeff_list=watershed_coeff)

        if save:
            fname = img_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
            out_fname = os.path.join(out_fname, fname)

        axs[k, 0].imshow(img)
        axs[k, 1].imshow(mask, cmap='gray', interpolation='none')
        axs[k, 2].imshow(pred, cmap='gray', interpolation='none')
        axs[k, 2].imshow(pred, cmap='gray', interpolation='none')

        axs[k, 3].imshow(mask, cmap='gray', interpolation='none')
        axs[k, 3].imshow(1 - pred, cmap='viridis', alpha=0.5, interpolation='none')
        for i in range(4):
            axs[k, i].xaxis.set_major_locator(ticker.NullLocator())
            axs[k, i].yaxis.set_major_locator(ticker.NullLocator())

    axs[0, 0].set_title('Input image')
    axs[0, 1].set_title('Ground truth')
    axs[0, 2].set_title('Prediction')
    axs[0, 3].set_title('Prediction overlayed\n on ground truth')

    if save:
        plt.savefig(out_fname + '_predicted.jpg')
    if show:
        plt.show()


# To plot the model losses and metrics
def plot_model_losses_and_metrics(loss_filepath):
    with open(loss_filepath, 'rb') as loss_file:
        model_details = pickle.load(loss_file)
    print(model_details.keys())
    fig, axs = plt.subplots(4, figsize=(5,10))
    best = np.argmin(model_details['val_loss'])
    print('\nBest at epoch: ', best)
    # SparseCategoricalCrossentropy
    axs[0].plot(model_details['sparse_categorical_crossentropy'], label = 'Training set')
    axs[0].plot(model_details['val_sparse_categorical_crossentropy'], label = 'Test set')
    axs[0].set(xlabel="Epochs", ylabel="CCE")
    axs[0].set_ylim(bottom = 0)
    axs[0].legend()
    print('\nCCE:\ntraining = ', model_details['sparse_categorical_crossentropy'][best], '\ntest = ', model_details['val_sparse_categorical_crossentropy'][best])
    # IoU
    axs[1].plot(model_details['spare_mean_iou'], label = 'Training set')
    axs[1].plot(model_details['val_spare_mean_iou'], label = 'Test set')
    axs[1].set(xlabel="Epochs", ylabel="IoU")
    axs[1].set_ylim([0,1])
    axs[1].legend()
    print('\nIoU:\ntraining = ', model_details['spare_mean_iou'][best], '\ntest = ', model_details['val_spare_mean_iou'][best])
    # SparseCategoricalAccuracy
    axs[2].plot(model_details['sparse_categorical_accuracy'], label = 'Training set')
    axs[2].plot(model_details['val_sparse_categorical_accuracy'], label = 'Test set')
    axs[2].set(xlabel="Epochs", ylabel="Categorical  Accuracy")
    axs[2].set_ylim(top = 1)
    axs[2].legend()
    print('\nCategorical Accuracy:\ntraining = ', model_details['sparse_categorical_accuracy'][best], '\ntest = ', model_details['val_sparse_categorical_accuracy'][best])
    # DiceLoss
    axs[3].plot(model_details['dice_loss'], label = 'Training set')
    axs[3].plot(model_details['val_dice_loss'], label = 'Test set')
    axs[3].set(xlabel="Epochs", ylabel="Dice Loss")
    axs[3].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axs[3].legend()
    print('\nDice Loss:\ntraining = ', model_details['dice_loss'][best], '\ntest = ', model_details['val_dice_loss'][best])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    initialise_run()
    img_path = [
        'data/vagus_dataset_11/validation/images/vago DX  - 27.06.18 - HH - vetrino 1 - pezzo 3 - campione 0010.npy',
        # 'data/vagus_dataset_11/validation/images/vago SX - 27.06.18 - pezzo 4 - fetta 0035.npy',
        # 'data/vagus_dataset_11/validation/images/Vago dx 21.02.19 DISTALE con elettrodo - vetrino 1 - fetta 0165.npy',
        'data/transfer_learning_dataset/train/images/P10sx1 +500 vet15 4x.npy',
    ]
    mask_path = [
        'data/vagus_dataset_11/validation/annotations/vago DX  - 27.06.18 - HH - vetrino 1 - pezzo 3 - campione 0010.npy',
        # 'data/vagus_dataset_11/validation/annotations/vago SX - 27.06.18 - pezzo 4 - fetta 0035.npy',
        # 'data/vagus_dataset_11/validation/annotations/Vago dx 21.02.19 DISTALE con elettrodo - vetrino 1 - fetta 0165.npy',
        'data/transfer_learning_dataset/train/annotations/P10sx1 +500 vet15 4x.npy',
    ]
    model_save_file = os.path.join(os.getcwd(), model_path)
    show_masks_vs_prediction(img_path, mask_path, trained_model_checkpoint=model_save_file, save=False, show=True)
    #plot_model_losses_and_metrics('model_losses/Adam_512_SCCE_fine_tune.pkl')