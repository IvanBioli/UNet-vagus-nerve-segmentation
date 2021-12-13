import os

import matplotlib.pyplot as plt
import numpy as np
import random
from skimage.measure import regionprops, label
from tensorflow import keras
from loss import SparseMeanIoU, dice_loss, nerve_segmentation_loss, sparse_cce_dice_combination_loss, tversky_loss
from eval import predict_mask

from config import initialise_run
from post_processing import identify_fasicle_regions
custom = {'sparse_cce_dice_combination_loss': sparse_cce_dice_combination_loss, 'SparseMeanIoU': SparseMeanIoU, 'dice_loss': dice_loss, 'nerve_segmentation_loss': nerve_segmentation_loss, 'tversky_loss': tversky_loss}

from statistics import mean


def get_samples(data_folder, test=False, num_samples=1, shuffle=True):
    img_folder = data_folder + '/images'
    img_paths = [img_folder + '/' + img_file for img_file in os.listdir(img_folder)]
    if not test:
        mask_folder = data_folder + '/annotations'
        mask_paths = [mask_folder + '/' + mask_file for mask_file in os.listdir(mask_folder)]

        paths = list(zip(img_paths, mask_paths))
        if shuffle:
            random.shuffle(paths)
    else:
        paths = img_paths

    if num_samples == -1:
        num_samples = len(paths)
    return paths[:num_samples]


def calculate_regions(pred, mask=None):
    regions_pred = regionprops(label(((1 - pred) * 255).astype(int)))
    if mask is not None:
        regions_mask = regionprops(label(((1 - mask) * 255).astype(int)))
        return (regions_pred, regions_mask)
    else:
        return (regions_pred, None)

def compute_bins(n, pred, mask=None):
    if mask is None:
        minimum = min(pred)
        maximum = max(pred)
    else:
        minimum = min(min(pred), min(mask))
        maximum = min(max(pred), max(mask))

    binwidth = (maximum - minimum) / n
    bins = np.arange(minimum, maximum + binwidth, binwidth)
    return bins

def show_fascicles_distribution(paths, test=False, trained_model_checkpoint=None, save=False, show=True):

    if trained_model_checkpoint is not None:
        trained_model = keras.models.load_model(trained_model_checkpoint, custom_objects = custom)

    if save:
        output_folder = os.path.join(os.getcwd(), 'results/visualisations/distributions')
        os.makedirs(output_folder, exist_ok=True)
        if test:
            fname = 'Distribution for unlabelled test set'
        else:
            fname = 'Distribution for training set'
        out_fname = os.path.join(output_folder, fname)

    if not test:
        areas_mask = []
        num_fascicles_mask = []
        eccentricity_mask = []
    areas_pred = []
    num_fascicles_pred = []
    eccentricity_pred = []
    areas_post = []
    num_fascicles_post = []
    eccentricity_post = []

    for p in paths:
        if not test:
            img_path, mask_path = p
            mask = np.load(mask_path)
        else:
            img_path = p
            mask = None

        img = np.load(img_path)
        pred = predict_mask(trained_model, img, 0, [0])

        regions_pred, regions_mask = calculate_regions(pred, mask)
        
        pred_post = predict_mask(trained_model, img, threshold=101, coeff_list=[0.4])
        regions_post, _ = calculate_regions(pred_post)
            
        if not test:
            areas_mask = areas_mask + [m.area for m in regions_mask]
            eccentricity_mask = eccentricity_mask + [m.eccentricity for m in regions_mask]
            num_fascicles_mask.append(len(regions_mask))
        areas_pred = areas_pred + [p.area for p in regions_pred]
        eccentricity_pred = eccentricity_pred + [p.eccentricity for p in regions_pred]
        num_fascicles_pred.append(len(regions_pred))

        areas_post = areas_post + [p.area for p in regions_post]
        eccentricity_post = eccentricity_post + [p.eccentricity for p in regions_post]
        num_fascicles_post.append(len(regions_post))

    fig, axs = plt.subplots(2, 3, figsize=(20, 12))

    nbins_area = 50
    nbins_fascicles = 10
    nbins_eccentricity = 50
    if not test:
        # Computing the bins for the areas' histogram and plotting the histogram
        bins_areas = compute_bins(nbins_area, areas_pred, areas_mask)
        axs[0][0].hist(areas_mask, bins=bins_areas, alpha=0.5, label='Ground truth')
        # Computing the bins for the fascicles' histogram and plotting the histogram
        bins_fascicles = compute_bins(nbins_fascicles, num_fascicles_pred, num_fascicles_mask)
        axs[0][1].hist(num_fascicles_mask, bins=bins_fascicles, alpha=0.5, label='Ground truth')
        bins_eccentricity = compute_bins(nbins_eccentricity, eccentricity_pred, eccentricity_mask)
        axs[0][2].hist(eccentricity_mask, bins=bins_eccentricity, alpha=0.5, label='Ground truth')
    else:
        bins_areas = compute_bins(nbins_area, areas_pred)
        bins_fascicles = compute_bins(nbins_fascicles, num_fascicles_pred)
        nbins_eccentricity = compute_bins(nbins_eccentricity, eccentricity_pred)

    # Histogram of Fascicles Areas for the predictions
    axs[0][0].hist(areas_pred, bins=bins_areas, alpha=0.5, label='Prediction without postprocessing')
    axs[1][0].hist(areas_post, bins=bins_areas, alpha=0.5, label='Prediction')
    axs[1][0].hist(areas_pred, bins=bins_areas, alpha=0.5, label='Prediction without postprocessing')
    axs[1][0].set_xlabel('Areas (pixels)')
    axs[0][0].set_ylabel('Occurrencies')
    axs[1][0].set_ylabel('Occurrencies')
    axs[0][0].legend(loc='upper right')
    axs[1][0].legend(loc='upper right')
    axs[0][0].set_title('Histogram of Fascicles Areas')

    # Histogram of Number of fascicles Areas for the predictions
    axs[0][1].hist(num_fascicles_pred, bins=bins_fascicles, alpha=0.5, label='Prediction without postprocessing')
    axs[1][1].hist(num_fascicles_post, bins=bins_fascicles, alpha=0.5, label='Prediction')
    axs[1][1].hist(num_fascicles_pred, bins=bins_fascicles, alpha=0.5, label='Prediction without postprocessing')
    axs[1][1].set_xlabel('Number of Fascicles')
    axs[0][1].set_ylabel('Occurrencies')
    axs[1][1].set_ylabel('Occurrencies')
    axs[0][1].legend(loc='upper right')
    axs[1][1].legend(loc='upper right')
    axs[0][1].set_title('Histogram of Number of Fascicles')

    # Histogram of the Eccentricity
    axs[0][2].hist(eccentricity_pred, bins=bins_eccentricity, alpha=0.5, label='Prediction without postprocessing')
    axs[1][2].hist(eccentricity_post, bins=bins_eccentricity, alpha=0.5, label='Prediction')
    axs[1][2].hist(eccentricity_pred, bins=bins_eccentricity, alpha=0.5, label='Prediction without postprocessing')
    axs[1][2].set_xlabel('Eccentricity')
    axs[0][2].set_ylabel('Occurrencies')
    axs[1][2].set_ylabel('Occurrencies')
    axs[0][2].legend(loc='upper right')
    axs[1][2].legend(loc='upper right')
    axs[0][2].set_title('Histogram of Eccentricity')

    # Print the quantiles of the areas and the eccentricity for the masks:
    if not test:
        print('0.01-quantile of the masks'' area:', np.quantile(areas_mask, 0.01))
        print('0.99-quantile of the masks'' area:', np.quantile(areas_mask, 0.99))
        print('0.01-quantile of the masks'' eccentricity:', np.quantile(eccentricity_mask, 0.01))
        print('0.99-quantile of the masks'' eccentricity:', np.quantile(eccentricity_mask, 0.99))

    # If postprocessing is involved
    if save:
        plt.savefig(out_fname + '_predicted.jpg')
    if show:
        plt.show()


if __name__ == '__main__':
    initialise_run()
    model_save_file = os.path.join(os.getcwd(), 'model_checkpoints/Adam_SCC_512_default.h5')
    # Loading data from train folder
    train_folder = os.path.join(os.getcwd(), 'data/vagus_dataset_10/train')
    sample_train = get_samples(train_folder, num_samples=-1)
    # Loading data from unlabelled folder
    unlabelled_folder = os.path.join(os.getcwd(), 'data/520/unlabelled')
    sample_unlabelled = get_samples(unlabelled_folder, test=True, num_samples=-1)
    # Showing fascicles distribution
    show_fascicles_distribution(sample_train, trained_model_checkpoint=model_save_file, save=True)
    show_fascicles_distribution(sample_unlabelled, trained_model_checkpoint=model_save_file, save=True)