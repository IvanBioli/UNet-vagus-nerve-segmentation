import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops, label
from tensorflow import keras
import tensorflow_addons as tfa
from loss import iou_score, dice_loss, tversky_loss
from eval import predict_mask

from config import initialise_run, minimum_fascicle_area, watershed_coeff
from data_utils import get_samples
# custom = {'dice_loss': dice_loss, 'nerve_segmentation_loss': nerve_segmentation_loss, 'tversky_loss': tversky_loss}

def calculate_regions(pred, mask=None):
    """
        Calculate the regions (grouped predicted fascicles) from given prediction.
        Return the regions of the mask if it is provided in the parameter
        :param pred - the input prediction
        :param mask - the input mask
    """
    regions_pred = regionprops(label(((pred) * 255).astype(int)))
    if mask is not None:
        regions_mask = regionprops(label(((mask) * 255).astype(int)))
        return (regions_pred, regions_mask)
    else:
        return (regions_pred, None)

def compute_bins(n, pred, mask=None):
    """
        Compute the number of bins to display the distribution of the stas on prediction and mask
        Return the number of bins of the mask if it is provided in the parameter
        :param n - TODO
        :param pred - the input prediction
        :param mask - the input mask
    """
    if mask is None:
        minimum = min(pred)
        maximum = max(pred)
    else:
        minimum = min(min(pred), min(mask))
        maximum = min(max(pred), max(mask))

    binwidth = (maximum - minimum) / n
    bins = np.arange(minimum, maximum + binwidth, binwidth)
    return bins

# def calculate_metrics(mask, pred_logits, metrics=["iou"]):
#     stats = {}

#     if "iou" in metrics:
#         stats["iou"] = iou_score(mask, pred_logits).numpy()

#     if "bce" in metrics:
#         scce = keras.losses.SparseCategoricalCrossentropy()
#         stats["bce"] = scce(mask, pred_logits).numpy()

#     if "dice" in metrics:
#         stats["dice"] = dice_loss(mask, pred_logits).numpy()

#     if "focal" in metrics:
#         fce = tfa.losses.SigmoidFocalCrossEntropy()
#         stats["focal"] = fce(mask, pred_logits).numpy()

#     if "tversky" in metrics:
#         stats["tversky"] = tversky_loss(mask, pred_logits).numpy()

#     return stats


def compute_area_ratio(sample_train):
    """
        Computes the average ratio between the areas of the fascicles and of the background
        :param sample_train - the sample masks on which the ratio is calculated
    """
    tot_area_fascicles = 0
    n_samples = len(sample_train)
    for _, mask_path in sample_train:
        mask = np.load(mask_path)
        regions_mask = regionprops(label(((mask) * 255).astype(int)))
        for r in regions_mask:
            tot_area_fascicles = tot_area_fascicles + r.area
    shape = mask.shape
    avg_ratio = tot_area_fascicles / (n_samples * shape[0] * shape[1] - tot_area_fascicles)
    return avg_ratio

# compute color histograms of an image
def get_image_histogram(img):
    """
        Compute color histograms of an image
        :param img - the input image
    """
    hist = np.zeros([3, 256])
    img_ = np.floor(img * 255)
    
    for x in range(img_.shape[0]):
        for y in range(img_.shape[1]):
            r = int(img_[x, y, 0])
            g = int(img_[x, y, 1])
            b = int(img_[x, y, 2])
            
            hist[0, r] += 1
            hist[1, g] += 1
            hist[2, b] += 1
    # Percentage of pixels
    hist = hist / (img.shape[0] * img.shape[1]) * 100
    return hist


def get_dataset_histogram(path_list):
    """
        Compute color histograms of a images dataset
        :param path_list - the path to the input image dataset
    """
    set_hist = np.zeros([3, 256])
    
    for img_path in path_list:
        img = np.load(img_path)
        
        img_hist = get_image_histogram(img)
        set_hist = np.add(set_hist, img_hist)
        
    set_hist = set_hist / len(path_list)
    return set_hist

if __name__ == '__main__':
    initialise_run()
    model_save_file = os.path.join(os.getcwd(), 'model_checkpoints/Adam_SCC_512_default.h5')
    # Loading data from train folder
    train_folder = os.path.join(os.getcwd(), 'data/vagus_dataset_11/train')
    sample_train = get_samples(train_folder, num_samples=-1)
    # Loading data from unlabelled folder
    # unlabelled_folder = os.path.join(os.getcwd(), 'data/vagus_dataset/unlabelled')
    # sample_unlabelled = get_samples(unlabelled_folder, test=True, num_samples=-1)
    # Showing fascicles distribution
    #show_fascicles_distribution(sample_train, trained_model_checkpoint=model_save_file, save=True)
    # show_fascicles_distribution(sample_unlabelled, trained_model_checkpoint=model_save_file, save=True)
    print(compute_area_ratio(sample_train))