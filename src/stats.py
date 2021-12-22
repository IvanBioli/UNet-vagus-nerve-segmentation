import os

import numpy as np
from skimage.measure import regionprops, label

from config import initialise_run
from data_utils import get_samples

def calculate_regions(pred, mask=None):
    """
        Calculate the regions (grouped predicted fascicles) from given prediction.

        Parameters
        ---------------
        pred: np.ndarray
            the input prediction
        mask: np.ndarray, optional 
            the input mask

        Returns
        ---------------
        the regions of the mask if it is provided in the parameter
    """
    regions_pred = regionprops(label(((pred) * 255).astype(int)))
    if mask is not None:
        regions_mask = regionprops(label(((mask) * 255).astype(int)))
        return (regions_pred, regions_mask)
    else:
        return (regions_pred, None)

def compute_bins(n, pred, mask=None):
    """
        Compute the bins to display the distribution of the statistics on prediction and mask on the same histogram

        Parameters
        ---------------
        n: int
            number of bins
        pred: np.ndarray
            the input prediction statistics
        mask: np.ndarray, optional 
            the input mask statistics

        Returns
        ---------------
        the bins for displaying the statistics of the input prediction and the input mask on the same histogram
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

def compute_area_ratio(sample_train):
    """
        Computes the average ratio between the areas of the fascicles and of the background

        Parameters
        ---------------
        sample_train: [(str, str)]
            the sample images and masks on which the ratio is calculated

        Returns
        ---------------
        average ratio from the sample
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

        Parameters
        ---------------
        img: np.ndarray
            the input image

        Returns
        ---------------
        the r, g, b histogram of an image
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

        Parameters
        ---------------
        path_list: [str]
            the path to the input image dataset

        Returns
        ---------------
        the averaged r, g, b histogram of every images in a dataset
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
    train_folder = os.path.join(os.getcwd(), 'data/original_dataset/train')
    sample_train = get_samples(train_folder, num_samples=-1)
    print(compute_area_ratio(sample_train))