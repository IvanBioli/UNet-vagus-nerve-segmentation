import cv2
from tensorflow import keras
import numpy as np
from config import debug
from augmentation import get_random_affine_transformation

import matplotlib.pyplot as plt

from data_utils import is_annotation


class VagusDataLoader(keras.utils.Sequence):
    """
        Custom Data Loader class to iterate over the data (as Numpy arrays)
        Attributes
        ---------------
        batch_size: int
            the number of images would be loaded in a batch
        img_size: tuple (int, int)
            the size of the image being loaded
        input_img_paths: str
            system paths to the input (original) images
        target_img_paths: str
            system paths to the target (mask) images
    """
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        """ Class constructor """
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        """ Overriding len() """
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """
            Overriding get()
            Parameters
            ---------------
            idx: int
                The index of the current batch to get from the dataset
            Returns
            ---------------
            the idx-th batch of images in the tuple format of (input, target)
        """
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")

        for j, (img_path, target_path) in enumerate(zip(batch_input_img_paths, batch_target_img_paths)):
            img = np.load(img_path)
            annotation = np.load(target_path)
            annotation = np.expand_dims(annotation, axis=2)
            current_transform = get_random_affine_transformation()
            augmented_img = current_transform(img, do_colour_transform=False)
            augmented_annotation = current_transform(annotation, is_annotation=True, do_colour_transform=False)
            augmented_annotation = cv2.threshold(augmented_annotation, 0.5, 1, cv2.THRESH_BINARY)[1]
            augmented_annotation = np.expand_dims(augmented_annotation, axis=2)
            x[j] = augmented_img
            y[j] = augmented_annotation

        # Useful debug code
        if debug:
            print(f'Data loader first x, y pair - x shape: {x.shape}, x min max: {np.min(x)}, {np.max(x)}, y shape: {y.shape}, y values: {np.unique(y, return_counts=True)}')
            print('x')
            plt.imshow(x[0, :, :, :])
            plt.show()
            print('y')
            plt.imshow(y[0, :, :, 0])
            plt.show()

        # Final checks on tensor formats
        assert is_annotation(y), print(np.unique(y))
        assert np.max(x) <= 1 and np.min(x) >= 0, print(np.unique(x))
        assert x.shape[-1] == 3

        return x, y