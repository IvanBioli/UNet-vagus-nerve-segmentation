import os

import cv2
import numpy as np
from pdf2image import convert_from_path

from src.config import img_size, initialise_run


def convert_dataset(input_directory, output_size=(160, 160), verbose=False, unlabelled_dir=False):
    """
        Converts images and masks to numpy arrays
        Masks from pdf -> npy with 1 channel
        Images from tif -> npy with 3 channels

        :param input_directory string folder path of input
        :param output_size file size of final output images
        :param verbose flag for printing in function
        :param unlabelled_dir flag for whether directory contains both images and masks or just images
    """

    if unlabelled_dir:
        directories = [('image', input_directory)]
    else:
        directories = [
            ('mask', os.path.join(input_directory, 'annotations')),
            ('image', os.path.join(input_directory, 'images'))
        ]

    for img_type, cur_directory in directories:
        for idx, file_name in enumerate(os.listdir(cur_directory)):
            fpath = os.path.join(cur_directory, file_name)
            out_fpath = os.path.join(cur_directory, file_name.rsplit('.', 1)[0] + '.npy')
            if img_type == 'mask':
                # Convert masks
                if file_name.endswith('.pdf'):
                    if verbose:
                        print('Converting mask: ', idx + 1)
                    pages = convert_from_path(fpath)
                    assert len(pages) == 1
                    image = pages[0]
                    image = np.array(image)
                    image = cv2.resize(image, output_size)  # resize mask before thresholding
                    (_, image) = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
                    assert ((np.unique(image) == [0, 255]).all())
                else:
                    raise ValueError(f'Invalid mask file type: {file_name}')
            elif img_type == 'image':
                if verbose:
                    print('Converting image: ', idx + 1)
                if file_name.endswith('.pdf'):
                    pages = convert_from_path(fpath)
                    assert len(pages) == 1
                    image = pages[0]
                    image = np.array(image)
                elif file_name.endswith('tif'):
                    image = cv2.imread(fpath)
                else:
                    raise ValueError(f'Invalid image file type: {file_name}')
                image = cv2.resize(image, output_size)  # resize image
            else:
                raise ValueError(f'Unknown image type: {img_type}')

            # rescale all images and masks to be between [0, 1]
            # masks should be 0 or 1 because there are 2 classes for this dataset
            image = image / 255
            if img_type == 'mask':
                assert ((np.unique(image) == [0, 1]).all())

            np.save(out_fpath, image)
            os.remove(fpath)


def run_dataset_convert():
    """ Script converts original dataset folder to model ready format. Only run this function once. """
    print('Converting train data')
    convert_dataset('data/original_dataset/train', output_size=img_size, verbose=False)
    print('Converting validation data')
    convert_dataset('data/original_dataset/validation', output_size=img_size, verbose=False)
    print('Converting unlabelled data')
    convert_dataset('data/original_dataset/unlabelled', output_size=img_size, verbose=False, unlabelled_dir=True)


if __name__ == '__main__':
    initialise_run()
    run_dataset_convert()
