import os
import random

import numpy as np


def get_samples(data_folder, test=False, num_samples=1, shuffle=True):
    """
        Gets random samples of images
        :param data_folder - dataset directory
        :param test - Flag to return masks or not
        :param num_samples - Number of samples to return
        :param shuffle - Shuffles dataset samples
    """

    img_folder = data_folder + '/images'
    img_paths = [img_folder + '/' + img_file for img_file in os.listdir(img_folder) if img_file.endswith('.npy')]
    if not test:
        mask_folder = data_folder + '/annotations'
        mask_paths = [mask_folder + '/' + img_file for img_file in os.listdir(img_folder) if img_file.endswith('.npy')]

        paths = list(zip(img_paths, mask_paths))
    else:
        paths = img_paths

    if shuffle:
        random.shuffle(paths)

    if num_samples == -1:
        num_samples = len(paths)
    return paths[:num_samples]


def is_annotation(annotation):
    """
        Returns true if the image provided is an annotation in the correct format.
        :param annotation - annotation image
    """
    return (np.unique(annotation) == np.array([0, 1])).all() and annotation.dtype == np.dtype('uint8')


def input_target_path_pairs(directory, print_examples=False):
    """
        Create image target pairs to feed into data loaders.
        :param directory - dataset directory
        :param print_examples - Flag to print / number of examples to print
    """
    input_dir = os.path.join(os.getcwd(), directory, 'images')
    target_dir = os.path.join(os.getcwd(), directory, 'annotations')

    input_img_paths = []
    target_img_paths = []

    for directory, paths in [(input_dir, input_img_paths), (target_dir, target_img_paths)]:
        for fname in os.listdir(directory):
            fpath = os.path.join(directory, fname)
            if fname.endswith('.npy'):
                paths.append(fpath)
            else:
                raise ValueError(f'Invalid file format {fpath}.')
        paths.sort()

    assert len(input_img_paths) == len(target_img_paths)

    print("Number of samples:", len(input_img_paths))

    if print_examples:
        for input_path, target_path in zip(input_img_paths[:print_examples], target_img_paths[:print_examples]):
            print(input_path, "|", target_path)

    return input_img_paths, target_img_paths
