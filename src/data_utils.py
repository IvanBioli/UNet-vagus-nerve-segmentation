import os

import numpy as np


def is_annotation(annotation):
    """ Returns true if the image provided is an annotation in the correct format. """
    return (np.unique(annotation) == np.array([0, 1])).all() and annotation.dtype == np.dtype('uint8')


def input_target_path_pairs(directory, print_examples=48):
    """ Create image target pairs to feed into data loaders. """
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
