import os

import PIL.Image
import cv2
import numpy as np
from pdf2image import convert_from_path
import imageio
from PIL import Image

def dataset_convert(input_folder, output_folder):
    """ Convert google drive vagus_dataset_1 into model ready format """
    input_folder = os.path.join(os.getcwd(), input_folder)
    output_folder = os.path.join(os.getcwd(), output_folder)
    for folder in ['images', 'annotations']:
        os.makedirs(os.path.join(output_folder, folder), exist_ok=True)
        directory = os.path.join(input_folder, folder)
        for fname in os.listdir(directory):
            fpath = os.path.join(directory, fname)
            out_fname = fname.rsplit('.', 1)[0] + '.jpg'
            if fname.endswith('.tif') or fname.endswith('.png') or fname.endswith('.jpg'):  #
                img = cv2.imread(fpath)
            elif fname.endswith('.pdf'):
                pages = convert_from_path(fpath)
                assert len(pages) == 1
                img = pages[0]
                img = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY)
            elif fname.endswith('.ai'):
                continue  # skip processing AI files
            else:
                raise ValueError(f'Invalid file format {fpath}')
            cv2.imwrite(os.path.join(output_folder, folder, out_fname), img, [int(cv2.IMWRITE_JPEG_QUALITY), 200])


def annotations_convert(folder):
    """ Convert annotations pngs to model ready format """
    input_directory = os.path.join(os.getcwd(), folder, 'annotations_old')
    directory = os.path.join(os.getcwd(), folder, 'annotations')
    for fname in os.listdir(input_directory):
        fpath = os.path.join(input_directory, fname)
        out_fname = fname.rsplit('.', 1)[0] + '.bmp'
        out_fpath = os.path.join(directory, out_fname)
        img_arr = np.asarray(cv2.imread(fpath), dtype=np.int8)
        converter = lambda x: 1 if x <= 100 else 0
        converted = np.vectorize(converter)(img_arr)
        if np.unique(converted).tolist() != [0, 1]:
            raise ValueError(f'Invalid conversion of annotation: {out_fpath}')
        # NOTE: do not use CV2 here, it modifies the values in the numpy array
        # imageio.imwrite(out_fpath, converted[:, :, 0])
        cv2.imwrite(out_fpath, converted, [cv2.IMWRITE_PNG_BILEVEL, 1])
        # PIL.Image.fromarray(converted).save(out_fpath, bits=1, optimize=True)



        # Check annotations written properly
        img_arr = np.asarray(cv2.imread(out_fpath))
        def checker(x):
            if not (x == 1 or x == 0):
                raise ValueError(f'Invalid annotation: {out_fpath}, {np.unique(img_arr)}')
            return x
        np.vectorize(checker)(img_arr)

def input_target_path_pairs(directory, print_examples=48):
    """ Create image target pairs """
    input_dir = os.path.join(os.getcwd(), directory, 'images')
    target_dir = os.path.join(os.getcwd(), directory, 'annotations')

    input_img_paths = []
    target_img_paths = []

    for directory, paths in [(input_dir, input_img_paths), (target_dir, target_img_paths)]:
        for fname in os.listdir(directory):
            fpath = os.path.join(directory, fname)
            if fname.endswith('.jpg') or fname.endswith('.bmp'):
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


if __name__ == '__main__':
    os.chdir('/home/albion/code/EPFL/ml/nerve-segmentation')
    # run the following lines only once
    # dataset_convert('data/vagus_dataset_2_ai', 'data/vagus_dataset_2')
    annotations_convert('data/vagus_dataset_2')
