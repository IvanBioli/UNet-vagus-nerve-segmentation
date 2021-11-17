import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img


if __name__ == '__main__':

    os.chdir('/home/albion/code/EPFL/ml/nerve-segmentation')
    pet_mask = load_img('../oxf-pets/annotations/trimaps/Abyssinian_1.png')
    vagus_mask = load_img('data/vagus_dataset_2/annotations/Vago dx 21.02.19 DISTALE con elettrodo - vetrino 1 - fetta 0005.jpg')
    print('0')