import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import cv2
import matplotlib.pyplot as plt

def show_images():
    im = cv2.imread('data/train/images/sample/vago DX - 27.06.18 - HH - vetrino 1 - prox - campione 0002.jpg')
    plt.imshow(im)
    plt.show()
    im2 = cv2.resize(im, (256, 256))
    plt.imshow(im2)
    plt.show()
    im3 = cv2.resize(im, (160, 160))
    plt.imshow(im3)
    plt.show()
    im4 = cv2.resize(im, (512, 512))
    plt.imshow(im4)
    plt.show()
    # print(im.shape)


if __name__ == '__main__':

    os.chdir('/home/albion/code/EPFL/ml/nerve-segmentation')
    pet_mask = load_img('../oxf-pets/annotations/trimaps/Abyssinian_1.png')
    vagus_mask = load_img('data/vagus_dataset_2/annotations/Vago dx 21.02.19 DISTALE con elettrodo - vetrino 1 - fetta 0005.jpg')
    print('0')