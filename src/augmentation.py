import matplotlib.pyplot as plt
import numpy as np
from keras_preprocessing.image import apply_affine_transform, random_channel_shift, random_brightness

from config import img_size, initialise_run, debug_img_filepath


def get_random_affine_transformation():
    def sample_rand_float(multiplier=0.2):
        return (np.random.rand() - 0.5) * multiplier * 2

    # colour and brightness transformations do not affect the annotation
    def apply_colour_transform(img, brightness_range=(0.6, 1.4), intensity_range=0.2, is_annotation=False):
        if not is_annotation:
            img = random_brightness(img, brightness_range) / 255
            img = random_channel_shift(img, intensity_range)
        return img

    affine_transform_args = dict(
        theta=np.random.randint(0, 40),
        tx=int(sample_rand_float() * img_size[1]),
        ty=int(sample_rand_float() * img_size[1]),
        shear=np.random.randint(0, 30),
        zx=np.random.uniform(0.7, 1),
        zy=np.random.uniform(0.7, 1),
    )

    def transformation(img, is_annotation=False):
        img = apply_affine_transform(img, fill_mode='constant', cval=1, **affine_transform_args)
        img = apply_colour_transform(img, is_annotation=is_annotation)
        return img

    return transformation


if __name__ == '__main__':
    initialise_run()
    test_img = np.load(debug_img_filepath)
    img_aug = random_channel_shift(test_img, 0.5)
    plt.imshow(test_img)
    plt.show()
    plt.imshow(img_aug)
    plt.show()
