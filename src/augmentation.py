import matplotlib.pyplot as plt
import numpy as np
from keras_preprocessing.image import apply_affine_transform, random_channel_shift, random_brightness

from config import img_size, initialise_run


def get_random_affine_transformation(white_background=True):
    """
        param: white_background - Flag to set background of data augmentation to white or use an average of an image
    """
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
        zx=np.random.uniform(0.9, 1.1),
        zy=np.random.uniform(0.9, 1.1),
    )

    def transformation(img, is_annotation=False, do_colour_transform=True):
        """

            param: do_colour_transform - Flag to apply colour and brightness transformations
        """
        if is_annotation:
            cur_cval = 0
        else:
            if white_background:
                cur_cval = 1
            else:
                cur_cval = np.average(img)
        img = apply_affine_transform(img, fill_mode='constant', cval=cur_cval, **affine_transform_args)
        if do_colour_transform:
            img = apply_colour_transform(img, is_annotation=is_annotation)
        return img

    return transformation
