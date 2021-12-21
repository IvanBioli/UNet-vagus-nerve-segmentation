import numpy as np
from keras_preprocessing.image import apply_affine_transform, random_channel_shift, random_brightness

from config import img_size


def get_random_transformation(white_background=True):
    """
        Return a transformation function object to randomly transform a given image

        Parameters
        ---------------
        white_background: bool, optional
            flag to set background of data augmentation to white or use an average of an image
        
        Returns
        ---------------
        a function that takes in an image and output a random transformation
    """

    affine_transform_args = dict(
        theta=np.random.randint(0, 359),
        tx=int(sample_rand_float() * img_size[1]),
        ty=int(sample_rand_float() * img_size[1]),
        shear=np.random.randint(0, 30),
        zx=np.random.uniform(0.9, 1.1),
        zy=np.random.uniform(0.9, 1.1),
    )

    def sample_rand_float(multiplier=0.2):
        """ Returns a random float """
        return (np.random.rand() - 0.5) * multiplier * 2

    # colour and brightness transformations do not affect the annotation
    def apply_colour_transform(img, brightness_range=(0.6, 1.4), intensity_range=0.2, is_annotation=False):
        """
            Apply colour transformation on a given image if it is not an annotation and return the transformed image

            Parameters
            ---------------
            img: np.ndarray
                the input image to be applied colour transformation on
            brightness_range: tuple (float, float), optional
                the range of brightness (compared the original one) to which the image will be transformed into
            intensity_range: float, optional
                the range of intensity to apply colour shift to the image
            is_annotation: boole, optional
                whether the image is an annotation 

            Returns
            ---------------
            the transformed image
        """
        if not is_annotation:
            img = random_brightness(img, brightness_range) / 255
            img = random_channel_shift(img, intensity_range)
        return img

    def transformation(img, is_annotation=False, do_colour_transform=True):
        """
            Apply transformation on a given image and return the transformed image

            Parameters
            ---------------
            img: np.ndarray
                the input image to be applied colour transformation on
            is_annotation: bool, optional
                whether the image is an annotation 
            do_colour_transform: bool, optional
                flag to apply colour and brightness transformations                

            Returns
            ---------------
            the transformed image
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
