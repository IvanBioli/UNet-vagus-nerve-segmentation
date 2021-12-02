import os

import cv2
import numpy as np
from keras_preprocessing.image import ImageDataGenerator, apply_affine_transform

from config import seed, batch_size, img_size, initialise_run
from data_utils import annotation_preprocessor, image_preprocessor


def get_random_affine_transformation():
    transform = lambda img: img
    # transform = lambda img: apply_affine_transform(img, theta=np.random.randint(0, 40), fill_mode='constant', cval=1)
    return transform


def get_image_annotation_generators(subset='validation', image_directory='data/train/images', annotation_directory='data/train/annotations', validation_split=0.2):
    data_gen_arcs = dict(
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.5, 1.3],
        fill_mode='constant',
        cval=255,
        validation_split=validation_split,
    )

    img_datagen = ImageDataGenerator(**data_gen_arcs, preprocessing_function=image_preprocessor)

    anno_datagen = ImageDataGenerator(**data_gen_arcs, preprocessing_function=annotation_preprocessor)

    img_dir_args = dict(
        directory=image_directory,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=None,
        seed=seed,
    )
    anno_dir_args = dict(
        directory=annotation_directory,
        target_size=img_size,
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode=None,
        seed=seed,
    )

    img_generator = img_datagen.flow_from_directory(subset=subset, **img_dir_args)
    anno_generator = anno_datagen.flow_from_directory(subset=subset, **anno_dir_args)

    return img_generator, anno_generator


def create_augmented_dataset(img_generator, anno_generator, folder, n=300):
    os.makedirs(os.path.join(folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(folder, 'annotations'), exist_ok=True)
    for i in range(n):
        x = img_generator.next()
        cv2.imwrite(os.path.join(folder, 'images', f'image_{i}.jpg'), x[0, :, :, :])
        y = anno_generator.next()
        # Anno generator outputs arrays between 0 and 1 (where 1 represents the fasicle)
        # Convert this to arrays between 0 and 255 (where 0 represents the fasicle)
        y = (1 - y[0, :, :, :]) * 255
        cv2.imwrite(os.path.join(folder, 'annotations', f'anno_{i}.jpg'), y)


if __name__ == '__main__':
    initialise_run()
    # 0 validation images here
    train_x, train_y = get_image_annotation_generators(subset='training', image_directory='data/train/images', annotation_directory='data/train/annotations', validation_split=0)
    test_x, test_y = get_image_annotation_generators(subset='training', image_directory='data/train/images', annotation_directory='data/train/annotations', validation_split=0)

    create_augmented_dataset(train_x, train_y, 'data/vagus_dataset_5/train', n=300)
    create_augmented_dataset(test_x, test_y, 'data/vagus_dataset_5/test', n=20)

"""

    # img_path = 'data/train/images'
    # anno_path = 'data/train/annotations_old'
    # # augmented_path = 'data/train/augmented'

    # train_generator = zip(train_img_generator, train_anno_generator)
    # val_generator = zip(val_img_generator, val_anno_generator)
    # create_augmented_dataset(train_img_generator, train_anno_generator, folder='data/vagus_dataset_4')

    # compare_augmented_image_annotations(val_img_generator, val_anno_generator)

    # compare_dataloader_image_annotations(train_data, val_data)

    # trained_model = train(train_generator, val_generator)


"""
