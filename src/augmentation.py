from keras_preprocessing.image import ImageDataGenerator

from src.config import seed, batch_size, img_size
from src.data_utils import annotation_preprocessor


def get_image_annotation_generators(subset='validation', image_directory='data/train/images', annotation_directory='data/train/annotations_old'):

    data_gen_arcs = dict(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant',
        cval=255,
        validation_split=0.2,
    )

    img_datagen = ImageDataGenerator(**data_gen_arcs)

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