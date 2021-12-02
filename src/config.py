import os

batch_size = 1
img_size = (160, 160)
num_classes = 2  # TODO change num classes to 2
val_samples = 10
epochs = 40

debug = True

seed = 1
train_sample_size = 62
val_sample_size = 15
steps_per_epoch = train_sample_size / batch_size
validation_steps = val_sample_size / batch_size


def initialise_run():
    """ Machine specific run initialisation """
    # devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(devices[0], True)
    os.chdir('/home/albion/code/EPFL/ml/nerve-segmentation')
    # os.chdir('D:/EPFL/ML/projects/nerve-segmentation/')