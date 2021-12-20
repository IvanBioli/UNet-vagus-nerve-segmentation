import os
import random
import numpy as np

batch_size = 1
img_size = (512, 512)
num_classes = 2
epochs = 100
fine_tune_epochs = 40
seed = 0

debug = False

cur_model_id = 'FL_and_BCE_Adam_default'
ft_model_id = f'{cur_model_id}_fine_tune'
model_path = f'model_checkpoints/{cur_model_id}.h5'
minimum_fascicle_area = 101
watershed_coeff = [0.35]


def initialise_run():
    """ Machine specific run initialisation """
    # devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(devices[0], True)
    # os.chdir('/home/albion/code/epfl/ml/nerve-segmentation')
    # os.chdir('D:/EPFL/ML/projects/nerve-segmentation/')
    os.chdir('C:/Users/ivanb/Documents/GitHub/ML Project 2')
    os.makedirs('model_checkpoints', exist_ok=True)
    os.makedirs('model_losses', exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
