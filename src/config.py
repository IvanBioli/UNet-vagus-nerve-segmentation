import os


batch_size = 1
img_size = (512, 512)
num_classes = 2
#img_size = (160, 160)
num_classes = 2
#val_samples = 10
epochs = 100  # Can only be a multiple of 10 at the moment
fine_tune_epochs= 100

debug = False
debug_img_filepath = 'data/vagus_dataset_6/images/img_1.npy'



seed = 1
train_sample_size = 62
val_sample_size = 15
steps_per_epoch = train_sample_size / batch_size
validation_steps = val_sample_size / batch_size

model_path = 'model_checkpoints/Adam_SCC_512_default.h5'
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

