batch_size = 1
img_size = (256, 256)
num_classes = 2  # TODO change num classes to 2
val_samples = 30
epochs = 5

debug = False

seed = 1
train_sample_size = 64
val_sample_size = 12
steps_per_epoch = train_sample_size / batch_size
validation_steps = val_sample_size / batch_size
