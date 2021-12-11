from tensorflow.keras import layers
from tensorflow import keras

# from src.config import batch_size
from config import batch_size


def get_model(img_size, num_classes):
    layer_no = 0
    decoder_layers = []
    encoder_layers = []
    
    decoder_layers.append(layer_no)
    
    inputs = keras.Input(shape=img_size + (3,))
    layer_no += 1

    # [First half of the network: downsampling inputs]

    # Entry block
    decoder_layers.append(layer_no)
    x = layers.Conv2D(batch_size, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    layer_no += 3

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        decoder_layers.append(layer_no)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        
        layer_no += 9
        previous_block_activation = x  # Set aside next residual

    # [Second half of the network: upsampling inputs]

    for filters in [256, 128, 64, 32]:
        encoder_layers.append(layer_no)
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        
        layer_no += 10
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    encoder_layers.append(layer_no)
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    layer_no += 1

    # Define the model
    model = keras.Model(inputs, outputs)
    return model, decoder_layers, encoder_layers