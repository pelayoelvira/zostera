from keras import layers
import keras

def get_model(img_size):
    #image size =512 x 512 
    inputs = keras.Input(shape=img_size + (3,))

    ### [Primera mitad de la red: downsampling] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Bloques 1, 2, 3 con diferente profundidad de características.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Proyectar residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])  # Añadir residual
        previous_block_activation = x  # Preparar el siguiente residual

    ### [Segunda mitad de la red: upsampling] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Proyectar residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Añadir residual
        previous_block_activation = x  # Preparar el siguiente residual

    # Capa de clasificación por píxel
    outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    # Definir el modelo
    model = keras.Model(inputs, outputs)
    return model
