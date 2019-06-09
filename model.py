import tensorflow as tf


def get_model(num_labels=27, input_shape=(45, 100, 176, 3), learning_rate=0.01):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape, dtype=tf.float32))
    model.add(tf.keras.layers.Conv3D(filters=32,
                                     kernel_size=(3, 5, 5),
                                     strides=(1, 2, 2),
                                     dilation_rate=(1, 1, 1),
                                     padding='same',
                                     data_format="channels_last",
                                     activation="relu"))
    model.add(tf.keras.layers.Conv3D(filters=32,
                                     kernel_size=(3, 3, 3),
                                     strides=(1, 1, 1),
                                     dilation_rate=(1, 1, 1),
                                     padding='same',
                                     activation="relu"))
                                     
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Conv3D(filters=64,
                                     kernel_size=(3, 3, 3),
                                     strides=(1, 1, 1),
                                     dilation_rate=(1, 1, 1),
                                     padding='same',
                                     activation="relu"))
    model.add(tf.keras.layers.Conv3D(filters=64,
                                     kernel_size=(3, 3, 3),
                                     strides=(1, 1, 1),
                                     dilation_rate=(1, 1, 1),
                                     padding='same',
                                     activation="relu"))
    model.add(tf.keras.layers.MaxPooling3D(pool_size=(3, 3, 3), padding="same"))
    model.add(tf.keras.layers.ConvLSTM2D(128, kernel_size=(3, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(num_labels, activation="softmax"))

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=learning_rate), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == "__main__":
    m = get_model(num_labels=27, input_shape=(45, 100, 176, 3), learning_rate=0.01)
    m.summary()
