import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf 
from tensorflow import keras
from keras import layers
from keras.datasets import cifar10

print(tf.config.list_physical_devices('CPU'))
physical_devices = tf.config.list_physical_devices('CPU')

(x_train, y_train), (x_test,y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_train.astype("float32") / 255.0

model = keras.Sequential(
    [
        #dimensions of the picture x, y and color RGB
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3, padding ='valid', activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10),
    ]
)

print(model.summary())


def my_Model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3)(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()
    x = layers.Conv2D(64, 5, paddings='same')(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(128, 3)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = my_Model()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=3e-4),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=2)

model.evaluate(x_test, y_test, batch_size=64, verbose=2)

