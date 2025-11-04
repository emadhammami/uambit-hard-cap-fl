
# src/uambit_cap/model_data.py

import numpy as np
import tensorflow as tf
from medmnist import INFO, BreastMNIST


def build_lenet5(input_shape=(28, 28, 1), num_classes=2):
    """
    LeNet-5 with explicit AveragePooling2D(pool_size=(2,2)) to match the notebook.
    Compiles with Adam(1e-3), categorical_crossentropy, and accuracy metric.
    """
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(6, (5, 5), activation="relu", padding="same")(inputs)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)  # fixed pool_size
    x = layers.Conv2D(16, (5, 5), activation="relu")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)  # fixed pool_size
    x = layers.Flatten()(x)
    x = layers.Dense(120, activation="relu")(x)
    x = layers.Dense(84, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_breastmnist():
    """
    Loads BreastMNIST train/val/test with medmnist, normalizes to [0,1],
    expands channel dim, and one-hot encodes labels using INFO metadata.
    Returns: (x_train,y_train),(x_val,y_val),(x_test,y_test), n_classes
    """
    info = INFO["breastmnist"]
    n_classes = len(info["label"])

    train_ds = BreastMNIST(split="train", download=True)
    val_ds = BreastMNIST(split="val", download=True)
    test_ds = BreastMNIST(split="test", download=True)

    x_train = train_ds.imgs.astype("float32") / 255.0
    x_val = val_ds.imgs.astype("float32") / 255.0
    x_test = test_ds.imgs.astype("float32") / 255.0

    # Add channel dimension
    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # One-hot labels
    y_train = tf.keras.utils.to_categorical(train_ds.labels.squeeze(), n_classes)
    y_val = tf.keras.utils.to_categorical(val_ds.labels.squeeze(), n_classes)
    y_test = tf.keras.utils.to_categorical(test_ds.labels.squeeze(), n_classes)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), n_classes
