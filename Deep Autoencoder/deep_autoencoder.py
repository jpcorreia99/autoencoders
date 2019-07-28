import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys

sys.path.append("..")
import data_loading

x_train, x_test = data_loading.load_and_process_data()

# Neste encoder há mais do que uma hidden layer
try:
    model = keras.models.load_model("deep_autoencoder.h5")
except OSError as err:
    model = keras.models.Sequential([
        keras.layers.Dense(128, activation="relu", input_shape=(784,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(784, activation="relu"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        x_train,
        x_train,
        epochs=50,
        batch_size=256,
        validation_data=(x_test, x_test)
    )
    model.save("deep_autoencoder.h5")

predictions = model.predict(x_test)

for i in range(2):
    plt.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.title("real")
    plt.show()
    plt.imshow(predictions[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.title("predicted")
    plt.show()
