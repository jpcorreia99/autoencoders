import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys

sys.path.append("..")
import data_loading

x_train_noisy, x_test_noisy, x_train, x_test = data_loading.load_and_process_noisy_data()

try:
    # verificar se já existe o modelo
    model = keras.models.load_model("image_denoising_autoencoder.h5")
except OSError as err:
    model = keras.Sequential([
        # shape= 28,28,1 - aind asó há uma imagem
        keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(x_train.shape[1:])),
        # agora há 32 imagens pois aplicamos 32 filtros fdiferentes
        keras.layers.MaxPooling2D((2, 2), padding="same"),  # (14,14)

        keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        keras.layers.MaxPooling2D((2, 2), padding="same"),  # (7,7)

        # (7,7,32) resolução atual  #32 versões da imagem pois estamos a aplicar 32 filtros por cada conv

        keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        keras.layers.UpSampling2D((2, 2)),

        keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        keras.layers.UpSampling2D((2, 2)),

        # juntamos todos os filtros num só
        keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        x_train_noisy,
        x_train,
        batch_size=128,
        epochs=50,
        validation_data=(x_test_noisy, x_test)
    )

    model.save("image_denoising_autoencoder.h5")

predictions = model.predict(x_test_noisy)
for i in range(2):
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.title("real")
    plt.show()
    plt.imshow(predictions[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.title("predicted")
    plt.show()
