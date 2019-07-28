import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys

sys.path.append("..")
import data_loading

x_train, x_test = data_loading.load_and_process_data()

# neste autoencoder limitamos o número de neurónios que podem estar ativos a qualquer momento, prevenindo overfitting
try:
    model = keras.models.load_model("simple_autoencoder.h5")
except OSError as err:  # caso não seja encontrado um modelo existente

    model = keras.models.Sequential([
        # o input original é uma imagem 28x28 que foi flattened (784 pixeis)
        keras.layers.Dense(32, activation="relu", input_shape=(784,),
                           activity_regularizer=keras.regularizers.l1(10e-5)),
        # os nossos 784 pixeis vão ficar comprimidos a 32. Este é o encoder
        keras.layers.Dense(784, activation="sigmoid"),
        # output layer tem novamente as dimensões da imagem original (o array de 784 pixeis). Este é o decoder

    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(x_train,
              x_train,
              epochs=100,
              batch_size=256,
              validation_data=(x_test, x_test),
              shuffle=True)

    model.save("sparse_autoencoder.h5")

# Utilização do modelo para tentar recriar as imagens
predictions = model.predict(x_test)

for i in range(2):
    plt.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.title("real")
    plt.show()
    plt.imshow(predictions[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.title("predicted")
    plt.show()
