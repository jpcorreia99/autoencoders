import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

x_train = x_train / 255.
x_test = x_test / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # imagens com apenas um channel (preto e branco)
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# autoencoder que utiliza convoluções e pooling para salientar os aspetos chave das imagens

try:
    model = keras.models.load_model("convolutional_autoencoder.h5")
except OSError as err:
    model = keras.Sequential([
        # 28,28,1, #padding- "same": manter as dimensões originais
        keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same", input_shape=(x_test.shape[1:])),
        # dimensões agora (28,28,32), há 32 imagens diferentes pois aplicamos 32 filtros diferentes
        keras.layers.MaxPooling2D((2, 2), padding="same"),  # dimensão agora (14,14,8)

        keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
        keras.layers.MaxPooling2D((2, 2), padding="same"),  # dimensãoi agora (7,7,8)

        keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
        keras.layers.MaxPooling2D((2, 2), padding="same"),  # dimensão agora (4,4,8) #arredondou de 3,5 para 4

        # Reconstrução da imagem
        keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
        keras.layers.UpSampling2D((2, 2)),  # dimensãoa agora (8,8)

        keras.layers.Conv2D(8, (3, 3), activation="relu", padding="same"),
        keras.layers.UpSampling2D((2, 2)),

        keras.layers.Conv2D(16, (3, 3), activation="relu"),
        keras.layers.UpSampling2D((2, 2)),  # dimensões originais

        # agora juntamos todos os filtros num só, por isso e que o argumento é 1
        keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"),  # output layer #(28,28,1)
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
        batch_size=128,
        validation_data=(x_test, x_test)
    )

    model.save("convolutional_autoencoder.h5")

predictions = model.predict(x_test)

for i in range(2):
    plt.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.title("real")
    plt.show()
    plt.imshow(predictions[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.title("predicted")
    plt.show()
