import numpy as np
import tensorflow as tf
from tensorflow import keras

#modulo responsável por carregar e processar os dados

def load_and_process_data():
    # Processamento dos dados
    data = keras.datasets.mnist #loading do dataset de dígitos escritos à mão
    (x_train, y_train), (x_test, y_test) = data.load_data()


    # normalização dos pixeis, em vez de estarem codificados entre 0 e 255, estão entre 0 e 1
    x_train = x_train / 255
    x_test = x_test / 255

    # o x_train é um conjunbto de 6000 imagens/arrays de 28x28 pixeis,
    # no entanto nós queremos dar como input à nossa rede neuronal um array de 784 pixeis,
    # logo o x_train passa a ser um conjunto de 6000 arrays de 784 pixeis
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    return x_train,x_test


def load_and_process_noisy_data():
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

    x_train = x_train / 255.
    x_test = x_test / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

    # corromper os dados, adicionando-lhes "noise"
    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0., 1.) #todos os valores <0 tornam-se 0 e todoso >1 tornam-se 1
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    return x_train_noisy,x_test_noisy, x_train, x_test