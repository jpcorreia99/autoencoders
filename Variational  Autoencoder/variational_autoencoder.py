import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras import backend as K  # permitir programar diretamente em tensorflow e tratar dos tensores


def sampling(args):
    """passamos os tensores para uma dimensão superior (o espaço latente)
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = keras.backend.shape(z_mean)[0]
    dim = keras.backend.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = keras.backend.random_normal(shape=(batch, dim))
    return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon


# processamento dos dados
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

original_dim = 28 * 28  # dimensões de cada imagem
x_train = np.reshape(x_train,
                     [-1, original_dim])  # estamos a transformar as matrizes das imagens num só array de (784,)
x_test = np.reshape(x_test, [-1, original_dim])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

batch_size = 128
latent_dim = 2
# modelo
# encoder - encarregado de codificar as imagens
# Neste caso não podemos usar o sequential porque vamos ter 2 tensores/valores/outputs do encoder e não só um
inputs = keras.layers.Input(shape=(784,), name='encoder_input')
x = keras.layers.Dense(512, activation='relu')(inputs)
z_mean = keras.layers.Dense(latent_dim, name='z_mean')(x)  # primeiro tensor
z_log_var = keras.layers.Dense(latent_dim, name='z_log_var')(x)  # segundo tensor
z = keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')(
    [z_mean, z_log_var])  # aplica a função sampling aos dois tensores/valores

encoder = keras.models.Model(inputs, [z_mean, z_log_var, z],
                             name='encoder')  # construção do modelo, input é as imagens e o output são os dois tensores numa dimensão latente

# descodificar os pontos latentes de novo para a dimensão original
# decoder
latent_inputs = keras.layers.Input(shape=(latent_dim,), name='z_sampling')
x = keras.layers.Dense(512, activation='relu')(latent_inputs)
outputs = keras.layers.Dense(784, activation='sigmoid')(x)

decoder = keras.models.Model(latent_inputs, outputs, name='decoder')

# inicializar o variational autoencoder
outputs = decoder(encoder(inputs)[2])
vae = keras.models.Model(inputs, outputs, name='vae_mlp')
print(vae.summary())


def vae_loss(x, x_decoded_mean):
    xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * keras.backend.mean(
        1 + z_log_sigma - keras.backend.square(z_mean) - keras.backend.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss


# compilação normal do modelo
vae.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])

# treino do modelo
vae.fit(x_train, x_train,
        shuffle=True,
        epochs=50,
        batch_size=128,
        validation_data=(x_test, x_test))

# ver as previsões como fazemos nos outros modelos

predictions = vae.predict(x_test)
for i in range(2):
    plt.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.title("real")
    plt.show()
    plt.imshow(predictions[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.title("predicted")
    plt.show()


# visualizar os clusters de dados, ou seja, os dados agrupados pelo seu tipo

z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)  # devolve os dois valores/tensores por cada dado
plt.figure(figsize=(12, 10))
plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
plt.colorbar()
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.show()

print(z_mean.shape)

# gerar novos dados
n = 15  # vamos criar uma folha com 15 x 15 dígitos
digit_size = 28  # tamanho de cada dígito
figure = np.zeros((digit_size * n, digit_size * n))  # a nossa folha que por enquanto s+o contem zeros

# linspace, cria um array com números devidamente espaçados entre -15 e 15, retorna 15 numeros
grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

for i, yi in enumerate(
        grid_x):  # o enumerate devolve-nos um contador, neste caso i, à medida que vamos iterando sobre grid_x
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(
            z_sample)  # dando-lhes os dois valores que o decoder daria como output, ele tenta trazê-los de novo à dimensão inicial
        digit = x_decoded[0].reshape(digit_size, digit_size)  # 28x28

        figure[i * digit_size: (i + 1) * digit_size,
        j * digit_size: (j + 1) * digit_size] = digit  # preencher um bocado da nossa folha com o dígito

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()





