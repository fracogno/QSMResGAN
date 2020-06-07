import tensorflow as tf

from model import base_cnn


class Discriminator(base_cnn.BaseCNN):

    def __init__(self, k_size, initializer, use_bias, batch_norm, dropout_rate):
        super(Discriminator, self).__init__()

        self.net = tf.keras.Sequential([
            self.CNN_layer_3D(32, k_size, 2, initializer, use_bias, False, False, dropout_rate, tf.keras.layers.LeakyReLU()),
            self.CNN_layer_3D(64, k_size, 2, initializer, use_bias, False, False, dropout_rate, tf.keras.layers.LeakyReLU()),
            self.CNN_layer_3D(128, k_size, 2, initializer, use_bias, False, False, dropout_rate, tf.keras.layers.LeakyReLU()),
            self.CNN_layer_3D(256, k_size, 1, initializer, use_bias, False, False, dropout_rate, tf.keras.layers.LeakyReLU()),
            self.CNN_layer_3D(1, k_size, 1, initializer, use_bias)
        ])

    def call(self, x, y, training=False):
        return self.net(tf.concat([x, y], -1), training=training)
