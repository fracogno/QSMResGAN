import tensorflow as tf

from model import base_cnn


class Discriminator(base_cnn.BaseCNN):

    def __init__(self, params, initializer):
        super(Discriminator, self).__init__()
        self.params = params

        self.net = tf.keras.Sequential([
            self.CNN_layer_3D(32, self.params["k_size"], 2, initializer, self.params["use_bias"], False, False, self.params["dropout_rate"], tf.keras.layers.LeakyReLU()),
            self.CNN_layer_3D(64, self.params["k_size"], 2, initializer, self.params["use_bias"], False, True, self.params["dropout_rate"], tf.keras.layers.LeakyReLU()),
            self.CNN_layer_3D(128, self.params["k_size"], 2, initializer, self.params["use_bias"], False, True, self.params["dropout_rate"], tf.keras.layers.LeakyReLU()),
            self.CNN_layer_3D(256, self.params["k_size"], 1, initializer, self.params["use_bias"], False, True, self.params["dropout_rate"], tf.keras.layers.LeakyReLU()),
            self.CNN_layer_3D(1, self.params["k_size"], 1, initializer, self.params["use_bias"], False, False, False, None)
        ])

    def call(self, x, y, training=False):
        return self.net(tf.concat([x, y], -1), training=training)
