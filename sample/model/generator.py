import tensorflow as tf

from model import resnet, base_cnn


class Generator(base_cnn.BaseCNN):

    def __init__(self, k_size, initializer, use_bias, batch_norm, dropout_rate):
        super(Generator, self).__init__()

        """self.conv0 = self.CNN_layer_3D(num_filters=32, kernel_size=self.params["k_size"], stride=1, initializer=initializer, use_bias=self.params["use_bias"], upsampling=False,
                                       apply_batch_norm=False, dropout_rate=0., activation=tf.keras.layers.LeakyReLU())"""
        """self.encoder = []
        num_filters = [32, 64, 128, 256, 512, 512]
        for i in range(len(num_filters)):
            self.encoder.append(
                self.CNN_layer_3D(num_filters=num_filters[i], k_size=self.params["k_size"], stride=2, initializer=initializer, use_bias=self.params["use_bias"],
                                  upsampling=False, apply_batch_norm=num_filters[i] != num_filters[0], dropout_rate=0., activation=tf.keras.layers.LeakyReLU()))
            self.encoder.append(resnet_block.ResnetIdentityBlock(num_filters[i], self.params["k_size"], self.params["use_bias"], initializer))

        num_filters = list(reversed(num_filters[:-1]))
        self.decoder = []
        for i in range(len(num_filters)):
            dropout_rate = 0.5 if i < 2 else 0.
            self.decoder.append(self.CNN_layer_3D(num_filters=num_filters[i], k_size=self.params["k_size"], stride=2, initializer=initializer, use_bias=self.params["use_bias"],
                                                  upsampling=True, apply_batch_norm=True, dropout_rate=dropout_rate, activation=tf.keras.layers.LeakyReLU()))
            self.decoder.append(resnet_block.ResnetIdentityBlock(num_filters[i], self.params["k_size"], self.params["use_bias"], initializer))

        self.last = self.CNN_layer_3D(num_filters=32, k_size=self.params["k_size"], stride=2, initializer=initializer, use_bias=self.params["use_bias"], upsampling=True,
                                      apply_batch_norm=True, dropout_rate=0., activation=tf.keras.layers.LeakyReLU())

        self.final = self.CNN_layer_3D(num_filters=1, k_size=self.params["k_size"], stride=1, initializer=initializer, use_bias=self.params["use_bias"], upsampling=False,
                                       apply_batch_norm=False, dropout_rate=0., activation=None)"""

        # Encoder
        self.conv0 = self.CNN_layer_3D(32, k_size, 1, initializer, use_bias, activation=tf.keras.layers.ReLU())
        self.e1 = resnet.ResBlock(64, k_size, 2, initializer, use_bias, False, False, dropout_rate)
        self.e2 = resnet.ResBlock(128, k_size, 2, initializer, use_bias, False, False, dropout_rate)
        self.e3 = resnet.ResBlock(256, k_size, 2, initializer, use_bias, False, False, dropout_rate)

        # Bottle neck
        self.b1 = resnet.ResBlock(512, k_size, 1, initializer, use_bias, False, False, dropout_rate)
        self.b2 = resnet.ResBlock(512, k_size, 1, initializer, use_bias, False, False, dropout_rate)
        self.b3 = resnet.ResBlock(1024, k_size, 1, initializer, use_bias, False, False, dropout_rate)
        self.b4 = resnet.ResBlock(512, k_size, 1, initializer, use_bias, False, False, dropout_rate)
        self.b5 = resnet.ResBlock(512, k_size, 1, initializer, use_bias, False, False, dropout_rate)

        # Decoder
        self.d3 = resnet.ResBlock(256, k_size, 2, initializer, use_bias, True, False, dropout_rate)
        self.d2 = resnet.ResBlock(128, k_size, 2, initializer, use_bias, True, False, dropout_rate)
        self.d1 = resnet.ResBlock(64, k_size, 2, initializer, use_bias, True, False, dropout_rate)
        self.conv1 = self.CNN_layer_3D(1, k_size, 1, initializer, use_bias)

    def call(self, input_tensor, training=False):

        """x = input_tensor
        skips = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x, training=training)
            if i % 2 == 1:
                skips.append(x)

        skips = list(reversed(skips[:-1]))
        num_skips = 0
        for i in range(len(self.decoder)):
            x = self.decoder[i](x, training=training)

            if i % 2 == 1:
                x = tf.concat([x, skips[num_skips]], -1)
                num_skips += 1

        x = self.last(x, training=training)
        x = self.final(x, training=training)"""

        x = self.conv0(input_tensor, training=training)
        e1 = self.e1(x, training=training)
        e2 = self.e2(e1, training=training)
        e3 = self.e3(e2, training=training)

        b1 = self.b1(e3, training=training)
        b2 = self.b2(b1, training=training)
        b3 = self.b3(b2, training=training)
        b4 = self.b4(b3, training=training)
        b5 = self.b5(b4, training=training)

        d3 = tf.concat([e2, self.d3(b5, training=training)], -1)
        d2 = tf.concat([e1, self.d2(d3, training=training)], -1)
        d1 = self.d1(d2, training=training)
        x = self.conv1(d1, training=training)

        return x
