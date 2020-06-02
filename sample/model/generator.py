import tensorflow as tf

from model import resnet, base_cnn


class Generator(base_cnn.BaseCNN):

    def __init__(self, params, initializer):
        super(Generator, self).__init__()
        self.params = params
        self.initializer = initializer

        # Encoder
        self.conv0 = self.CNN_layer_3D(32, self.params["k_size"], 1, self.initializer, self.params["use_bias"], False, False, 0., False)
        self.e1 = resnet.ResBlock(64, self.params["k_size"], 2, self.initializer, self.params["use_bias"], False, self.params["use_batch_norm"], self.params["dropout_rate"])
        self.e2 = resnet.ResBlock(128, self.params["k_size"], 2, self.initializer, self.params["use_bias"], False, self.params["use_batch_norm"], self.params["dropout_rate"])
        self.e3 = resnet.ResBlock(256, self.params["k_size"], 2, self.initializer, self.params["use_bias"], False, self.params["use_batch_norm"], self.params["dropout_rate"])

        # Bottle neck
        self.b1 = resnet.ResBlock(512, self.params["k_size"], 1, self.initializer, self.params["use_bias"], False, self.params["use_batch_norm"], self.params["dropout_rate"])
        self.b2 = resnet.ResBlock(512, self.params["k_size"], 1, self.initializer, self.params["use_bias"], False, self.params["use_batch_norm"], self.params["dropout_rate"])
        self.b3 = resnet.ResBlock(1024, self.params["k_size"], 1, self.initializer, self.params["use_bias"], False, self.params["use_batch_norm"], self.params["dropout_rate"])
        self.b4 = resnet.ResBlock(512, self.params["k_size"], 1, self.initializer, self.params["use_bias"], False, self.params["use_batch_norm"], self.params["dropout_rate"])
        self.b5 = resnet.ResBlock(512, self.params["k_size"], 1, self.initializer, self.params["use_bias"], False, self.params["use_batch_norm"], self.params["dropout_rate"])

        # Decoder
        self.d3 = resnet.ResBlock(256, self.params["k_size"], 2, self.initializer, self.params["use_bias"], True, self.params["use_batch_norm"], self.params["dropout_rate"])
        self.d2 = resnet.ResBlock(128, self.params["k_size"], 2, self.initializer, self.params["use_bias"], True, self.params["use_batch_norm"], self.params["dropout_rate"])
        self.d1 = resnet.ResBlock(64, self.params["k_size"], 2, self.initializer, self.params["use_bias"], True, self.params["use_batch_norm"], self.params["dropout_rate"])
        self.conv1 = self.CNN_layer_3D(1, self.params["k_size"], 1, self.initializer, self.params["use_bias"], True, False, 0., False)

    def call(self, input_tensor, training=False):
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

        return tf.nn.tanh(x)
