import tensorflow as tf
import numpy as np
import src.utilities as util, src.ResUNET as ResUNET

base_path = "/home/francesco/UQ/deepQSMGAN/"
data_path = "data/shapes_shape64_ex100_2019_08_20"

epochs = 2500
batch_size = 1
lr = 0.0001
beta1 = 0.5
l1_weight = 100.0
labelSmoothing = 0.9

input_shape = (64, 64, 64, 1)
train_data_filename = util.generate_file_list(file_path=base_path + data_path + "/train/", p_shape=input_shape)
train_input_fn = util.data_input_fn(train_data_filename, p_shape=input_shape, batch=batch_size, nepochs=epochs, shuffle=True)
X_tensor, Y_tensor, _ = train_input_fn()

G_output = ResUNET.getGenerator(X_tensor)

loss = tf.reduce_mean(tf.abs(Y_tensor - G_output))
print(loss)

varsOpt = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
for i in varsOpt:
    print(i)
optimizer = tf.train.AdamOptimizer(lr, beta1).minimize(loss, var_list=varsOpt)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    while True:
        try:
            _, val_loss = sess.run([optimizer, loss])
            print(val_loss)
        except tf.errors.OutOfRangeError:
            break