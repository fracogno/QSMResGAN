import tensorflow as tf
import numpy as np
import datetime
import os

import Pix2Pix, utilities as util
from tensorflow.python.client import device_lib 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used
print(device_lib.list_local_devices())

# Paths
checkpoint_path = "/home/francesco/UQ/Job/QSM-GAN/checkpoints_2019-5-7_1223_shapes_shape64_ex512_2019_05_01/model.ckpt"
data_path = "/home/francesco/UQ/Job/QSM-GAN/data/shapes_shape64_ex512_2019_05_01/"



tf.reset_default_graph()
input_shape = (64, 64, 64, 1)

train_data_filename = util.generate_file_list(file_path=data_path + "/train/", p_shape=input_shape)
eval_data_filename = util.generate_file_list(file_path=data_path + "/eval/", p_shape=input_shape)

train_input_fn = util.data_input_fn(train_data_filename, p_shape=input_shape, batch=1, nepochs=1, shuffle=True)
eval_input_fn = util.data_input_fn(eval_data_filename, p_shape=input_shape, batch=64, nepochs=1, shuffle=True)

# Unpack tensors
train_data = train_input_fn()
val_data = eval_input_fn()

# Construct a suitable iterator (and ops) for switching between the datasets
iterator = tf.data.Iterator.from_structure(train_data[2].output_types, train_data[2].output_shapes)
X_tensor, Y_tensor = iterator.get_next()

training_init_op = iterator.make_initializer(train_data[2])
validation_init_op = iterator.make_initializer(val_data[2])


# Create networks
with tf.variable_scope("generator"):
    Y_generated = Pix2Pix.getGenerator(X_tensor["x"])

with tf.name_scope("real_discriminator"):
    with tf.variable_scope("discriminator"):
        D_logits_real = Pix2Pix.getDiscriminator(X_tensor["x"], Y_tensor)
      
with tf.name_scope("fake_discriminator"):
    with tf.variable_scope("discriminator", reuse=True):
        D_logits_fake = Pix2Pix.getDiscriminator(X_tensor["x"], Y_generated)

accuracy = tf.reduce_mean(tf.abs(Y_generated - Y_tensor))

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, checkpoint_path) 

    sess.run(validation_init_op)
    total_val_L1 = []
    while True:
        try:
            
            summary, val_L1 = sess.run([merged_summary_op, accuracy])
            break
        except tf.errors.OutOfRangeError:
            break
