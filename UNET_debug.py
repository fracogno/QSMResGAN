import os
import tensorflow as tf
import numpy as np
from datetime import datetime
import src.ResUNet as network, src.utilities as util

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

base_path = "/scratch/cai/deepQSMGAN/"
data_path = "data/shapes_shape64_ex100_2019_08_30"
#base_path = "/home/francesco/UQ/deepQSMGAN/"
#data_path = "data/shapes_shape64_ex100_2019_08_20"
checkpointName = base_path + "ckp_" + datetime.now().strftime("%Y%m%d_%H%M") + "_" + data_path.split("/")[-1]

input_shape = (64, 64, 64, 1)
train_data_filename = util.generate_file_list(file_path=base_path + data_path + "/train/", p_shape=input_shape)
train_input_fn = util.data_input_fn(train_data_filename, p_shape=input_shape, batch=1, nepochs=3000, shuffle=True)
X_tensor, Y_tensor, _ = train_input_fn()

Y_generated = network.getGenerator(X_tensor)    
loss = tf.reduce_mean(tf.abs(Y_tensor - Y_generated)) 
optimizer = tf.train.AdamOptimizer(0.0001, 0.5).minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

train_summaries = [tf.summary.scalar('loss', loss)]
train_merged_summaries = tf.summary.merge(train_summaries)

for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'):
    print(i)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    train_summary_writer = tf.summary.FileWriter(checkpointName + '/train', graph=tf.get_default_graph())
    global_step = 0
    while True:
        try:
            _, summary = sess.run([optimizer, train_merged_summaries])
            train_summary_writer.add_summary(summary, global_step)
            global_step += 1 
        except tf.errors.OutOfRangeError:
            break