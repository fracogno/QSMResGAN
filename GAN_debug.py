import os
import tensorflow as tf
import numpy as np
from datetime import datetime
import src.ResUNet as network, src.utilities as util, src.loss as loss

# Set GPU ??
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Paths
base_path = "/scratch/cai/deepQSMGAN/"
data_path = "data/shapes_shape64_ex100_2019_08_30"

#base_path ="/home/francesco/UQ/deepQSMGAN/"
#data_path = "data/shapes_shape64_ex100_2019_08_20"
checkpointsPath = base_path + "ckp_" + datetime.now().strftime("%Y%m%d_%H%M") + "_" + data_path.split("/")[-1]

# Parameters
epochs = 3000
batchSize = 1
lr = 1e-4
beta1 = 0.5
L1_Weight = 100.0
labelSmoothing = 0.9

# Training data
input_shape = (64, 64, 64, 1)
train_data_filename = util.generate_file_list(file_path=base_path + data_path + "/train/", p_shape=input_shape)
train_input_fn = util.data_input_fn(train_data_filename, p_shape=input_shape, batch=batchSize, nepochs=epochs, shuffle=True)
X_tensor, Y_tensor, _ = train_input_fn()

# Networks
Y_generated = network.getGenerator(X_tensor)    
D_logits_real = network.getDiscriminator(X_tensor, Y_tensor)
D_logits_fake = network.getDiscriminator(X_tensor, Y_generated, True)

# Losses and optimizer
D_loss = loss.discriminatorLoss(D_logits_real, D_logits_fake, labelSmoothing)
G_loss, G_gan, G_L1 = loss.generatorLoss(D_logits_fake, Y_generated, Y_tensor, L1_Weight)
optimizer = loss.getOptimizer(lr, beta1, D_loss, G_loss)

# Tensorboard
train_summaries = [tf.summary.scalar('D_loss', D_loss), tf.summary.scalar('G_gan', G_gan), tf.summary.scalar('L1_loss', G_L1)]
train_merged_summaries = tf.summary.merge(train_summaries)

# Print variables
for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'):
    print(i)
for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'):
    print(i)

# Training
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    train_summary_writer = tf.summary.FileWriter(checkpointsPath + '/train', graph=tf.get_default_graph())

    globalStep = 0
    while True:
        try:
            _, summary = sess.run([optimizer, train_merged_summaries])
            train_summary_writer.add_summary(summary, globalStep)
            globalStep += 1 
        except tf.errors.OutOfRangeError:
            break