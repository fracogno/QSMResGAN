import tensorflow as tf
import numpy as np
import datetime
import os
import nibabel as nib
import matplotlib.pyplot as plt

import Pix2Pix, utilities as util
from tensorflow.python.client import device_lib 


def norm(data):
	"""
	Normalizes the input by subtracting by the mean and dividing by the standard deviation
	:param data: input data to be normalized
	:param includemeanstd: Boolean whether to return the mean and standard deviation
	:return: Normalized data and mean and standard deviation
	"""
	mean = np.mean(data)
	std = np.std(data)

	data = (data - mean) / std

	return data, mean, std

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used
print(device_lib.list_local_devices())

# Data
path = "/scratch/cai/qsm_recon_challenge_deepQSM/"

index = 3
Y_names = ["phs_tissue_16x_MEDI.nii", \
            "phs_tissue_16x_STI.nii", \
            "phs_tissue_16x2018-10-18-2208arci-UnetMadsResidual-batch40-fs4-cost_L2-drop_0.05_ep50-shapes_shape64_ex100_2018_10_18_paper_DeepQSM.nii", \
            "chi_cosmos_16x.nii"]

X = nib.load(path + "phs_tissue_16x.nii").get_data()
Y = nib.load(path + Y_names[index]).get_data()

X = np.expand_dims(X[40:104,48:112,32:96], axis=-1)
Y = np.expand_dims(Y[40:104,48:112,32:96], axis=-1)

stabilizationFactor = 10
X = X * stabilizationFactor
Y = Y * stabilizationFactor

# 64 => [40:104,48:112,32:96]
# 128 => [8:136,16:144,:]

print(X.shape)
print(Y.shape)

# Paths
base_path = "/scratch/cai/QSM-GAN/"
checkpoint_path = base_path + "checkpoints_2019-5-22_019_shapes_shape64_ex100_2018_10_18" + "/model.ckpt"

tf.reset_default_graph()

input_shape = 64
X_tensor = tf.placeholder(tf.float32, shape=[None, input_shape, input_shape, input_shape, 1], name='X')
Y_tensor = tf.placeholder(tf.float32, shape=[None, input_shape, input_shape, input_shape, 1], name='Y')


# Create networks
with tf.variable_scope("generator"):
    Y_generated = Pix2Pix.getGenerator(X_tensor)

with tf.name_scope("real_discriminator"):
    with tf.variable_scope("discriminator"):
        D_logits_real = Pix2Pix.getDiscriminator(X_tensor, Y_tensor)
      
with tf.name_scope("fake_discriminator"):
    with tf.variable_scope("discriminator", reuse=True):
        D_logits_fake = Pix2Pix.getDiscriminator(X_tensor, Y_generated)

accuracy = tf.reduce_mean(tf.abs(Y_generated - Y_tensor))

with tf.Session() as sess:
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	saver.restore(sess, checkpoint_path)

	X_final = sess.run(Y_generated, feed_dict={X_tensor : [X]})
	X_final /= stabilizationFactor

	plt.imsave(base_path + "results/gan.png", X_final[0, :, :, int(input_shape//2), 0], cmap="gray")
        
print("END")
