import tensorflow as tf
import numpy as np
import datetime
import os
import nibabel as nib
import matplotlib.pyplot as plt

import Pix2Pix, utilities as util

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

# Paths
data_path = "/scratch/cai/QSM-GAN/data/qsm_recon_challenge_deepQSM/phs_tissue_16x.nii"
base_path = "/scratch/cai/QSM-GAN/"
checkpoint_name = "checkpoints_2019-5-24_1845_shapes_shape64_ex100_2018_10_18"


# 64 => [40:104,48:112,32:96]
# 128 => [8:136,16:144,:]
X = nib.load(data_path).get_data()
X = np.expand_dims(X[8:136,16:144,:], axis=-1)
stabilizationFactor = 10
X = X * stabilizationFactor
print(X.shape)

tf.reset_default_graph()

input_shape = 128
X_tensor = tf.placeholder(tf.float32, shape=[None, input_shape, input_shape, input_shape, 1], name='X')

# Create networks
with tf.variable_scope("generator"):
    Y_generated = Pix2Pix.getGenerator(X_tensor)

with tf.Session() as sess:
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	saver.restore(sess, base_path + checkpoint_name + "/model.ckpt")

	X_final = sess.run(Y_generated, feed_dict={X_tensor : [X]})
	X_final /= stabilizationFactor

	plt.imsave(base_path + "results/gan.png", X_final[0, :, :, int(input_shape//2), 0], cmap="gray")
        
print("END")