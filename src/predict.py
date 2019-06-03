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

# Load data
X = nib.load(data_path).get_data()
stabilizationFactor = 10
X = X * stabilizationFactor
print(X.shape)

# Add padding
SIZE = 256
val_X = (SIZE - X.shape[0]) // 2
val_Y = (SIZE - X.shape[1]) // 2
val_Z = (SIZE - X.shape[2]) // 2
X = np.pad(X, [(val_X, ), (val_Y, ), (val_Z, )],  'constant', constant_values=(0.0))
X = np.expand_dims(X, axis=-1)
print(X.shape)

# Start prediction
tf.reset_default_graph()
X_tensor = tf.placeholder(tf.float32, shape=[None, SIZE, SIZE, SIZE, 1], name='X')

# Create networks
with tf.variable_scope("generator"):
    Y_generated = Pix2Pix.getGenerator(X_tensor)

with tf.Session() as sess:
	saver = tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	saver.restore(sess, base_path + checkpoint_name + "/model.ckpt")

	X_final = sess.run(Y_generated, feed_dict={X_tensor : [X]})
	X_final /= stabilizationFactor

	X_final = X_final[0, val_X:-val_X, val_Y:-val_Y, val_Z:-val_Z]
	print(X_final.shape)

	plt.imsave(base_path + "results/gan.png", X_final[:, :, int(X_final.shape[-2]//2), 0], cmap="gray")
        
print("END")