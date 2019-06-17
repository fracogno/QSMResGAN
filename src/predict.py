import tensorflow as tf
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import Pix2Pix, utilities as util
import pickle

# Paths
base_path = "/scratch/cai/QSM-GAN/"
data_path = "challenge/"
filename = "phs_tissue.nii"
#checkpoint_name = "checkpoints_2019-5-24_1845_shapes_shape64_ex100_2018_10_18"
checkpoint_name = "checkpoints_2019-5-28_1616_shapes_shape64_ex100_2018_10_18"

# Load data
X = nib.load(base_path + data_path + filename).get_data()
print(X.shape)

# Normalize
norm_val = 0 # 0 => NO NORM, 1 => STABILIZATION FACTOR, 2 => NORMALIZE
if norm_val == 1:
	stabilizationFactor = 10
	X = X * stabilizationFactor
elif norm_val == 2:
	X, phase_mean, phase_std = util.norm(X)

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
	
	# Remove normalization
	if norm_val == 1:
		X_final /= stabilizationFactor
	elif norm_val == 2:
		X_final = X_final + (3 * phase_mean)
		X_final = X_final * (3 * phase_std)

	assert(X_final.shape[0] == 1 and X_final.shape[4] == 1)
	X_final = X_final[0, val_X:-val_X, val_Y:-val_Y, val_Z:-val_Z, 0]
	print(X_final.shape)

	with open(base_path + data_path + filename.split(".")[0] + "_result" + ".pkl",'wb') as f: pickle.dump(X_final, f)
print("END")