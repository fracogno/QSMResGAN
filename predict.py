import tensorflow as tf
import numpy as np
import scipy.io
import Pix2Pix, utilities as util
import pickle
import nibabel as nib
import argparse

parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('--input_filename', type=str, help='Path to input file')
parser.add_argument('--checkpoint', type=str, help='Path to checkpoint')
args = parser.parse_args()

# Paths
base_path = "/scratch/cai/QSM-GAN/"

# Load data
#X_original = scipy.io.loadmat(base_path + "challenge/" + args.input_filename + ".mat")[args.input_filename]
img = nib.load(base_path + "challenge/" + args.input_filename + ".nii")
X_original = img.get_data()
msk = nib.load(base_path + "challenge/Sim1/MaskBrainExtracted.nii").get_data()

TEin_s = 8 / 1000
frequency_rad = X_original * TEin_s * 2 * np.pi
centre_freq = 297190802
frequency_ppm = frequency_rad / (2 * np.pi * TEin_s * centre_freq) * 1e6

X_original = frequency_ppm

# Add one if shape is not EVEN
X = np.pad(X_original, [(int(X_original.shape[0] % 2 != 0), 0), (int(X_original.shape[1] % 2 != 0), 0), (int(X_original.shape[2] % 2 != 0), 0)],  'constant', constant_values=(0.0))
print(X.shape)

Y = scipy.io.loadmat(base_path + "challenge/chi_33.mat")["chi_33"]

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
	saver.restore(sess, base_path + args.checkpoint)

	# Predict from network
	predicted = sess.run(Y_generated, feed_dict={X_tensor : [X]})

	# De-normalize the raw output
	assert(predicted.shape[0] == 1 and predicted.shape[-1] == 1)
	predicted = predicted[0, val_X:-val_X, val_Y:-val_Y, val_Z:-val_Z, 0]
	print(predicted.shape)

	# Remove where not EVEN
	predicted = predicted[int(X_original.shape[0] % 2 != 0):, int(X_original.shape[1] % 2 != 0):, int(X_original.shape[2] % 2 != 0):]
	print(predicted.shape)

	assert(predicted.shape[0] == msk.shape[0] and predicted.shape[1] == msk.shape[1] and predicted.shape[2] == msk.shape[2])
	masked = predicted * msk

	# Save result
	scipy.io.savemat("/scratch/cai/QSM-GAN/results/" + args.checkpoint.split("/")[0] + "-" + \
			 str(args.normalization) + "-" + args.input_filename.split("/")[0] + "-" + args.input_filename.split("/")[1] + ".mat" , {"QSMGAN" : masked} )

	# Save NII
	masked_img = nib.Nifti1Image(masked, img.affine)
	nib.save(masked_img, "/scratch/cai/QSM-GAN/results/" + args.checkpoint.split("/")[0] + "-" + str(args.normalization) + "-" + \
				 args.input_filename.split("/")[0] + "-" + args.input_filename.split("/")[1] + ".nii")

	# Compute RMSE
	predicted = predicted.flatten()
	Y = Y.flatten()
	rmse = 100 * np.linalg.norm( predicted - Y ) / np.linalg.norm(Y)
	print(rmse)