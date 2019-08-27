import tensorflow as tf
import numpy as np
import src.network as network, src.utilities as util
import pickle
import nibabel as nib

# Paths
base_path = "/home/francesco/UQ/deepQSMGAN/"
checkpoint_path = "ckp_2019827_110_shapes_shape64_ex100_2019_08_20/"

# Validation
X_val_nib = nib.load(base_path + "QSM_Challenge2_download_stage2/DatasetsStep2/Sim2Snr2/Frequency.nii.gz")
X_tmp = X_val_nib.get_data()
Y_tmp = nib.load(base_path + "QSM_Challenge2_download_stage2/DatasetsStep2/Sim2Snr2/GT/Chi.nii.gz")
Y_val = Y_tmp.get_data()
mask = nib.load(base_path + "QSM_Challenge2_download_stage2/DatasetsStep2/Sim2Snr2/MaskBrainExtracted.nii.gz").get_data()
finalSegment = nib.load(base_path + "QSM_Challenge2_download_stage2/DatasetsStep2/Sim2Snr2/GT/Segmentation.nii.gz").get_data()

# Rescale validation
TEin_s = 8 / 1000
frequency_rad = X_tmp * TEin_s * 2 * np.pi
centre_freq = 297190802
X_val_original = frequency_rad / (2 * np.pi * TEin_s * centre_freq) * 1e6
print("X val original shape " + str(X_val_original.shape))

# Add one if shape is not EVEN
X_val = np.pad(X_val_original, [(int(X_val_original.shape[0] % 2 != 0), 0), (int(X_val_original.shape[1] % 2 != 0), 0), (int(X_val_original.shape[2] % 2 != 0), 0)],  'constant', constant_values=(0.0))
print("X val evened shape " + str(X_val.shape))

# Pad to multiple of 2^n
VAL_SIZE = 256
X_tensor = tf.placeholder(tf.float32, shape=[None, VAL_SIZE, VAL_SIZE, VAL_SIZE, 1], name='X_val')
val_X = (VAL_SIZE - X_val.shape[0]) // 2
val_Y = (VAL_SIZE - X_val.shape[1]) // 2
val_Z = (VAL_SIZE - X_val.shape[2]) // 2
X_val = np.pad(X_val, [(val_X, ), (val_Y, ), (val_Z, )],  'constant', constant_values=(0.0))
print("X val padded to 2^n multiple: " + str(X_val.shape))
print("Y val: " + str(Y_val.shape))
print("Mask shape: " + str(mask.shape))

# Network
Y_generated = network.getGenerator(X_tensor)    

num_metrics = len(util.getMetrics(Y_val, Y_val, mask, finalSegment))
with tf.Session() as sess:

	for i in range(num_metrics):
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, base_path + checkpoint_path + "model-metric" + str(i))

		# Predict from network
		predicted = sess.run(Y_generated, feed_dict={ X_tensor : [np.expand_dims(X_val, axis=-1)] })

		#Remove paddings and if it was not even shape
		predicted = predicted[0, val_X:-val_X, val_Y:-val_Y, val_Z:-val_Z, 0]
		predicted = predicted[int(X_val_original.shape[0] % 2 != 0):, int(X_val_original.shape[1] % 2 != 0):, int(X_val_original.shape[2] % 2 != 0):]
		assert(predicted.shape[0] == mask.shape[0] and predicted.shape[1] == mask.shape[1] and predicted.shape[2] == mask.shape[2])
		predicted = predicted * mask

		# Calculate metrics over validation and save it
		metrics = util.getMetrics(Y_val, predicted, mask, finalSegment)
		print(metrics)

		# Save NII
		masked_img = nib.Nifti1Image(masked, np.eye(4))
		nib.save(masked_img, base_path + "result-metric" + str(i) + ".nii")