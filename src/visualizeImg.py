import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Data
base_path = "/home/francesco/UQ/Job/QSM-GAN/"

realData = True

if realData:

	path = "data/qsm_recon_challenge_deepQSM/"

	index = 3
	Y_names = ["phs_tissue_16x_MEDI.nii", \
	            "phs_tissue_16x_STI.nii", \
	            "phs_tissue_16x2018-10-18-2208arci-UnetMadsResidual-batch40-fs4-cost_L2-drop_0.05_ep50-shapes_shape64_ex100_2018_10_18_paper_DeepQSM.nii", \
	            "chi_cosmos_16x.nii"]

	X = nib.load(base_path + path + "phs_tissue_16x.nii").get_data()
	Y = nib.load(base_path + path + Y_names[index]).get_data()

	X = X*100

	X = np.expand_dims(X[40:104,48:112,32:96], axis=-1)
	Y = np.expand_dims(Y[40:104,48:112,32:96], axis=-1)

	plt.imsave(base_path + "results/"+Y_names[index]+".png", Y[:, :, 32, 0], cmap="gray")
else:
	last_path = "shapes_shape64_ex512_2019_05_01"

	X, Y = [], []
	for i in range(129, 329):
		X.append(nib.load(base_path + "data/" + last_path + "/traintrain1-size64-ex512_" + str(i) + "_forward_tfrange.nii").get_data())
		Y.append(nib.load(base_path + "data/" + last_path + "/traintrain1-size64-ex512_" + str(i) + "_ground_truth_tfrange.nii").get_data())
	X = np.array(X)
	Y = np.array(Y)

	plt.imsave(base_path + "results/sample_fake.png", Y[0, :, :, 32], cmap="gray")

print(np.max(X))
print(np.min(X))
print(np.mean(X))

print(X.shape)
print(Y.shape)'''