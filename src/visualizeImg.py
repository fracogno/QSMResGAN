import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Data
base_path = "/home/francesco/UQ/Job/QSM-GAN/"
path = "data/qsm_recon_challenge_deepQSM/"

filename = "phs_tissue_16x_MEDI.nii"
#filename = "phs_tissue_16x_STI.nii"
#filename = "phs_tissue_16x2018-10-18-2208arci-UnetMadsResidual-batch40-fs4-cost_L2-drop_0.05_ep50-shapes_shape64_ex100_2018_10_18_paper_DeepQSM.nii"
#filename = "chi_cosmos_16x.nii"

# Load data
X = nib.load(base_path + path + filename).get_data()
print(X.shape)

img_data = np.expand_dims(X, axis=-1)[:, :, int(X.shape[-1]//2), 0]
plt.imsave(base_path + "results/" + filename + ".png", img_data, cmap="gray")
#plt.imshow(img_data, cmap="gray")
#plt.show()

print(np.max(X))
print(np.min(X))
print(np.mean(X))