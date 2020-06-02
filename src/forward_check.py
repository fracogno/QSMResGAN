import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def generate_3d_dipole_kernel(p_shape, voxel_size, b_vec):
    """
    Creates a 3D dipole kernel
    :param p_shape: is a tupel that defines the shape of the dipole kernel
    :param plot:
    :return:
    from https://github.com/stevenxcao/susceptibility-imaging-toolkit/blob/master/qsm_star.py
    """
    fov = np.array(p_shape) * np.array(voxel_size)

    ry, rx, rz = np.meshgrid(np.arange(-p_shape[1] // 2, p_shape[1] // 2),
                             np.arange(-p_shape[0] // 2, p_shape[0] // 2),
                             np.arange(-p_shape[2] // 2, p_shape[2] // 2))

    rx, ry, rz = rx / fov[0], ry / fov[1], rz / fov[2]

    sq_dist = rx ** 2 + ry ** 2 + rz ** 2
    sq_dist[sq_dist == 0] = 1e-6
    d2 = ((b_vec[0] * rx + b_vec[1] * ry + b_vec[2] * rz) ** 2) / sq_dist
    kernel = (1 / 3 - d2)

    return kernel



def forward_sample(chi_sample, kernel):
    """
    QSM forward problem for one sample.

    :param chi_sample:
    :param kernel:
    :return:
    """
    scaling = np.sqrt(chi_sample.size)
    chi_fft = np.fft.fftn(chi_sample) / scaling
    tissue_phase = np.real(np.fft.ifftn(chi_fft * np.fft.fftshift(kernel)) * scaling)

    return tissue_phase


# Loading data
path = "/home/francesco/Documents/University/UQ/QSMResGAN/data/"
X = nib.load(path + "QSM_Challenge2_download_stage2/DatasetsStep2/Sim2Snr2/Frequency.nii.gz").get_data()
chi = nib.load(path + "QSM_Challenge2_download_stage2/DatasetsStep2/Sim2Snr2/GT/Chi.nii.gz").get_data()
msk = nib.load(path + "QSM_Challenge2_download_stage2/DatasetsStep2/Sim2Snr2/MaskBrainExtracted.nii.gz").get_data()
print(X.shape)
print(chi.shape)

# Forward model
dipole_kernel = generate_3d_dipole_kernel(X.shape, (1, 1, 1), [0, 0, 1])
forward_data_normed = forward_sample(chi_sample=chi, kernel=dipole_kernel)
print(forward_data_normed.shape)

# Scaling
frequency_ppm = forward_data_normed
TEin_s = 8 / 1000
centre_freq = 297190802
frequency_rad = frequency_ppm * (2 * np.pi * TEin_s * centre_freq) / 1e6
frequency = frequency_rad / (TEin_s * 2 * np.pi)
print(frequency.shape)

# Visualize
plt.imshow(frequency[:,:,X.shape[2]//2], cmap="gray")
plt.show()

plt.imshow(X[:,:,X.shape[2]//2], cmap="gray")
plt.show()