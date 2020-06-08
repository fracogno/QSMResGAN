import os
import numpy as np
import datetime
import pickle
import json
import nibabel as nib
from pathlib import Path


def get_base_path(training):
    base_path = str(Path(__file__).parent.parent.parent) + "/"

    if training:
        checkpoint_path = base_path + "ckp_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "/"
        os.mkdir(checkpoint_path)
        return base_path, checkpoint_path
    else:
        return base_path


def get_data_folder_path(base_path, shape, samples):
    folder_path = base_path + "shape" + str(shape) + "_ex" + str(samples) + "_" + str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")) + "/"
    os.mkdir(folder_path)
    return folder_path


def save_pickle(path, array):
    with open(path, 'wb') as handle:
        pickle.dump(array, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def save_json(path, array):
    with open(path, 'w') as f:
        json.dump(array, f)


def load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def save_nii(volume, path):
    nib.save(nib.Nifti1Image(volume, np.eye(4)), path)


def load_nii(path):
    return nib.load(path).get_data()


def add_padding(volumes, pad_size):
    assert (len(volumes.shape) == 5 and len(pad_size) == 3)
    padded_volumes = []
    shape = volumes.shape[1:]
    for volume in volumes:
        # Add one if shape is not EVEN
        padded = np.pad(volume[:, :, :, 0], [(int(shape[0] % 2 != 0), 0), (int(shape[1] % 2 != 0), 0), (int(shape[2] % 2 != 0), 0)], 'constant', constant_values=(0.0))

        # Calculate how much padding to give
        val_x = (pad_size[0] - padded.shape[0]) // 2
        val_y = (pad_size[1] - padded.shape[1]) // 2
        val_z = (pad_size[2] - padded.shape[2]) // 2

        # Append padded volume
        padded_volumes.append(np.pad(padded, [(val_x,), (val_y,), (val_z,)], 'constant', constant_values=(0.0)))

    padded_volumes = np.array(padded_volumes)
    assert (padded_volumes.shape[1] == pad_size[0] and padded_volumes.shape[2] == pad_size[1] and padded_volumes.shape[3] == pad_size[2])

    return np.expand_dims(padded_volumes, -1), np.array(shape[:-1]), np.array([val_x, val_y, val_z])


def remove_padding(volumes, orig_shape, values):
    assert (len(volumes.shape) == 5 and len(orig_shape) == 3 and len(values) == 3)
    unpadded_volumes = []
    for volume in volumes:
        # Remove padding
        removed = volume[values[0]:-values[0], values[1]:-values[1], values[2]:-values[2]]

        # Append volume
        unpadded_volumes.append(removed[int(orig_shape[0] % 2 != 0):, int(orig_shape[1] % 2 != 0):, int(orig_shape[2] % 2 != 0):])

    unpadded_volumes = np.array(unpadded_volumes)
    shape = unpadded_volumes.shape
    assert (shape[1] == orig_shape[0] and shape[2] == orig_shape[1] and shape[3] == orig_shape[2])

    return unpadded_volumes


def apply_mask(volumes, masks):
    assert (len(volumes.shape) == 5 and len(masks.shape) == 5)
    return np.array([volumes[i] * masks[i] for i in range(len(volumes))])


def add_noise(noise_typ, volume, param):
    """
        'gauss'     : Gaussian-distributed additive noise.
        'poisson'   : Poisson-distributed noise generated from the data.
        's&p'       : Replaces random pixels with 0 or 1.
        'speckle'   : Multiplicative noise using out = image + n*image, where n is uniform noise with specified mean & variance.
    """
    assert (len(volume.shape) == 5)
    if noise_typ == "gauss":
        batch, row, col, dep, ch = volume.shape
        mean = 0
        var = param
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (batch, row, col, dep, ch))
        return volume + gauss
    elif noise_typ == "SP":
        assert (volume.shape[0] == 1 and volume.shape[-1] == 1)
        volume = volume[0, :, :, :, 0]
        s_vs_p = 0.5
        amount = param
        out = np.copy(volume)

        # Salt mode
        """num_salt = np.ceil(amount * volume.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in volume.shape]
        out[tuple(coords)] = 1"""

        # Pepper mode
        num_pepper = np.ceil(amount * volume.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in volume.shape]
        out[tuple(coords)] = 0

        return np.expand_dims(np.array([out]), -1)
    elif noise_typ == "speckle":
        batch, row, col, dep, ch = volume.shape
        mean = 0
        var = param
        sigma = var ** 0.5
        gauss = sigma * np.random.randn(batch, row, col, dep, ch)
        return volume + volume * gauss
    else:
        raise NotImplementedError("Noise type not found.")
