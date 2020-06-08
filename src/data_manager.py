import tensorflow as tf
import numpy as np
import glob
import misc

def get_QSM_datasets(base_path, training_data_folder, qsm_challenge_2017_folder, qsm_challenge_2019_folder, shape, batch_size, noisy_data=False):
    datasets = {}

    # QSM challenges data
    datasets["qsm_2017"] = get_QSM_challenge_2017_data(base_path + qsm_challenge_2017_folder, normalize=False)
    datasets["qsm_2019"] = get_QSM_challenge_2019_data(base_path + qsm_challenge_2019_folder, (192, 224, 224), normalize=True)

    if noisy_data:
        for path in glob.glob(base_path + "noisy_data/*"):
            x = np.float32(np.expand_dims(np.array([misc.load_nii(path + "/x.nii")]), -1))
            y = np.float32(np.expand_dims(np.array([misc.load_nii(path + "/y.nii")]), -1))
            mask = np.float32(np.expand_dims(np.array([misc.load_nii(path + "/mask.nii")]), -1))

            datasets[path.split("/")[-1]] = [{"x": x,
                                              "y": y,
                                              "mask": mask,
                                              "name": path.split("/")[-1]}]

    return datasets


def get_QSM_challenge_2017_data(base_path, normalize):
    x = misc.load_nii(base_path + "/data/phs_tissue.nii.gz")
    if normalize:
        TEin_s = 8 / 1000
        frequency_rad = x * TEin_s * 2 * np.pi
        centre_freq = 297190802
        x = frequency_rad / (2 * np.pi * TEin_s * centre_freq) * 1e6

    y = misc.load_nii(base_path + "/data/chi_33.nii.gz")
    mask = misc.load_nii(base_path + "/data/msk.nii.gz")

    x, y, mask = np.array([x]), np.array([y]), np.array([mask])
    assert (x.shape == y.shape and x.shape == mask.shape)

    return [{"x": np.float32(np.expand_dims(x, -1)),
             "y": np.float32(np.expand_dims(y, -1)),
             "mask": np.float32(np.expand_dims(mask, -1)),
             "name": "qsm_2017"}]


def get_QSM_challenge_2019_data(base_path, pad_size, normalize):
    qsm_2019, initial_shape, pad_axis = [], None, None

    for path in glob.glob(base_path + "/Sim*Snr*"):
        x = misc.load_nii(path + "/Frequency.nii.gz")
        if normalize:
            TEin_s = 8 / 1000
            frequency_rad = x * TEin_s * 2 * np.pi
            centre_freq = 297190802
            x = frequency_rad / (2 * np.pi * TEin_s * centre_freq) * 1e6

        x = np.float32(np.expand_dims(np.array([x]), -1))
        y = np.float32(np.expand_dims(np.array([misc.load_nii(path + "/GT/Chi.nii.gz")]), -1))
        mask = np.float32(np.expand_dims(np.array([misc.load_nii(path + "/MaskBrainExtracted.nii.gz")]), -1))
        assert (x.shape == y.shape and x.shape == mask.shape)

        if not pad_size is None:
            x, initial_shape, pad_axis = misc.add_padding(x, pad_size)
            y, _, _ = misc.add_padding(y, pad_size)
            mask, _, _ = misc.add_padding(mask, pad_size)

        qsm_2019.append({"x": x,
                         "y": y,
                         "mask": mask,
                         "initial_shape": initial_shape,
                         "pad_axis": pad_axis,
                         "name": path.split("/")[-1]})
    return qsm_2019
