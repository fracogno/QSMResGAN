import tensorflow as tf
import numpy as np
import glob
import scipy.io

from utils import misc, tfrecord


def get_QSM_datasets(base_path, training_data_folder, qsm_challenge_2017_folder, qsm_challenge_2019_folder, shape, batch_size, get_train_data):
    datasets = {}
    records_manager = tfrecord.TFRecordManager()

    # Simulated data
    if get_train_data:
        datasets["train"] = records_manager.load_dataset(base_path + training_data_folder + "/train/", shape, batch_size)
        datasets["val"] = records_manager.load_dataset(base_path + training_data_folder + "/val/", shape, batch_size)

    # QSM challenges data
    datasets["qsm_2017"] = get_QSM_challenge_2017_data(base_path + qsm_challenge_2017_folder, (256, 256, 256), normalize=False)
    datasets["qsm_2019"] = get_QSM_challenge_2019_data(base_path + qsm_challenge_2019_folder, (256, 256, 256), normalize=True)

    return datasets


def get_QSM_challenge_2017_data(base_path, pad_size, normalize):
    initial_shape, pad_axis = None, None

    x = misc.load_nii(base_path + "/data/phs_tissue.nii.gz")
    if normalize:
        TEin_s = 8 / 1000
        frequency_rad = x * TEin_s * 2 * np.pi
        centre_freq = 297190802
        x = frequency_rad / (2 * np.pi * TEin_s * centre_freq) * 1e6

    x = np.float32(np.expand_dims(np.array([x]), -1))
    y = np.float32(np.expand_dims(np.array([misc.load_nii(base_path + "/data/chi_33.nii.gz")]), -1))
    mask = np.float32(np.expand_dims(np.array([misc.load_nii(base_path + "/data/msk.nii.gz")]), -1))

    if not pad_size is None:
        x, initial_shape, pad_axis = misc.add_padding(x, pad_size)
        y, _, _ = misc.add_padding(y, pad_size)
        mask, _, _ = misc.add_padding(mask, pad_size)

    assert (x.shape == y.shape and x.shape == mask.shape)
    return [{"x": x,
             "y": y,
             "mask": mask,
             "initial_shape": initial_shape,
             "pad_axis": pad_axis,
             "name": "vn"}]


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

        if not pad_size is None:
            x, initial_shape, pad_axis = misc.add_padding(x, pad_size)
            y, _, _ = misc.add_padding(y, pad_size)
            mask, _, _ = misc.add_padding(mask, pad_size)

        assert (x.shape == y.shape and x.shape == mask.shape)
        qsm_2019.append({"x": x,
                         "y": y,
                         "mask": mask,
                         "initial_shape": initial_shape,
                         "pad_axis": pad_axis,
                         "name": path.split("/")[-1]})
    return qsm_2019
