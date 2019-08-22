import os
import tensorflow as tf
import numpy as np
import cv2

def generate_file_list(file_path, p_shape):
    """
    Generates a list of the filenames and the paths in 'filepath'
    :param file_path: the path to the folder where the files of interest resides
    :param p_shape:
    :return: a list where the filenames and filepaths have been joined
    """
    filenames = os.listdir(file_path)

    for index, item in enumerate(filenames):
        if item.__contains__('size' + str(p_shape[0])):
            filenames[index] = file_path + item
        else:
            raise FileNotFoundError('you have files in the folder that does not match the shapes')

    return filenames


def data_input_fn(filenames, p_shape, batch=None, nepochs=None, shuffle=True):
    def _parser(record):
        features = {
            'forward_img': tf.FixedLenFeature([], tf.string, default_value=""),
            'ground_truth_img': tf.FixedLenFeature([], tf.string, default_value="")
        }
        parsed_record = tf.parse_single_example(record, features)
        forward_image = tf.decode_raw(parsed_record['forward_img'], tf.float32)
        forward_image = tf.reshape(forward_image, [p_shape[0], p_shape[1], p_shape[2], 1])

        ground_truth = tf.decode_raw(parsed_record['ground_truth_img'], tf.float32)
        ground_truth = tf.reshape(ground_truth, [p_shape[0], p_shape[1], p_shape[2], 1])

        return forward_image, ground_truth

    def _input_fn():

        dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').map(_parser)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=5000)
        dataset = dataset.repeat(nepochs)
        dataset = dataset.batch(batch)
        # TODO some dataset cache can be made so you can cache on the harddisk

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()

        return features, labels, dataset

    return _input_fn


def norm(data):
    """
    Normalizes the input by subtracting by the mean and dividing by the standard deviation
    :param data: input data to be normalized
    :param includemeanstd: Boolean whether to return the mean and standard deviation
    :return: Normalized data and mean and standard deviation
    """
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / std

    return data, mean, std


def computeddRMSE(true, fake, mask):

    true_flat = true.flatten()
    fake_flat = fake.flatten()
    mask_flat = np.array(mask.flatten(), dtype=bool)

    # Get only elements in mask
    true_new = true_flat[mask_flat]
    fake_new = fake_flat[mask_flat]
    
    # Demean
    true_demean = true_new - np.mean(true_new);
    fake_demean = fake_new - np.mean(fake_new);

    rmse = 100 * np.linalg.norm( true_demean - fake_demean ) / np.linalg.norm(true_demean);

    # Detrend
    P1 = np.polyfit(true_demean, fake_demean, 1)
    P = [0, 0]
    P[0] = 1 / P1[0]
    P[1] = -P1[1] / P1[0]
    res = np.polyval(P, fake_demean)

    # RMSE
    ddrmse = 100 * np.linalg.norm( res - true_demean ) / np.linalg.norm(true_demean);

    return rmse, ddrmse


def dilateMask(mask):
    kernel = np.ones((3,3))
    for i in range(mask.shape[-1]):
        mask[:,:,i] = cv2.dilate(mask[:,:,i], kernel)

    return mask


def getMetrics(Y, X, msk, FinalSegment):
    # Metric 1 & 2
    rmse, ddRMSE_detrend = computeddRMSE(Y, X, msk)

    # Metric 3
    msk2 = msk.copy()
    choice = np.logical_or(np.greater(FinalSegment, 9), np.less(FinalSegment, 7))
    msk2[choice] = 0
    _, ddRMSE_detrend_Tissue = computeddRMSE(Y, X, msk2)

    # Metric 4
    msk2 = msk.copy()
    choice = FinalSegment != 11
    msk2[choice] = 0
    msk2 = dilateMask(msk2)
    _, ddRMSE_detrend_Blood = computeddRMSE(Y, X, msk2)
    ddRMSE_detrend_Blood = 0.0

    # Metric 5
    msk2 = msk.copy()
    choice = FinalSegment >= 7
    msk2[choice] = 0
    _, ddRMSE_detrend_DGM = computeddRMSE(Y, X, msk2)

    # Metric 6
    DGMmean_true_ds, DGMmean_recon = [], []
    for tissue in range(1,7):
        DGMmean_true_ds.append(np.mean(Y[FinalSegment == tissue]))
        DGMmean_recon.append(np.mean(X[FinalSegment == tissue]))

    P = np.polyfit(DGMmean_true_ds,DGMmean_recon, 1)
    DGM_slope_ds = 1 * P[0]
    deviationFromLinearSlope = abs(1-DGM_slope_ds)

    # Metric 7
    calcStreak = 0.0

    # Metric 8
    deviationFromCalcMoment = 0.0


    return round(rmse, 4), round(ddRMSE_detrend, 4), round(ddRMSE_detrend_Tissue, 4), round(ddRMSE_detrend_Blood, 4), \
            round(ddRMSE_detrend_DGM, 4), round(deviationFromLinearSlope, 4), round(calcStreak, 4), round(deviationFromCalcMoment, 4)