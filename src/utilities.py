import os
import tensorflow as tf
import numpy as np
import nibabel as nib
import glob


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


"""def dilateMask(mask):
    kernel = np.ones((3,3))
    for i in range(mask.shape[-1]):
        mask[:,:,i] = cv2.dilate(mask[:,:,i], kernel)

    return mask"""


def getMetrics(Y, X, masks):
    
    allRmse, allddRmse = [], []
    for i in range(len(X)):
        rmse, ddRMSE_detrend = computeddRMSE(Y[i], X[i], masks[i])
        allRmse.append(rmse)
        allddRmse.append(ddRMSE_detrend)

    allRmse, allddRmse = np.array(allRmse), np.array(allddRmse)

    return np.mean(allRmse), np.mean(allddRmse)


def getMetricsOLD(Y, X, msk, FinalSegment):
    # Metric 1 & 2
    rmse, ddRMSE_detrend = computeddRMSE(Y, X, msk)

    # Metric 3
    msk2 = msk.copy()
    choice = np.logical_or(np.greater(FinalSegment, 9), np.less(FinalSegment, 7))
    msk2[choice] = 0
    _, ddRMSE_detrend_Tissue = computeddRMSE(Y, X, msk2)

    # Metric 4
    '''msk2 = msk.copy()
    choice = FinalSegment != 11
    msk2[choice] = 0
    msk2 = dilateMask(msk2)
    _, ddRMSE_detrend_Blood = computeddRMSE(Y, X, msk2)
    #ddRMSE_detrend_Blood = 0.0'''

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


    return round(rmse, 4), round(ddRMSE_detrend, 4), round(ddRMSE_detrend_Tissue, 4), round(ddRMSE_detrend_DGM, 4), round(deviationFromLinearSlope, 4)



def loadChallengeData(path, normalize=True):

    X, Y, mask = [], [], []
    for sim in range(1,3):
        for snr in range(1,3):

            X_tmp = nib.load(path + "Sim" + str(sim) + "Snr" + str(snr) + "/Frequency.nii.gz").get_data()

            if normalize:
                # Rescale validation input of the network
                TEin_s = 8 / 1000
                frequency_rad = X_tmp * TEin_s * 2 * np.pi
                centre_freq = 297190802
                X_scaled = frequency_rad / (2 * np.pi * TEin_s * centre_freq) * 1e6
            else:
                X_scaled = X_tmp

            X.append(X_scaled)
            Y.append(nib.load(path + "Sim" + str(sim) + "Snr" + str(snr) + "/GT/Chi.nii.gz").get_data())
            mask.append(nib.load(path + "Sim" + str(sim) + "Snr" + str(snr) + "/MaskBrainExtracted.nii.gz").get_data())


    X, Y, mask = np.array(X), np.array(Y), np.array(mask)
    assert(X.shape == Y.shape and X.shape == mask.shape)

    return X, Y, mask


def loadChallengeOneData(path):
    X, Y, mask = [], [], []

    for i in range(4):
        mask.append(nib.load(path + "msk.nii.gz").get_data())

        X_tmp = nib.load(path + "phs_tissue.nii.gz").get_data()
        if i < 2:
            TEin_s = 8 / 1000
            frequency_rad = X_tmp * TEin_s * 2 * np.pi
            centre_freq = 297190802
            X_tmp = frequency_rad / (2 * np.pi * TEin_s * centre_freq) * 1e6
        X.append(X_tmp)
    
    Y.append(nib.load(path + "chi_33.nii.gz").get_data())
    Y.append(nib.load(path + "chi_cosmos.nii.gz").get_data())
    Y.append(nib.load(path + "chi_33.nii.gz").get_data())
    Y.append(nib.load(path + "chi_cosmos.nii.gz").get_data())

    return np.array(X), np.array(Y), np.array(mask)



def saveNii(volume, path):
    nib.save(nib.Nifti1Image(volume, np.eye(4)), path)



def addPadding(volumes, size):

    paddedVolumes = []
    for volume in volumes:

        # Add one if shape is not EVEN
        padded = np.pad(volume, [(int(volume.shape[0] % 2 != 0), 0), (int(volume.shape[1] % 2 != 0), 0), (int(volume.shape[2] % 2 != 0), 0)],  'constant', constant_values=(0.0))

        val_X = (size[0] - padded.shape[0]) // 2
        val_Y = (size[1] - padded.shape[1]) // 2
        val_Z = (size[2] - padded.shape[2]) // 2
        padded = np.pad(padded, [(val_X, ), (val_Y, ), (val_Z, )],  'constant', constant_values=(0.0))

        paddedVolumes.append(np.expand_dims(padded, axis=-1))

    paddedVolumes = np.array(paddedVolumes)
    assert(paddedVolumes.shape[1] == size[0] and paddedVolumes.shape[2] == size[1] and paddedVolumes.shape[3] == size[2])

    return paddedVolumes, volumes[0].shape, (val_X, val_Y, val_Z)


def removePadding(volumes, originalShape, values):

    removedVolumes = []
    for volume in volumes:
        removed = volume[values[0]:-values[0], values[1]:-values[1], values[2]:-values[2]]
        removed = removed[int(originalShape[0] % 2 != 0):, int(originalShape[1] % 2 != 0):, int(originalShape[2] % 2 != 0):, 0]
        removedVolumes.append(removed)

    return np.array(removedVolumes)


def applyMaskToVolume(volumes, masks):

    for i in range(len(volumes)):
        volumes[i] = volumes[i] * masks[i]

    return volumes


def getTrainingDataTF(path, batchSize, epochs):
    input_shape = (64, 64, 64, 1)
    train_data_filename = generate_file_list(file_path=path + "/train/", p_shape=input_shape)
    train_input_fn = data_input_fn(train_data_filename, p_shape=input_shape, batch=batchSize, nepochs=epochs, shuffle=True)
    X_tensor, Y_tensor, _ = train_input_fn()
    assert(X_tensor.shape[1:] == Y_tensor.shape[1:])

    return X_tensor, Y_tensor


def loadRealData():
	basePath = "/scratch/cai/deepQSMGAN/data/realData/"
	#basePath = "/home/francesco/UQ/deepQSMGAN/data/realData/"
	maskPath = "cut_phase/"

	X, masks, names = [], [], []

	for folder in glob.glob(basePath + maskPath + "*"):
		for folder2 in glob.glob(folder + "/*"):
			phaseName = glob.glob(folder2 + "/*scaledTOPPM.nii")
			assert(len(phaseName) == 1)
			phase = nib.load(phaseName[0]).get_data()
			
			mask = nib.load(folder2 + "/eroded_mask.nii").get_data()
			assert(mask.shape == phase.shape)
			X.append(phase)
			masks.append(mask)
			names.append(phaseName[0])

		if len(X) > 30:
			break

	X = np.array(X)
	masks = np.array(masks)
	print(X.shape)
	print(masks.shape)

	return X, masks, names