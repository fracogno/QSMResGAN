import src.utilities as utils
import src.forward_check as forward_check
import numpy as np
import nibabel as nib
import glob
import tensorflow as tf

def calculateMetricsMethod(groundTruthPath, methodPath):
	X, Y, mask = utils.loadChallengeData(path, normalize=False)
	print(X.shape)
	print(Y.shape)
	print(mask.shape)

	Y_predicted = []
	for sim in range(1,3):
		for snr in range(1,3):
			Y_predicted.append(nib.load(methodPath + "Sim" + str(sim) + "Snr" + str(snr) + ".nii").get_data())

	metrics = utils.getMetrics(Y, Y_predicted, mask)
	print(metrics)

def calculateMetricDeepQSM():
	X, Y, mask = utils.loadChallengeData(path)
	print(X.shape)
	print(Y.shape)
	print(mask.shape)

	_, originalShape, valuesSplit = utils.addPadding(Y, 256)

	resAll = []
	for i in range(len(Y)):
		res = nib.load("/home/francesco/UQ/deepQSMGAN/data/deepQSM/inputs-padded-normalized-"+str(i)+"_DeepQSM.nii").get_data()
		res = utils.removePadding(np.array(np.expand_dims([res],-1)), originalShape, valuesSplit)[0]
		res = res * mask[i]
		resAll.append(res)
		utils.saveNii(res, "/home/francesco/UQ/deepQSMGAN/data/deepQSM/out-" + str(i) + ".nii")

	resAll = np.array(resAll)
	print(resAll.shape)
	metrics = utils.getMetrics(Y, resAll, mask)
	print(metrics)

def prepareInputDeepQSM(path):
	X, Y, mask = utils.loadChallengeData(path, False)
	print(X.shape)
	print(Y.shape)
	print(mask.shape)

	X_padded, originalShape, valuesSplit = utils.addPadding(X, 256)
	print(X_padded.shape)
	for i in range(len(X_padded)):
		utils.saveNii(X_padded[i,:,:,:,0], "/home/francesco/UQ/deepQSMGAN/data/deepQSM/inputs/inputs-padded-" + str(i) + ".nii")

def saveNormalized(path):
	X, Y, mask = utils.loadChallengeData(path)
	print(X.shape)
	print(Y.shape)
	print(mask.shape)

	for i in range(len(X)):
		utils.saveNii(Y[i] * mask[i], "/home/francesco/UQ/deepQSMGAN/data/QSM_Challenge2_Normalized/chi-masked-" + str(i) + ".nii")

def saveTrainingData():
	basePath ="/home/francesco/UQ/deepQSMGAN/"
	dataPath = "data/shapes_shape64_ex100_2019_08_20"
	X_tensor, Y_tensor = utils.getTrainingDataTF(basePath + dataPath, 20, 1)

	with tf.Session() as sess:

		X, Y = sess.run([X_tensor, Y_tensor])
		print(X.shape)
		print(Y.shape)
		print(str(X.max()) + " " + str(X.min()))
		print(str(Y.max()) + " " + str(Y.min()))

def forwardModel():
	predicted = []
	for sim in range(1,3):
		for snr in range(1,3):
			predicted.append(nib.load("/home/francesco/UQ/deepQSMGAN/data/deepQSMResGAN/" + "Sim" + str(sim) + "Snr" + str(snr) + ".nii").get_data())

	predicted = np.array(predicted)
	print(predicted.shape)

	X, originalShape, valuesSplit = utils.addPadding(predicted, 256)
	X = X[:,:,:,:,0]
	print(X.shape)

	dipole_kernel = forward_check.generate_3d_dipole_kernel(X.shape, (1, 1, 1), [0, 0, 1])
	forward_data_normed = forward_check.forward_sample(chi_sample=chi, kernel=dipole_kernel)
	print(forward_data_normed.shape)

def splitPhase():
	predicted = []
	path = "/home/francesco/UQ/deepQSMGAN/data/QSM_Challenge2_download_stage2/DatasetsStep2/"
	for sim in range(1,3):
		for snr in range(1,3):
			aa = nib.load(path + "Sim" + str(sim) + "Snr" + str(snr) + "/Phase.nii")
			tmp = aa.get_data()
			for i in range(4):
				print(tmp[:,:,:,i].shape)
				nib.save(nib.Nifti1Image(tmp[:,:,:,i], aa.affine), path + "Sim" + str(sim) + "Snr" + str(snr) + "/Phase-" + str(i) + ".nii.gz")

def metricsChallengeOne():
	basePath ="/home/francesco/UQ/deepQSMGAN/"
	X, Y, masks = utils.loadChallengeOneData(basePath + "data/20170327_qsm2016_recon_challenge/data/")

	print(X.shape)
	for i in range(len(X)):
		
		predicted = nib.load(basePath + "data/deepQSMResGAN/Challenge1/out-metric0-vol-" + str(i) + ".nii").get_data()
		print(utils.getMetrics([Y[i]], [np.array(predicted)], [masks[i]]))

def normalizeMag():
	path = "/home/francesco/UQ/deepQSMGAN/data/QSM_Challenge2_download_stage2/DatasetsStep2/"
	for sim in range(1,3):
		for snr in range(1,3):
			tmp = nib.load(path + "Sim" + str(sim) + "Snr" + str(snr) + "/Magnitude.nii").get_data()

			TEin_s = 8 / 1000
			frequency_rad = tmp * TEin_s * 2 * np.pi
			centre_freq = 297190802
			X = frequency_rad / (2 * np.pi * TEin_s * centre_freq) * 1e6

			print(X.shape)

			utils.saveNii(X, path + "Sim" + str(sim) + "Snr" + str(snr) + "/Norm-magnitude.nii")


def mergePhase():
	path = "/home/francesco/UQ/deepQSMGAN/data/QSM_Challenge2_download_stage2/DatasetsStep2/"
	for sim in range(1,3):
		for snr in range(1,3):
			phases = []
			for i in range(4):
				aa = nib.load(path + "Sim" + str(sim) + "Snr" + str(snr) + "/Phase-"+str(i)+"test_QSM_000.nii.gz")
				phase = aa.get_data()
				phases.append(phase)
			phases = np.mean(np.array(phases), axis=0)
			print(phases.shape)
			nib.save(nib.Nifti1Image(phases, aa.affine), "/home/francesco/UQ/deepQSMGAN/data/TGVQSM/Sim" + str(sim) + "Snr" + str(snr) + ".nii")


def erodeMEDI():
	basePath = "/scratch/cai/deepQSMGAN/data/realData/"
	basePath = "/home/francesco/UQ/deepQSMGAN/data/realData/"
	maskPath = "cut_phase/"



	for folder in glob.glob(basePath + maskPath + "*"):
		for folder2 in glob.glob(folder + "/*"):
			phaseName = glob.glob(folder2 + "/*scaledTOPPM_medi.nii")
			assert(len(phaseName) == 1)
			aa = nib.load(phaseName[0])
			phase = aa.get_data()
			
			mask = nib.load(folder2 + "/eroded_mask.nii").get_data()
			assert(mask.shape == phase.shape)
			nib.save(nib.Nifti1Image(phase*mask, aa.affine), folder2 + "/MEDI_eroded_mask.nii")


path = "/home/francesco/UQ/deepQSMGAN/data/QSM_Challenge2_download_stage2/DatasetsStep2/"
methodPath = "/home/francesco/UQ/deepQSMGAN/data/deepQSMResGAN/"
#calculateMetricsMethod(path, methodPath)
#calculateMetricDeepQSM()
#saveTrainingData()
#saveNormalized(path)
#prepareInputDeepQSM(path)
#forwardModel()
#splitPhase()
#metricsChallengeOne()
#normalizeMag()
#utils.loadRealData()
mergePhase()
#erodeMEDI()

