import tensorflow as tf
import numpy as np
import ResUNet as network, utilities as utils, data_manager, misc
import os

base_path = os.getcwd()
base_path = "/scratch/cai/QSMResGAN/" if "scratch" in base_path else "/" + "/".join(base_path.split("/")[:-1]) + "/"
ckp_path = "ckp_20191022_1335_shapes_shape64_ex100_2019_08_30/"

#base_path = "/home/"
dataset = data_manager.get_QSM_datasets(base_path + "dataset/", None, "20170327_qsm2016_recon_challenge", "QSM_Challenge2_download_stage2", 64, None, noisy_data=True)

is_train = tf.placeholder(tf.bool, name='is_train')
X_tensor_2017 = tf.placeholder(tf.float32, shape=[None, dataset["qsm_2017"][0]["x"].shape[1], dataset["qsm_2017"][0]["x"].shape[2], dataset["qsm_2017"][0]["x"].shape[3], 1])
X_tensor_2019 = tf.placeholder(tf.float32, shape=[None, dataset["qsm_2019"][0]["x"].shape[1], dataset["qsm_2019"][0]["x"].shape[2], dataset["qsm_2019"][0]["x"].shape[3], 1])
Y_pred_2017 = network.getGenerator(X_tensor_2017)
Y_pred_2019 = network.getGenerator(X_tensor_2019, reuse=True)

print(X_tensor_2017.shape)
print(X_tensor_2019.shape)
print(Y_pred_2017.shape)
print(Y_pred_2019.shape)

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, base_path + ckp_path + "model-metric0")

    for mode in list(dataset.keys()):
        rmse_tot, ddrmse_tot = [], []
        for batch in dataset[mode]:
            if "qsm_2017" in mode:
                pred = sess.run(Y_pred_2017, feed_dict={X_tensor_2017: batch["x"], is_train: False})
            elif "qsm_2019" in mode:
                pred = sess.run(Y_pred_2019, feed_dict={X_tensor_2019: batch["x"], is_train: False})
            else:
                raise NotImplementedError("Mode " + str(mode))

            # Apply mask and compute error
            pred = misc.apply_mask(pred, batch["mask"])
            rmse, ddrmse = utils.computeddRMSE(batch["y"], pred, batch["mask"])
            rmse_tot.append(rmse)
            ddrmse_tot.append(ddrmse)

            misc.save_nii(pred[0, :, :, :, 0], base_path + ckp_path + "test-" + batch["name"].replace('.', ''))

        print(mode + " : RMSE " + str(round(np.array(rmse_tot).mean(), 3)) + " | ddRMSE : " + str(round(np.array(ddrmse_tot).mean(), 3)))
