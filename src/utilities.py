#!/usr/bin/env python3

import math
import os

import tensorflow as tf
import numpy as np
import scipy.io
from scipy import ndimage

def norm(data, includemeanstd=True):
    """
    Normalizes the input by subtracting by the mean and dividing by the standard deviation
    :param data: input data to be normalized
    :param includemeanstd: Boolean whether to return the mean and standard deviation
    :return: Normalized data and mean and standard deviation
    """

    mean = np.mean(data)
    std = np.std(data)

# # The static means and standard deviations calculated on 6400 synthetically generated examples.
#     mean = -5.973889335136964e-05
#     std = 0.026221523548747

    # mean and std of clinicial dataset used for paper.
    # mean = -0.000991373
    # std = 0.0759349

    data = (data - mean) / std

    if includemeanstd:
        return data, mean, std

    return data


def generate_one_sim_example(p_dim):
    """
    simulates one example of simulation data with spheres, squares and rectangles introduced in random angles
    :param p_dim:
    :return: one example of simulation data
    """
    np.random.seed()  # re-seed - important when running in parallel-mode.

    # rectangles_total = np.random.randint(80, 120)
    # squares_total = np.random.randint(80, 120)
    # spheres_total = np.random.randint(80, 120)
    #
    rectangles_total = np.random.randint(40, 60)
    squares_total = np.random.randint(40, 60)
    spheres_total = np.random.randint(40, 60)

    ground_truth_data = np.zeros((p_dim, p_dim, p_dim))
    sus_low = -1    # standard deviation of susceptibility values
    sus_high = 1  # standard deviation of susceptibility values
    offset = 0
    shape_size_min_factor = 0.1
    shape_size_max_factor = 0.4

    for shapes in range(rectangles_total):
        susceptibility_value = np.random.uniform(low=sus_low, high=sus_high)
        shape_size_min = np.floor(p_dim * shape_size_min_factor)
        shape_size_max = np.floor(p_dim * shape_size_max_factor)
        random_sizex = np.random.randint(low=shape_size_min, high=shape_size_max)
        random_sizey = np.random.randint(low=shape_size_min, high=shape_size_max)
        random_sizez = np.random.randint(low=shape_size_min, high=shape_size_max)
        x_pos = np.random.randint(0, p_dim)
        y_pos = np.random.randint(0, p_dim)
        z_pos = np.random.randint(0, p_dim)

        x_pos_max = x_pos + random_sizex
        if x_pos_max >= p_dim:
            x_pos_max = p_dim

        y_pos_max = y_pos + random_sizey
        if y_pos_max >= p_dim:
            y_pos_max = p_dim

        z_pos_max = z_pos + random_sizez
        if z_pos_max >= p_dim:
            z_pos_max = p_dim

        rectangle_temp = np.zeros((p_dim, p_dim, p_dim))
        rectangle_temp[x_pos:x_pos_max, y_pos:y_pos_max, z_pos:z_pos_max] = susceptibility_value

        angle = np.random.randint(0, 180)
        rectangle_temp = ndimage.rotate(rectangle_temp, angle=angle, reshape=False, axes=(0, np.random.randint(1, 2)))

        ground_truth_data = ground_truth_data + rectangle_temp

    for shapes in range(squares_total):
        susceptibility_value = np.random.uniform(low=sus_low, high=sus_high)
        shape_size_min = np.floor(p_dim * shape_size_min_factor)
        shape_size_max = np.floor(p_dim * shape_size_max_factor)
        random_size = np.random.randint(shape_size_min, shape_size_max)
        x_pos = np.random.randint(0, p_dim)
        y_pos = np.random.randint(0, p_dim)
        z_pos = np.random.randint(0, p_dim)

        x_pos_max = x_pos + random_size
        if x_pos_max >= p_dim:
            x_pos_max = p_dim

        y_pos_max = y_pos + random_size
        if y_pos_max >= p_dim:
            y_pos_max = p_dim

        z_pos_max = z_pos + random_size
        if z_pos_max >= p_dim:
            z_pos_max = p_dim

        square_temp = np.zeros((p_dim, p_dim, p_dim))
        square_temp[x_pos:x_pos_max, y_pos:y_pos_max, z_pos:z_pos_max] = susceptibility_value

        angle = np.random.randint(0, 180)
        square_temp = ndimage.rotate(square_temp, angle=angle, reshape=False, axes=(0, np.random.randint(1, 2)))

        ground_truth_data = ground_truth_data + square_temp

    def circle(p_shape):
        susceptibility_value = np.random.uniform(low=sus_low, high=sus_high)
        shape_size_min = np.floor(p_shape * shape_size_min_factor/2)
        shape_size_max = np.floor(p_shape * shape_size_max_factor/2)
        random_size = np.random.randint(low=shape_size_min, high=shape_size_max)
        x_pos = np.random.randint(0, p_shape)
        y_pos = np.random.randint(0, p_shape)
        z_pos = np.random.randint(0, p_shape)

        xx, yy, zz = np.mgrid[:p_shape, :p_shape, :p_shape]

        shape_generated = ((xx - x_pos) ** 2 + (yy - y_pos) ** 2 + (zz - z_pos) ** 2) < random_size ** 2
        shape_generated = shape_generated.astype(float)
        shape_generated = shape_generated * susceptibility_value  # multiplies because its boolean (so basically an addition)

        return shape_generated.astype(float)

    for circles in range(spheres_total):
        a = circle(p_dim)
        ground_truth_data = ground_truth_data + a

    return ground_truth_data+offset


def generate_one_sparse_example(p_dim):
    """
    simulates one sparse example
    :param p_dim:
    :return: one example of simulation data
    """
    np.random.seed()  # re-seed - important when running in parallel-mode.

    sus_std = 0.02    # standard deviation of susceptibility values
    shape_size_min_factor = 0.05
    shape_size_max_factor = 0.4

    def circle(dim):
        susceptibility_value = np.random.normal(loc=0.0, scale=sus_std)
        shape_size_min = np.floor(dim * shape_size_min_factor / 2)
        shape_size_max = np.floor(dim * shape_size_max_factor / 2)
        random_size = np.random.randint(shape_size_min, shape_size_max)
        x_pos = np.random.randint(10, dim - 10)
        y_pos = np.random.randint(10, dim - 10)
        z_pos = np.random.randint(10, dim - 10)

        xx, yy, zz = np.mgrid[:dim, :dim, :dim]

        shape_generated = ((xx - x_pos) ** 2 + (yy - y_pos) ** 2 + (zz - z_pos) ** 2) < random_size ** 2
        shape_generated = shape_generated.astype(float)
        shape_generated = shape_generated * susceptibility_value
        # multiplies because its boolean (so basically an addition)

        return shape_generated.astype(float)

    ground_truth_data = circle(p_dim)

    return ground_truth_data


def generate_one_fourier_example(p_dim):
    """
    simulates one fourier example
    :param p_dim:
    :return: one example of simulation data
    """
    np.random.seed()  # re-seed - important when running in parallel-mode.

    sus_low = -0.2    # min of susceptibility values
    sus_high = 0.2   # max deviation of susceptibility values

    susceptibility_value = np.random.uniform(low=sus_low, high=sus_high)
    x_pos, y_pos, z_pos = (np.random.randint(0, p_dim) for i in range(3))

    xf = np.zeros((p_dim, p_dim, p_dim))
    xf[x_pos, y_pos, z_pos] = susceptibility_value

    # Z = ifftn(xf)
    # ground_truth_data = np.real(Z)
    #
    ground_truth_data = xf

    return ground_truth_data


def cut_one_example(p_dim, i, data_gt, data_fw):
    """
    simulates one example of simulation data with spheres, squares and rectangles introduced in random angles
    :param p_dim:
    :return: one example of simulation data
    """
    np.random.seed()  # re-seed - important when running in parallel-mode.
    brainX, brainY, brainZ = data_gt.shape

    randomX = np.random.randint(0, brainX - p_dim)
    randomY = np.random.randint(0, brainY - p_dim)
    randomZ = np.random.randint(0, brainZ - p_dim)

    ground_truth_data = data_gt[randomX:randomX + p_dim, randomY:randomY + p_dim, randomZ:randomZ + p_dim]
    forward_data = data_fw[randomX:randomX + p_dim, randomY:randomY + p_dim, randomZ:randomZ + p_dim]

    return ground_truth_data, forward_data


def sim_pattern_data(p_shape, n_examples):
    """Simulates a set of artificial data samples"""
    sim_patches = np.ndarray(((n_examples,) + p_shape))
    for sample in range(n_examples):
        one_data_matrix = generate_one_sim_example(p_shape, i=1)
        sim_patches[sample, :, :] = one_data_matrix

    return np.asarray(sim_patches, dtype='float32')


def predict_and_save(predict_results, ground_truth, phase_forward, mask, p_shape, name, mean, std):
    '''

    :param predict_results: the prediction object
    :param ground_truth: The ground truth of the prediction - is only used to save it together with the prediction
    :param phase_forward: The input for the network on which the network is to reconstruct the susceptibility map.
    :param mask: The brai mask - for saving purposes only.
    :param p_shape: The shape of the prediction
    :param name: Name of the .mat file
    :param mean: the mean that should be used to denormalize the prediction afterwards
    :param std: the std that should be used to denormalize the prediction afterwards
    :return: saves a .mat file with the predictions and all the other variables listed as input.
    '''
    preds = np.zeros(ground_truth.shape)
    for i, p in enumerate(predict_results):
        print("Prediction %s: %s" % (i + 1, p["images"]))

        prediction = np.reshape(p['images'], (p_shape, p_shape, p_shape))
        preds[i, :, :, :] = prediction

    scipy.io.savemat(name, {'prediction': preds,
                            'ground_truth': ground_truth,
                            'forward': phase_forward,
                            'meanForward': mean,
                            'stdForward': std,
                            'msk': mask})
    return


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


def load_contest_data(p_shape, form):
    """
    Loads either cosmos or phase data, dependent on the 'form' you specify.
    It will always load the middle of a volume if the shape is not 160
    Download the contest data from here and place in source directory:
    https://www.dropbox.com/s/6isiqmqf09ho6vu/20160801_Recon_Challenge_Public.zip?dl=0&file_subpath=%2F20160801_Recon_Challenge_Public
    :param p_shape: shape of the output you want given as a tuple
    :param form: form of data, either 'cosmos' or 'phase'
    :return: data and mask
    """
    mask = scipy.io.loadmat('data_contest/msk.mat')
    mask = mask['msk']

    if form == 'cosmos':
        data = scipy.io.loadmat('data_contest/chi_cosmos.mat')
        data = data['chi_cosmos']

    if form == 'phase':
        data = scipy.io.loadmat('data_contest/phs_tissue.mat')
        data = data['phs_tissue']

    if form == 'chi33':
        data = scipy.io.loadmat('data_contest/chi_33.mat')
        data = data['chi_33']

    if not p_shape[0]==160:
        diff = int(((160 - p_shape)/2)-1)
        data = data[diff:diff+p_shape,diff:diff+p_shape,diff:diff+p_shape]
        mask = mask[diff:diff+p_shape,diff:diff+p_shape,diff:diff+p_shape]

    data = np.expand_dims(data, axis=0)
    return data, mask



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

        return {"x": forward_image}, ground_truth

    def _input_fn():

        dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP').map(_parser)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.repeat(nepochs)
        dataset = dataset.batch(batch)
        # TODO some dataset cache can be made so you can cache on the harddisk

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()

        return features, labels, dataset

    return _input_fn