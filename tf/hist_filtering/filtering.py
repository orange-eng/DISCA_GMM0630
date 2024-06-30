import numpy as np
import tensorflow as tf
from tqdm import *
import skimage,os


def hist_filtering(X, neighbor_size = 4):
    """
    X: the input data (w,h,l) or (b,w,h,l,c)
    neighbor_size: the neighbor arrays that are counted
    """
    output_index = []

    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=[0, -1])
    X = tf.cast(X, tf.double)

    sum_tf = tf.keras.layers.Conv3D(filters=1, kernel_size=2 * neighbor_size + 1, strides=(1, 1, 1), padding='same',
                                    use_bias=True, kernel_initializer='Ones', bias_initializer='zeros', trainable=False)

    for i in tqdm(np.unique(X.numpy())):
        if i == 0:
            background_index = tf.where(X == 0)
            background_index = background_index.numpy()
            output_index.append(background_index[np.random.randint(background_index.shape[0], \
                                                                   size=min(background_index.shape[0], 10000)),])
            continue

        matrix_label_i = tf.where(X == i, X, 0)
        matrix_neighbor_i = sum_tf(matrix_label_i)
        qts = np.quantile(matrix_neighbor_i, 0.95)
        #matrix_label_i_filtered = tf.where(matrix_neighbor_i >= qts)
        #matrix_label_i_filtered = matrix_label_i_filtered.numpy()
        matrix_label_i_filtered = np.array(np.where(matrix_neighbor_i >= qts)).T 
        output_index.append(matrix_label_i_filtered[np.random.randint(matrix_label_i_filtered.shape[0], \
                                                                      size=min(matrix_label_i_filtered.shape[0],
                                                                               5000)),])
    output_index = np.concatenate(output_index)
    return output_index


def con_hist_filtering(X, scanning_bottom=100, scanning_upper=20000,saving_path = None):
    """
    X: the input data (w,h,l)
    saving_path: saving the path of each labels components
    neighbor_size: the neighbor arrays that are counted
    """
    dic_components_nums = {}
    output_index = []
    for label_i in np.unique(X):
        if label_i == 0:
            #print('remember to delete, label 0!')
            background_index = np.array(np.where(X == label_i)).T
            output_index.append(background_index[np.random.randint(background_index.shape[0], \
                                                size=min(background_index.shape[0], int(0.1*np.prod(X.shape)))),])
            #dic_components_nums[label_i] = components_nums
            continue

        # con_nums doesn't include the background 0
        con_labels, con_nums = skimage.measure.label(np.where(X == label_i, X, 0), return_num=True)
        #print('remember to delete',np.unique(con_labels), con_nums)
        components_nums = []
        for i in range(con_nums):
            components_nums.append(np.sum(con_labels == i+1))
        #
        dic_components_nums[label_i] = components_nums
        #

        qts_bottom = np.min([np.quantile(components_nums, 0.05),scanning_bottom]) # np.quantile(components_nums, 0.25)
        qts_up = np.min([np.quantile(components_nums, 0.85),scanning_upper]) #is the maximum 20000?
        for i in range(con_nums):
            if np.sum(con_labels == i+1) >= qts_bottom and np.sum(con_labels == i+1) <= qts_up:
                output_index.append(np.floor(np.mean(np.array(np.where(con_labels == i+1)).T, axis=0).reshape([1, -1])))
    
    if len(output_index)==0:
        background_index = np.array(np.where(X != 0)).T
        output_index.append(background_index[np.random.randint(background_index.shape[0], \
                                                size=min(background_index.shape[0], 10)),])
    output_index = np.concatenate(output_index)

    if saving_path is not None:
        np.save(saving_path, dic_components_nums)

    return output_index