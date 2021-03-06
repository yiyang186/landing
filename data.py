import os

import numpy as np
import pandas as pd

from configure import *
from utils import *


def sample_minibatch(data, batch_size=100, training=False):
    """
    Sample minibatch from dataset

    Input:
    - data
    - batch_size:
    - training: Boolean;

    Return:
    - 
    """
    pass

def get_target(filenames, time_range):
    """
    Get maximum VRTG in time_range for every flight from filenames.

    Input:
    - filenames: A list within string, paths of flight files.
    - time_range: A tuple with start time point and end time point, this
      function get the maximum VRTG in time_range.

    Returns:
    - targets: A dictionary mapping flight filenames to maximum VRTG
    """
    start, end = time_range
    targets = np.zeros(len(filenames))

    for n, fname in enumerate(filenames):
        flight_df = pd.read_csv(PATH + fname,
                                names=COLUMNS,
                                usecols=COLUMNS[ACC_VRT],
                                skiprows=start,
                                nrows=end-start)
        targets[n] = flight_df.values.max()
    return targets


def get_data(filenames, time_range):
    """
    Get input data for prediction.

    Input:
    - filenames: A list within string, paths of flight files.
    - time_range: A tuple with start time point and end time point, this
      function get the input data in time_range.

    Returns:
    - data: A Numpy array, of shape (N, T, C).
      - N: the number of flights
      - T: the length of time_range
      - C: COL_NUM, the number of columns all you need. Defined in configure.py
    """
    start, end = time_range
    data = np.zeros((len(filenames), end-start, COL_NUM))

    for n, fname in enumerate(filenames):
        flight_df = pd.read_csv(PATH + fname,
                                names=COLUMNS,
                                skiprows=start,
                                nrows=end-start
                                ).fillna(method='pad'
                                ).fillna(method='bfill')

        arr_env = decompose_wind(flight_df)
        arr_drv = compose_sstick(flight_df, sstick=False)
              
        others      = [ACC_LNG, ACC_LAT, ACC_VRT, PITH, ROLL, *SINGLE_COL]
        data_others = [abs_max(flight_df[COLUMNS[i]].values) for i in others]
        arr_others  = np.r_[data_others].T

        data[n] = np.c_[arr_env, arr_drv, arr_others]
    return data


def split_data(num_train=5000, num_validation=500, num_test=500, seed=None,
               X_time_range=(240, 300), y_time_range=(300, 305)):
    """
    Load the dataset from disk and perform preprocessing, and split dataset to
    train, validation, test part.

    Inputs:
    - num_train: A integer for train set size. Default is 5000.
    - num_validation: A integer for  validation set size. Default is 500.
    - num_test: A integer for test set size. Default is 500.
    - seed: A integer for random seed to keep the same shuffle result.
    - X_time_range: A tuple with start time point and end time point, this
      function get the input data in time_range.
    - y_time_range: A tuple with start time point and end time point, this
      function get the maximum VRTG in time_range.

    Return:
    - splited: A dictionary mapping some strings to parts of splited dataset.
    """
    filenames = np.array(os.listdir(PATH))
    num_read = num_train + num_validation + num_test
    if num_read > len(filenames):
        raise ValueError("No enough data. Reduce numbers.")

    if seed:
        np.random.seed(seed)

    # Shuffle
    indics = np.arange(len(filenames))
    np.random.shuffle(indics)

    # Load dataset
    mask = indics[: num_read]
    filenames = filenames[mask]
    target = get_target(filenames, y_time_range)
    data = get_data(filenames, X_time_range)
    
    # Split dataset
    splited = {}

    splited['X_val'] = data[: num_validation]
    splited['y_val'] = target[: num_validation]
    splited['f_val'] = filenames[: num_validation]

    splited['X_train'] = data[num_validation: -num_test]
    splited['y_train'] = target[num_validation: -num_test]
    splited['f_train'] = filenames[num_validation: -num_test]

    splited['X_test'] = data[-num_test: ]
    splited['y_test'] = target[-num_test: ]
    splited['f_test'] = filenames[-num_test: ]

    return splited