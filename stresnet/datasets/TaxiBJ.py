# -*- coding: utf-8 -*-
"""
    load BJ Data from multiple sources as follows:
        meteorologic data
"""
from __future__ import print_function

import os
import pickle
from copy import copy
import numpy as np
import h5py
from stresnet.datasets import load_stdata, stat
from stresnet.preprocessing.minmax_normalization import MinMaxNormalization
from stresnet.preprocessing import remove_incomplete_days, timestamp2vec
from stresnet.datasets.STMatrix import STMatrix
from stresnet.config import Config


# np.random.seed(1337)  # for reproducibility

# parameters
DATAPATH = Config().DATAPATH


def load_holiday(timeslots, fname=os.path.join(DATAPATH, 'TaxiBJ', 'BJ_Holiday.txt')):
    f = open(fname, 'r')
    holidays = f.readlines()
    holidays = set([h.strip() for h in holidays])
    H = np.zeros(len(timeslots))
    for i, slot in enumerate(timeslots):
        if slot[:8] in holidays:
            H[i] = 1
    print(H.sum())
    # print(timeslots[H==1])
    return H[:, None]


def load_meteorol(timeslots, fname=os.path.join(DATAPATH, 'TaxiBJ', 'BJ_Meteorology.h5')):
    '''
    timeslots: the predicted timeslots
    In real-world, we dont have the meteorol data in the predicted timeslot, instead, we use the meteoral at previous timeslots, i.e., slot = predicted_slot - timeslot (you can use predicted meteorol data as well)
    '''
    f = h5py.File(fname, 'r')
    Timeslot = f['date'].value
    WindSpeed = f['WindSpeed'].value
    Weather = f['Weather'].value
    Temperature = f['Temperature'].value
    f.close()

    M = dict()  # map timeslot to index
    for i, slot in enumerate(Timeslot):
        M[slot] = i

    WS = []  # WindSpeed
    WR = []  # Weather
    TE = []  # Temperature
    for slot in timeslots:
        predicted_id = M[slot]
        cur_id = predicted_id - 1
        WS.append(WindSpeed[cur_id])
        WR.append(Weather[cur_id])
        TE.append(Temperature[cur_id])

    WS = np.asarray(WS)
    WR = np.asarray(WR)
    TE = np.asarray(TE)

    # 0-1 scale
    WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
    TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())

    print("shape: ", WS.shape, WR.shape, TE.shape)

    # concatenate all these attributes
    merge_data = np.hstack([WR, WS[:, None], TE[:, None]])

    # print('meger shape:', merge_data.shape)
    return merge_data


def load_data(T=48, nb_flow=2, len_closeness=None, len_period=None, len_trend=None,
              len_test=None, preprocess_name='preprocessing.pkl',
              meta_data=True, meteorol_data=True, holiday_data=True, channel=False, C_in_P =1):
    """
    load all data
    此时是分别将lc, lp, lt 组成序列，对应多分支结构
    channel 是 LSTM类 网络专用， LSTM类 channel True
    C_in_P 假设C_in_P 为2 表示 要考虑 前一天 同一时间段 和 同一时间段的上一个时间段
    """
    assert(len_closeness + len_period + len_trend > 0)
    # load data
    # 13 - 16
    data_all = []
    timestamps_all = list()
    for year in range(13, 17):
        fname = os.path.join(
            DATAPATH, 'TaxiBJ', 'BJ{}_M32x32_T30_InOut.h5'.format(year))
        print("file name: ", fname)
        stat(fname)
        data, timestamps = load_stdata(fname)
        # print(timestamps)
        # remove a certain day which does not have 48 timestamps
        data, timestamps = remove_incomplete_days(data, timestamps, T)
        data = data[:, :nb_flow]
        data[data < 0] = 0.
        data_all.append(data)
        timestamps_all.append(timestamps)
        print("\n")

    # minmax_scale
    data_train = np.vstack(copy(data_all))[:-len_test]
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = [mmn.transform(d) for d in data_all]

    fpkl = open(preprocess_name, 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    XC, XP, XT = [], [], []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is
        # a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(
            len_closeness=len_closeness, len_period=len_period, len_trend=len_trend,channels=channel,C_in_P=C_in_P)
        XC.append(_XC)
        XP.append(_XP)
        XT.append(_XT)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y


    meta_feature = []
    if meta_data:
        # load time feature
        time_feature = timestamp2vec(timestamps_Y)
        meta_feature.append(time_feature)
    if holiday_data:
        # load holiday
        holiday_feature = load_holiday(timestamps_Y)
        meta_feature.append(holiday_feature)
    if meteorol_data:
        # load meteorol data
        meteorol_feature = load_meteorol(timestamps_Y)
        meta_feature.append(meteorol_feature)

    meta_feature = np.hstack(meta_feature) if len(
        meta_feature) > 0 else np.asarray(meta_feature)
    metadata_dim = meta_feature.shape[1] if len(
        meta_feature.shape) > 1 else None
    # if metadata_dim < 1:
    #     metadata_dim = None
    if meta_data and holiday_data and meteorol_data:
        print('time feature:', time_feature.shape, 'holiday feature:', holiday_feature.shape,
              'meteorol feature: ', meteorol_feature.shape, 'mete feature: ', meta_feature.shape)

    XC = np.vstack(XC)
    XP = np.vstack(XP)
    XT = np.vstack(XT)
    Y = np.vstack(Y)
    print("XC shape: ", XC.shape, "XP shape: ", XP.shape,
          "XT shape: ", XT.shape, "Y shape:", Y.shape)

    XC_train, XP_train, XT_train, Y_train = XC[
        :-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
    XC_test, XP_test, XT_test, Y_test = XC[
        -len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
    timestamp_train, timestamp_test = timestamps_Y[
        :-len_test], timestamps_Y[-len_test:]

    X_train = []
    X_test = []
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_train, XP_train, XT_train]):
        if l > 0:
            X_train.append(X_)
    for l, X_ in zip([len_closeness, len_period, len_trend], [XC_test, XP_test, XT_test]):
        if l > 0:
            X_test.append(X_)
    print('train shape:', XC_train.shape, Y_train.shape,
          'test shape: ', XC_test.shape, Y_test.shape)

    if metadata_dim is not None:
        meta_feature_train, meta_feature_test = meta_feature[
            :-len_test], meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)
    for _X in X_train:
        print(_X.shape, )
    print()
    for _X in X_test:
        print(_X.shape, )
    print()
    return X_train, Y_train, X_test, Y_test,  mmn, metadata_dim, timestamp_train, timestamp_test



def load_data_new (T=48, nb_flow=2, len_closeness=None, len_period=None, len_trend=None,
              len_test=None, preprocess_name='preprocessing.pkl',
              meta_data=True, meteorol_data=True, holiday_data=True, channel=False, C_in_P =1):
    """
    load all data
    此时的x ,y =[i - range(lt), i - range(lp), i - range(lc)] , [i]
    也就是说把lc, lp, lt 所对应的序列合在一起， 对应单线结构
	"""
    assert (len_closeness + len_period + len_trend > 0)
    # load data
    # 13 - 16
    data_all = []
    timestamps_all = list()
    for year in range(13, 17):
        fname = os.path.join(
                DATAPATH, 'TaxiBJ', 'BJ{}_M32x32_T30_InOut.h5'.format(year))
        print("file name: ", fname)
        stat(fname)
        data, timestamps = load_stdata(fname)
        # print(timestamps)
        # remove a certain day which does not have 48 timestamps
        data, timestamps = remove_incomplete_days(data, timestamps, T)
        data = data[:, :nb_flow]
        data[data < 0] = 0.
        data_all.append(data)
        timestamps_all.append(timestamps)
        print("\n")

    # minmax_scale
    data_train = np.vstack(copy(data_all))[:-len_test]
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = [mmn.transform(d) for d in data_all]

    fpkl = open(preprocess_name, 'wb')
    for obj in [mmn]:
        pickle.dump(obj, fpkl)
    fpkl.close()

    X = []
    Y = []
    timestamps_Y = []
    for data, timestamps in zip(data_all_mmn, timestamps_all):
        # instance-based dataset --> sequences with format as (X, Y) where X is
        # a sequence of images and Y is an image.
        st = STMatrix(data, timestamps, T, CheckComplete=False)
        _X, _Y, _timestamps_Y = st.create_dataset_new(
            len_closeness=len_closeness, len_period=len_period, len_trend=len_trend,channels=channel, C_in_P=C_in_P)
        X.append(_X)
        Y.append(_Y)
        timestamps_Y += _timestamps_Y

    meta_feature = []
    if meta_data:
        # load time feature
        time_feature = timestamp2vec(timestamps_Y)
        meta_feature.append(time_feature)
    if holiday_data:
        # load holiday
        holiday_feature = load_holiday(timestamps_Y)
        meta_feature.append(holiday_feature)
    if meteorol_data:
        # load meteorol data
        meteorol_feature = load_meteorol(timestamps_Y)
        meta_feature.append(meteorol_feature)

    meta_feature = np.hstack(meta_feature) if len(
            meta_feature) > 0 else np.asarray(meta_feature)
    metadata_dim = meta_feature.shape[1] if len(
            meta_feature.shape) > 1 else None
    # if metadata_dim < 1:
    #     metadata_dim = None
    if meta_data and holiday_data and meteorol_data:
        print('time feature:', time_feature.shape, 'holiday feature:', holiday_feature.shape,
              'meteorol feature: ', meteorol_feature.shape, 'mete feature: ', meta_feature.shape)

    X = np.vstack(X)
    Y = np.vstack(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)

    XS_train, Y_train = X[:-len_test], Y[:-len_test]
    XS_test, Y_test = X[-len_test:], Y[-len_test:]
    timestamp_train, timestamp_test = timestamps_Y[:-len_test], timestamps_Y[-len_test:]

    print('train shape:', XS_train.shape, Y_train.shape,
          'test shape: ', XS_test.shape, Y_test.shape)

    X_train =[]
    X_test = []
    X_train.append(XS_train)
    X_test.append(XS_test)
    if metadata_dim is not None:
        meta_feature_train, meta_feature_test = meta_feature[
            :-len_test], meta_feature[-len_test:]
        X_train.append(meta_feature_train)
        X_test.append(meta_feature_test)

    return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test




def get_data(T=48, nb_flow=2):
    '''
    add by Lucy , 为了得到经过min_max normalize后的所有data 和对应的timestamps
    :param T:
    :param nb_flow:
    :param len_test:
    :return: data_all_mmn , timestamps_all
    '''
    # load data
    # 13 - 16
    len_test = 48*14
    data_all = []
    timestamps_all = list()
    year = 16
    fname = os.path.join(
            DATAPATH, 'TaxiBJ', 'BJ{}_M32x32_T30_InOut.h5'.format(year))
    print("file name: ", fname)
    stat(fname)
    data, timestamps = load_stdata(fname)
    # print(timestamps)
    # remove a certain day which does not have 48 timestamps
    data, timestamps = remove_incomplete_days(data, timestamps, T)
    data = data[:, :nb_flow]
    data[data < 0] = 0.
    data = data[-len_test:]
    timestamps = timestamps[-len_test:]
    print('type data ' +str(type(data)))
    data_all.append(data)
    timestamps_all.append(timestamps)
    print("\n")


    data_train = np.vstack(copy(data_all))
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = [mmn.transform(d) for d in data_all]

    return data_all_mmn, timestamps_all, mmn

def get_all_data(T=48, nb_flow=2, len_test=None):
    data_all = []
    timestamps_all = list()
    for year in range(13, 17):
        fname = os.path.join(
                DATAPATH, 'TaxiBJ', 'BJ{}_M32x32_T30_InOut.h5'.format(year))
        print("file name: ", fname)
        stat(fname)
        data, timestamps = load_stdata(fname)
        # print(timestamps)
        # remove a certain day which does not have 48 timestamps
        data, timestamps = remove_incomplete_days(data, timestamps, T)
        data = data[:, :nb_flow]
        data[data < 0] = 0.
        data_all.append(data)
        timestamps_all.append(timestamps)
        print("\n")

    # minmax_scale
    data_train = np.vstack(copy(data_all))[:-len_test]
    print('train_data shape: ', data_train.shape)
    mmn = MinMaxNormalization()
    mmn.fit(data_train)
    data_all_mmn = [mmn.transform(d) for d in data_all]

    return data_all_mmn, timestamps_all, mmn


def main():
    # T=48
    X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test = load_data(T=48, nb_flow=2,
                                                                                                     len_closeness=3,
                                                                                                     len_period=1,
                                                                                                     len_trend=1,
                                                                                                     len_test=48 * 7,
                                                                                                     preprocess_name='preprocessing.pkl',
                                                                                                     meta_data=True,
                                                                                                     meteorol_data=True,
                                                                                                     holiday_data=True,
                                                                                                     channel=False)
    data_all_mmn, timestamps_all, mmn = get_data()

    print('type data_all_mmn ' + str(type(data_all_mmn)))
    data_all_mmn = np.array(data_all_mmn)
    timestamps_all = np.array(timestamps_all)
    print('data_all_mmn shape ' + str(data_all_mmn.shape))
    print('timestamps_all shape ' +str(timestamps_all.shape))
    print('max ' + str(mmn._max))
    print('min ' + str(mmn._min))


if __name__ == '__main__':
    # X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test=load_data(T=48, nb_flow=2, len_closeness=3, len_period=1, len_trend=1,
    #           len_test=48*7, preprocess_name='preprocessing.pkl',
    #           meta_data=True, meteorol_data=True, holiday_data=True, channel=True)
    main()


