# -*- coding: utf-8 -*-
"""
Usage:
    THEANO_FLAGS="device=gpu0" python exptTaxiBJ.py [number_of_residual_units]
"""
from __future__ import print_function
import os
import sys
import pickle
import time
import h5py
import numpy as np

BASE_DIR = "E:\yhy\python\FConvLSTM_master"
sys.path.append(BASE_DIR)

import stresnet.metrics as metrics
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger

from stresnet.models.FConvLSTM import *
from stresnet.models.A_ConvLSTM import *
from stresnet.config import Config

from stresnet.datasets import TaxiBJ
np.random.seed(1337)  # for reproducibility

DATAPATH = Config().DATAPATH  # data path, you may set your own data path with the global envirmental variable DATAPATH
CACHEDATA = True
ConvLSTM_with_ANN = False

nb_epoch = 100  # number of epoch at training stage
nb_epoch_cont = 100  # number of epoch at training (cont) stage

regularizers_l2=0.0

batch_size = 32  # batch size
T = 48  # number of time intervals in one day
lr = 0.0002  # learning rate # learning rate

# len_closeness = 3 # length of closeness dependent sequence
len_period = 1 # length of peroid dependent sequence
len_trend = 1 # length of trend dependent sequence

if len(sys.argv) == 1:
    print(__doc__)
    sys.exit(-1)
    # nb_residual_unit = 2  # number of residual units
else:
    n_rules = int(sys.argv[1])
    nb_convlstm_unit = int(sys.argv[2]) # number of residual units
    C_in_P = int(sys.argv[3])
    dense_droppout = float(sys.argv[4])
    # dense_droppout1 = float(sys.argv[5])
    len_closeness = int(sys.argv[5])

nb_flow = 2  # there are two types of flows: inflow and outflow
# divide data into two subsets: Train & Test, of which the test set is the
# last 4 weeks
days_test = 7*4
len_test = T*days_test
len_val = 2*len_test
map_height, map_width = 32, 32  # grid size
patience = 4

path_result = os.path.join('RET', 'BJ')
path_model = os.path.join('MODEL', 'BJ')
path_cache = os.path.join('CACHE', 'BJ')
path_log = 'log_BJ'
print('path_result ', path_result)
print('path_model ', path_model)
print('path_cache ', path_cache)

for path in [path_result, path_model, path_log]:
    os.makedirs(path, exist_ok=True)

if CACHEDATA and not os.path.isdir(path_cache):
    os.mkdir(path_cache)

def build_model(external_dim):
    c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height,
              map_width) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None

    model = FConvLSTM(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf, external_dim=external_dim,
                     nb_convlstm_unit=nb_convlstm_unit, droprate=0.0, C_in_P=C_in_P,
                     dense_dropping=dense_droppout, dense_dropping1=0.0, regularizers_l2=regularizers_l2)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam,  metrics=[metrics.rmse, metrics.mae])
    model.summary()

    return model

def build_model_AConvLSTM(external_dim):
    '''
    A_ConvLSTM 中的n_nodes 为 FConvLSTM 中的n_rules 的两倍时，模型复杂度近似（可学习的参数近似）
    :param external_dim:
    :return:
    '''
    c_conf = (len_closeness, nb_flow, map_height,
              map_width) if len_closeness > 0 else None
    p_conf = (len_period, nb_flow, map_height,
              map_width) if len_period > 0 else None
    t_conf = (len_trend, nb_flow, map_height,
              map_width) if len_trend > 0 else None

    model = A_ConvLSTM(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf, external_dim=external_dim,
                      n_nodes=n_rules, nb_convlstm_unit=nb_convlstm_unit, droprate=0.0, C_in_P=C_in_P,
                      dense_dropping=dense_droppout, dense_dropping1=0.0)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse, metrics.mae])
    # model.summary()

    return model


def read_cache(fname):
    mmn = pickle.load(open('preprocessing.pkl', 'rb'))

    f = h5py.File(fname, 'r')
    num = int(f['num'].value)
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in range(num):
        X_train.append(f['X_train_%i' % i].value)
        X_test.append(f['X_test_%i' % i].value)
    Y_train = f['Y_train'].value
    Y_test = f['Y_test'].value
    external_dim = f['external_dim'].value
    timestamp_train = f['T_train'].value
    timestamp_test = f['T_test'].value
    f.close()

    return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test


def cache(fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test):
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('num', data=len(X_train))

    for i, data in enumerate(X_train):
        h5.create_dataset('X_train_%i' % i, data=data)
    # for i, data in enumerate(Y_train):
    for i, data in enumerate(X_test):
        h5.create_dataset('X_test_%i' % i, data=data)
    h5.create_dataset('Y_train', data=Y_train)
    h5.create_dataset('Y_test', data=Y_test)
    external_dim = -1 if external_dim is None else int(external_dim)
    h5.create_dataset('external_dim', data=external_dim)
    h5.create_dataset('T_train', data=timestamp_train)
    h5.create_dataset('T_test', data=timestamp_test)
    h5.close()


def get_input_lists(X_train, X_test):
    # train_inputs_list, test_inputs_list =[], []
    if len(X_train) == 3:
        if (len_closeness == 0):
            lp = X_train[0]
            lp_flat = lp.reshape([-1, len_period * C_in_P * nb_flow * map_width * map_height])
            lt = X_train[1]
            lt_flat = lt.reshape([-1, len_trend * C_in_P * nb_flow * map_width * map_height])
            ext = X_train[2]
            train_inputs_list = [lp_flat, lp, lt_flat, lt, ext]

            lp = X_test[0]
            lt = X_test[1]
            ext = X_test[2]
            lp_flat = lp.reshape([-1, len_period * C_in_P * nb_flow * map_width * map_height])
            lt_flat = lt.reshape([-1, len_trend * C_in_P * nb_flow * map_width * map_height])
            test_inputs_list = [lp_flat, lp, lt_flat, lt, ext]

        elif len_period == 0:
            lc = X_train[0]
            lc_flat = lc.reshape([-1, len_closeness * nb_flow * map_width * map_height])
            lt = X_train[1]
            lt_flat = lt.reshape([-1, len_trend * C_in_P * nb_flow * map_width * map_height])
            ext = X_train[2]
            train_inputs_list = [lc_flat, lc, lt_flat, lt, ext]

            lc = X_test[0]
            lt = X_test[1]
            ext = X_test[2]
            lc_flat = lc.reshape([-1, len_closeness * nb_flow * map_width * map_height])
            lt_flat = lt.reshape([-1, len_trend  * C_in_P * nb_flow * map_width * map_height])
            test_inputs_list = [lc_flat, lc, lt_flat, lt, ext]

        elif len_trend == 0:
            lc = X_train[0]
            lc_flat = lc.reshape([-1, len_closeness * nb_flow * map_height * map_width])
            lp = X_train[1]
            lp_flat = lp.reshape([-1, len_period * C_in_P * nb_flow * map_width * map_height])
            ext = X_train[2]
            train_inputs_list = [lc_flat, lc, lp_flat, lp, ext]

            lc = X_test[0]
            lp = X_test[1]
            ext = X_test[2]
            lc_flat = lc.reshape([-1, len_closeness * nb_flow * map_height * map_width])
            lp_flat = lp.reshape([-1, len_period * C_in_P * nb_flow * map_width * map_height])
            test_inputs_list = [lc_flat, lc, lp_flat, lp, ext]
    else:
        lc = X_train[0]
        lc_flat = lc.reshape([-1, len_closeness * nb_flow * map_height * map_width])
        lp = X_train[1]
        lp_flat = lp.reshape([-1, len_period * C_in_P * nb_flow * map_width * map_height])
        lt = X_train[2]
        lt_flat = lt.reshape([-1, len_trend * C_in_P * nb_flow * map_width * map_height])
        ext = X_train[3]
        train_inputs_list = [lc_flat, lc, lp_flat, lp, lt_flat, lt, ext]

        lc = X_test[0]
        lp = X_test[1]
        lt = X_test[2]
        ext = X_test[3]
        lc_flat = lc.reshape([-1, len_closeness * nb_flow * map_height * map_width])
        lp_flat = lp.reshape([-1, len_period * C_in_P * nb_flow * map_width * map_height])
        lt_flat = lt.reshape([-1, len_trend * C_in_P * nb_flow * map_width * map_height])
        test_inputs_list = [lc_flat, lc, lp_flat, lp, lt_flat, lt, ext]
    return train_inputs_list, test_inputs_list


def get_input_lists1(X_train, X_test):
    ''''''
    # train_inputs_list, test_inputs_list=[],[]
    lc = X_train[0]
    lc_flat = lc.reshape([-1, len_closeness * nb_flow * map_height * map_width])
    lp = X_train[1]
    lp_flat = lp.reshape([-1, len_period * C_in_P * nb_flow * map_width * map_height])
    lt = X_train[2]
    lt_flat = lt.reshape([-1, len_trend * C_in_P * nb_flow * map_width * map_height])
    train_inputs_list = [lc_flat, lc, lp_flat, lp, lt_flat, lt]

    lc = X_test[0]
    lp = X_test[1]
    lt = X_test[2]
    lc_flat = lc.reshape([-1, len_closeness * nb_flow * map_height * map_width])
    lp_flat = lp.reshape([-1, len_period * C_in_P * nb_flow * map_width * map_height])
    lt_flat = lt.reshape([-1, len_trend * C_in_P * nb_flow * map_width * map_height])
    test_inputs_list = [lc_flat, lc, lp_flat, lp, lt_flat, lt]
    return train_inputs_list, test_inputs_list


def main():

        print('loading data...\n')
        ts_initial = time.time()
        print('datapath ', DATAPATH)
        fname = os.path.join(path_cache, 'TaxiBJ_C{}_P{}_T{}_CinP{}.h5'.format(len_closeness, len_period, len_trend, C_in_P))
        print('fname: ', fname)
        if os.path.exists(fname) and CACHEDATA:
            X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = read_cache(fname)
            print("load %s successfully" % fname)
        else:
            X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = TaxiBJ.load_data(
                T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
                preprocess_name='preprocessing.pkl', meta_data=True, meteorol_data=True, holiday_data=True, channel=True, C_in_P=C_in_P)
        # if CACHEDATA:
        #     cache(fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test)

        train_input_list, test_input_list = get_input_lists(X_train, X_test)

        print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
        print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts_initial))

        print('=' * 10)
        print("compiling model...")
        print("**at the first time, it takes a few minites to compile if you use [Theano] as the backend**")

        ts = time.time()
        if ConvLSTM_with_ANN :
            model = build_model_AConvLSTM(external_dim)
            hyperparams_name = 'AConvLSTM_nodes{}.c{}.p{}.t{}.lc{}.conv_unit{}'.format(n_rules, len_closeness,len_period,
                                                                                       len_trend, C_in_P, nb_convlstm_unit)

        else:
            model = build_model(external_dim)
            hyperparams_name = 'FConvLSTM_rules{}.c{}.p{}.t{}.lc{}.conv_unit{}'.format(n_rules, len_closeness, len_period,
                                                                                     len_trend,C_in_P, nb_convlstm_unit)
        fname_param = os.path.join('MODEL', '{}.best.h5'.format(hyperparams_name))
        early_stopping = EarlyStopping(monitor='val_rmse', patience=patience, mode='min')
        model_checkpoint =ModelCheckpoint(fname_param, monitor='val_rmse', verbose=0, save_best_only=True,  mode='min')
        path_log = './log_BJ'
        log_path = os.path.join(path_log, '{}_step1'.format(hyperparams_name))
        tensorboard = TensorBoard(log_dir=log_path, histogram_freq=1)
        csv = CSVLogger(os.path.join(path_result, hyperparams_name + '.csv'), separator=',', append=False)
        print("\nelapsed time (compiling model): %.3f seconds\n" %(time.time() - ts))

        print('=' * 10)
        print("training model...")
        ts = time.time()

        history = model.fit(train_input_list, Y_train,
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            validation_split=0.10,
                            callbacks=[early_stopping, model_checkpoint, tensorboard, csv],
                            verbose=1)
        model.save_weights(os.path.join('MODEL', '{}.h5'.format(hyperparams_name)), overwrite=True)
        pickle.dump((history.history), open(os.path.join(path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
        print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

        f = open('E:\yhy\python\FConvLSTM_master\experiments\FConvLSTM\\result_record\\result1', 'a')
        f.write('\n\n\nnb_epoch={} nb_epoch_cnt={}'.format(nb_epoch, nb_epoch_cont))
        f.write('\nrules=%d lc=%d lp=%d lt=%d nb_convlstm_units=%d C_in_P=%d dense_dropout=%f  regularizers_l2=%f'
                % (n_rules, len_closeness, len_period, len_trend, nb_convlstm_unit, C_in_P, dense_droppout, regularizers_l2))

        print('=' * 10)
        print('evaluating using the model that has the best loss on the valid set')
        model.load_weights(fname_param)
        score = model.evaluate(train_input_list, Y_train, batch_size=32, verbose=0)
        f.write('\nTrain score(rmse):  (norm): %.4f  (real): %.4f' % (score[1], score[1] * (mmn._max - mmn._min) / 2.))
        f.write('\nTrain score(mae):  (norm): %.4f  (real): %.4f' % (score[2], score[2] * (mmn._max - mmn._min) / 2.))

        score = model.evaluate(test_input_list, Y_test, batch_size=32, verbose=0)
        f.write('\nTrain score(rmse):  (norm): %.4f  (real): %.4f' % (score[1], score[1] * (mmn._max - mmn._min) / 2.))
        f.write('\nTrain score(mae):  (norm): %.4f  (real): %.4f' % (score[2], score[2] * (mmn._max - mmn._min) / 2.))
        f.write("\nelapsed time (eval): %.3f seconds\n" % (time.time() - ts))

        print('=' * 10)
        print("training model (cont)...")
        fname_param = os.path.join('MODEL', '{}.cont.best.h5'.format(hyperparams_name))
        log_path = os.path.join(path_log, '{}_step2'.format(hyperparams_name))
        tensorboard = TensorBoard(log_dir=log_path)

        model_checkpoint = ModelCheckpoint(fname_param, monitor='rmse', verbose=0, save_best_only=True, mode='min')
        history = model.fit(train_input_list, Y_train, epochs=nb_epoch_cont, verbose=1, batch_size=batch_size, callbacks=[
            model_checkpoint, tensorboard])
        pickle.dump((history.history), open(os.path.join(
            path_result, '{}.cont.history.pkl'.format(hyperparams_name)), 'wb'))
        model.save(os.path.join(
            'MODEL', '{}_cont.h5'.format(hyperparams_name)), overwrite=True)
        print("\nelapsed time (training cont): %.3f seconds\n" % (time.time() - ts_initial))

        print('=' * 10)
        print('evaluating using the final model')
        score = model.evaluate(train_input_list, Y_train, batch_size=32, verbose=0)
        print('Train score(rmse): (norm): %.4f  (real): %.4f' % (score[1], score[1] * (mmn._max - mmn._min) / 2.))
        f.write('\n train cont')
        f.write('\nTrain score(rmse): (norm): %.4f  (real): %.4f' % (score[1], score[1] * (mmn._max - mmn._min) / 2.))
        f.write('\nTrain score(mae): (norm): %.4f  (real): %.4f' % (score[2], score[2] * (mmn._max - mmn._min) / 2.))

        score = model.evaluate(test_input_list, Y_test, batch_size=32, verbose=0)
        print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
              (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))

        f.write('\nTest score(rmse):(norm): %.4f  (real): %.4f' % (score[1], score[1] * (mmn._max - mmn._min) / 2.))
        f.write('\nTest score(mae):(norm): %.4f  (real): %.4f' % (score[2], score[2] * (mmn._max - mmn._min) / 2.))
        f.write('\ntotal time=%.3f seconds\n'% (time.time() - ts_initial))


if __name__ == '__main__':
    main()
