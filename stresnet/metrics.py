# -*- coding: utf-8 -*-
import numpy as np
from keras import backend as K
''' mae 和 mre add by lucy, 2018/12/25
'''


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

def mean_relative_error(y_ture, y_pred):
    return K.mean(K.abs(y_pred - y_ture)/y_ture)

def root_mean_square_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

def mre(y_true, y_pred):
    idx = (y_true > 1e-6).nonzero()
    return K.mean(K.abs(y_pred[idx] - y_true[idx])/y_true[idx])

# aliases
mse = MSE = mean_squared_error
# mae=MAE=mean_absolute_error
# mre = MRE = mean_relative_error
# rmse = RMSE = root_mean_square_error


def masked_mean_squared_error(y_true, y_pred):
    idx = (y_true > 1e-6).nonzero()
    return K.mean(K.square(y_pred[idx] - y_true[idx]))


def masked_rmse(y_true, y_pred):
    return masked_mean_squared_error(y_true, y_pred) ** 0.5

def main():
    y_pred =np.array([1, 2, 2, 3])
    y_true =np.array([1, 1, 1, 2])
    print(' y_pred shape: '+ str(y_pred.shape))
    t_mae = mae(y_true, y_pred)
    t_mre = mre(y_true, y_pred)
    t_rmse = rmse(y_true, y_pred)
    print('mae: ' +str(t_mae))
    print('mre: ' + str(t_mre))
    print('rmse: ' + str(t_rmse))


if __name__ == '__main__':
    main()
