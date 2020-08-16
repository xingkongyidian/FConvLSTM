'''
    ST-ResNet: Deep Spatio-temporal Residual Networks
'''

from __future__ import print_function
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.layers import (
    Input,
    Activation,
    Dense,
    Reshape,
    LSTM
)
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
#from keras.utils.visualize_util import plot
from keras.utils import plot_model
from keras.layers.merge import add
from stresnet.models.iLayer import iLayer

def _shortcut(input, residual):
    #return merge([input, residual], mode='sum')
    return add([input, residual])


#Batch normalization layer,Normalize the activations of the previous layer at each batch
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Convolution2D(padding="same", strides=(1, 1), filters=nb_filter, kernel_size=(nb_row, nb_col) )(activation)
    return f


def _residual_unit(nb_filter, init_subsample=(1, 1)):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3)(residual)
        return _shortcut(input, residual)
    return f

# basic unit of single conv
def _residual_unit1(nb_filter, init_subsample=(1, 1)):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)
        return _shortcut(input, residual)
    return f

def ResUnits(residual_unit, nb_filter, repetations=1):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            input = residual_unit(nb_filter=nb_filter,
                                  init_subsample=init_subsample)(input)
        return input
    return f


def stresnet(c_conf=(3, 2, 32, 32), p_conf=(3, 2, 32, 32), t_conf=(3, 2, 32, 32), external_dim=8, nb_residual_unit=3):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    conf = (len_seq, nb_flow, map_height, map_width)
    external_dim
    '''

    # main input
    main_inputs = []
    outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, nb_flow, map_height, map_width = conf
            input = Input(shape=(nb_flow * len_seq, map_height, map_width))
            main_inputs.append(input)
            # Conv1
           # conv1 = Convolution2D(
                #nb_filter=64, nb_row=3, nb_col=3, border_mode="same")(input)
            # [nb_residual_unit] Residual Units
            conv1 = Convolution2D(padding='same', filters=64, kernel_size=(3, 3))(input)
            residual_output = ResUnits(_residual_unit, nb_filter=64,
                              repetations=nb_residual_unit)(conv1)
            # Conv2
            activation = Activation('relu')(residual_output)
            conv2 = Convolution2D(padding="same", filters=2, kernel_size=(3,3)) (activation)
            outputs.append(conv2)

    # parameter-matrix-based fusion
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        new_outputs = []
        for output in outputs:
            new_outputs.append(iLayer()(output))
        main_output = add(new_outputs)

    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(units=10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(units=nb_flow * map_height * map_width)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((nb_flow, map_height, map_width))(activation)
        main_output = add([main_output, external_output])
    else:
        print('external_dim:', external_dim)

    main_output = Activation('tanh')(main_output)
    model = Model(inputs=main_inputs, outputs=main_output)
    return model

def lstm(nb_steps, nb_features, layers):
    model = Sequential()
    model.add(LSTM(nb_features, input_shape=(nb_steps, nb_features), return_sequences=True))
    for i in range(layers):
        model.add(LSTM(nb_features, return_sequences=True))
    model.add(LSTM(nb_features, return_sequences=False))
    model.add(Dense(nb_features, activation='tanh'))
    return model

def deepst(c_conf=(3, 2, 32, 32), p_conf=(3, 2, 32, 32), t_conf=(3, 2, 32, 32), external_dim=8, nb_conv =3):
    main_inputs = []
    outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, nb_flow, map_height, map_width = conf
            input = Input(shape=(nb_flow * len_seq, map_height, map_width))
            main_inputs.append(input)
            conv1 = Convolution2D(padding='same', filters=64, kernel_size=(3, 3))(input)
            conv1 = Activation('relu')(conv1)
            outputs.append(conv1)

    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        new_outputs = []
        for output in outputs:
            new_outputs.append(iLayer()(output))
        main_output = add(new_outputs)


    conv2 = Convolution2D(padding='same', filters=64, kernel_size =(3,3)) (main_output)
    for i in range(nb_conv):
        conv2 = Activation('relu')(conv2)
        conv2 = Convolution2D(padding='same', filters=64, kernel_size =(3,3))(conv2)
    main_output = Convolution2D(padding='same', filters=2, kernel_size =(3,3))(conv2)

    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(units=10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(units=nb_flow * map_height * map_width)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((nb_flow, map_height, map_width))(activation)
        main_output = add([main_output, external_output])
    else:
        print('external_dim:', external_dim)

    main_output = Activation('tanh')(main_output)
    model = Model(inputs=main_inputs, outputs=main_output)
    return model

if __name__ == '__main__':
    model = stresnet(external_dim=28, nb_residual_unit=12)
    #plot(model, to_file='ST-ResNet.png', show_shapes=True)
    #model.summary()
    # model = lstm(12, 2*32*32, 2)
    model.summary()
    # plot_model(model, to_file='E:\yhy\python\FConvLSTM_master\model_png\ST-ResNet.png', show_shapes=True)
    # model = deepst(external_dim=28, nb_conv=2)
    # model.summary()
    # plot_model(model, to_file='E:\yhy\python\FConvLSTM_master\model_png\DeepST.png', show_shapes=True)