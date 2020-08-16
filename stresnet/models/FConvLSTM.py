'''
Fuzzy + ConvLSTM
'''

from __future__ import print_function
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.layers import (
Input,
Activation,
Dense,
Reshape,
ConvLSTM2D,
Dropout,
BatchNormalization,
Flatten
)

from keras.models import Model
from keras.layers.merge import add
from stresnet.models.iLayer import iLayer
from stresnet.models.RuleLayer import RuleLayer
from stresnet.models.FusionLayer import FusionLayer
from keras import regularizers
from keras.utils.vis_utils import plot_model
#regularizers_l2 = 0.000

def bn_ConvLSTM(nb_filter, droprate,regularizers_l2, bn =False ):
    def f(input):
        convlstm=ConvLSTM2D(filters=nb_filter,kernel_size=(3,3), strides=(1,1),
                            padding='same', data_format='channels_first',
                            kernel_regularizer=regularizers.l2(regularizers_l2),
                            dropout=droprate, return_sequences=True)(input)

        convlstm = Activation('relu')(convlstm)
        if (bn):
            convlstm = BatchNormalization(axis=1)(convlstm)

        return convlstm

    return f

def ConvLSTMUnits(nb_filter, droprate, regularizers_l2, repeatations=1):
    def f(input):
        for i in range(repeatations):
            input = bn_ConvLSTM(nb_filter=nb_filter, droprate=droprate, regularizers_l2=regularizers_l2)(input)
        return input
    return f

def FConvLSTM(c_conf=(3, 2, 32, 32), p_conf=(3, 2, 32, 32), t_conf=(3, 2, 32, 32), external_dim=8, n_rules=10,
              nb_convlstm_unit=1,droprate=0.0, C_in_P=1,dense_dropping=0.3, dense_dropping1=0.3, regularizers_l2=0.001):
    '''
    :param c_conf:
    :param p_conf:
    :param t_conf:
    :param external_dim:
    :param n_rules:
    :param nb_convlstm_unit:
    :param dropping_rate:
    :param bn:
    :return:
    '''
    nb_filter = 64
    main_inputs = []
    out_puts = []
    if c_conf is not None:
        len_seq, nb_flow, map_height, map_width = c_conf
        if len_seq > 0:
            output=[]
            rule_input = Input(shape=(len_seq * nb_flow * map_height * map_width,), name='C_rule_input')
            main_inputs.append(rule_input)
            rule_layer = RuleLayer(n_rules, name='rule_layer')(rule_input)
            output.append(rule_layer)

            input = Input(shape=(len_seq, nb_flow, map_height, map_width), name='DC_input')
            main_inputs.append(input)
            convlstm_units = ConvLSTMUnits(nb_filter=nb_filter, droprate=droprate, regularizers_l2=regularizers_l2, repeatations=nb_convlstm_unit)(input)
            convlstm2 = ConvLSTM2D(filters=nb_flow, kernel_size=(3, 3),
                                   padding='same', data_format='channels_first',
                                   return_sequences=False)(convlstm_units)
            convlstm2 = Flatten()(convlstm2)
            output.append(convlstm2)

            fusion_layer = FusionLayer(name='C_fusionlayer')(output)
            acti_fusion = Activation('tanh')(fusion_layer)
            dense_drop = Dropout(rate=dense_dropping)(acti_fusion)
            dense_layer1 = Dense(units=nb_flow * map_height * map_width, activation='tanh', name='C_dense1')(dense_drop)
            dense_drop1 = Dropout(rate=dense_dropping1)(dense_layer1)
            dense_layer2 = Dense(units=nb_flow * map_height * map_width, activation='tanh', name='C_dense2')(dense_drop1)

            out_puts.append(dense_layer2)

    for conf in [p_conf, t_conf]:
        if conf is not None:
            len_seq, nb_flow, map_height, map_width = conf
            if len_seq > 0:
                output = []
                rule_input = Input(shape=(len_seq * C_in_P*nb_flow * map_height * map_width,))
                main_inputs.append(rule_input)
                rule_layer = RuleLayer(n_rules)(rule_input)
                output.append(rule_layer)

                input = Input(shape=(len_seq*C_in_P, nb_flow, map_height, map_width))
                main_inputs.append(input)
                convlstm_units = ConvLSTMUnits(nb_filter=nb_filter, droprate=droprate, regularizers_l2=regularizers_l2, repeatations=nb_convlstm_unit)(input)
                convlstm2 = ConvLSTM2D(filters=nb_flow, kernel_size=(3, 3),
                                       padding='same', data_format='channels_first',
                                       return_sequences=False)(convlstm_units)
                convlstm2 = Flatten()(convlstm2)
                output.append(convlstm2)

                fusion_layer = FusionLayer()(output)
                acti_fusion = Activation('tanh')(fusion_layer)
                dense_drop = Dropout(rate=dense_dropping)(acti_fusion)
                dense_layer1 = Dense(units=nb_flow * map_height * map_width, activation='tanh')(dense_drop)
                dense_drop1 = Dropout(rate=dense_dropping1)(dense_layer1)
                dense_layer2 = Dense(units=nb_flow * map_height * map_width, activation='tanh')(dense_drop1)

                out_puts.append(dense_layer2)

    if len(out_puts) == 1:
        main_output = out_puts[0]
    else:
        new_outputs = []
        for out in out_puts:
            new_outputs.append(iLayer()(out))
        main_output = add(new_outputs, name='CDW_output')

    if external_dim and external_dim > 0:
        input = Input(shape=(external_dim,))
        main_inputs.append(input)
        embedding = Dense(units=10)(input)
        embedding = Activation('tanh')(embedding)
        h1 = Dense(units=nb_flow * map_height * map_width)(embedding)
        activation = Activation('tanh')(h1)
        external_output = activation
        main_output = add([main_output, external_output])
    else:
        print('external_dim: ', external_dim)

    main_output = Activation('tanh')(main_output)
    main_output = Reshape((nb_flow, map_height, map_width), name='final_output')(main_output)
    model = Model(main_inputs, main_output)

    return model

if __name__ == '__main__':

    model1 = FConvLSTM(external_dim=28, nb_convlstm_unit=1)
    model1.summary()