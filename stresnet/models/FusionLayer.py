# -*- coding: utf-8 -*-
'''
for FDCNnet to build fusion layer (融合模糊网络和深度网络)
'''
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input
from keras.models import Model
from keras.models import load_model
# from keras.layers import Dense
import numpy as np

class FusionLayer(Layer):

    def __init__(self, **kwargs):
        super(FusionLayer, self).__init__(**kwargs)

    def build(self, inputs_shape):
        f_size = inputs_shape[0][1]
        d_size = inputs_shape[1][1]
        self.Wf = self.add_weight(name='Wf', shape=[f_size, f_size+d_size],
                                  initializer='uniform', trainable=True)
        self.Wd = self.add_weight(name='Wd', shape=[d_size, f_size + d_size],
                                  initializer='uniform', trainable=True)
        self.biases = self.add_weight(name='biases', shape=[1, f_size + d_size],
                                  initializer='zeros', trainable=True)
        super(FusionLayer, self).build(inputs_shape)

    def call(self, inputs,mask=None):
        out_puts = K.dot(inputs[0], self.Wf) + K.dot(inputs[1], self.Wd) + self.biases
        return out_puts


    def compute_output_shape(self, inputs_shape):
        f_size = inputs_shape[0][1]
        d_size = inputs_shape[1][1]
        output_shape = (inputs_shape[0][0], f_size + d_size)
        return output_shape

def main():
    main_inputs=[]
    finput = Input(shape=(4,))
    dinput = Input(shape=(2*32*32,))

    main_inputs.append(finput)
    main_inputs.append(dinput)
    print(len(main_inputs))
    print(main_inputs[0])
    print(main_inputs[1])
    out_put = FusionLayer()(main_inputs)

    model = Model(main_inputs, out_put)
    model.summary()
    model.save('fusion_layer_testmodel.h5')

if __name__ == '__main__':
    main()
    model =load_model('fusion_layer_testmodel.h5', {'FusionLayer':FusionLayer})
    print('load model successfully')