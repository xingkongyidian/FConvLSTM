# -*- coding: utf-8 -*-
'''
for FDCNnet to build rule layer (建立模糊神经网络)
'''
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf
from keras.models import load_model

import numpy as np


class RuleLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(RuleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[1]
        self.sigma = self.add_weight(name='sigma', shape=[input_shape[1] * self.output_dim],
                                     initializer='random_normal', trainable=True)
        self.mu = self.add_weight(name='mu', shape=[input_shape[1] * self.output_dim],
                                  initializer='random_normal', trainable=True)
        super(RuleLayer, self).build(input_shape)

    def call(self, input, mask=None):
        # gauss = K.exp(-K.square((K.tile(input, (1, self.output_dim)) - self.mu)) / K.square(self.sigma))
        # reshape = K.reshape(gauss, [-1, self.output_dim, self.input_dim])
        # out_puts= tf.reduce_prod(reshape, axis=2)
        out_puts = tf.reduce_prod(
            K.reshape(
                K.exp(-K.square((K.tile(input, (1, self.output_dim)) - self.mu)) / K.square(self.sigma)),
                (-1, self.output_dim, self.input_dim)), axis=2)  # Rule activations
        return out_puts

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_output_shape_at(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {"output_dim": self.output_dim}
        base_config = super(RuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def main():
    input = Input(shape=(5,))
    output = RuleLayer(4)(input)
    model = Model(input, output)
    model.summary()
    model.save('test_model.h5')
    for layer in model.layers:
        for weight in layer.weights:
            print(weight.name, weight.shape)

    names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = model.get_weights()
    for name, weight in zip(names, weights):
        print(name, weight.shape)
        print(weight)


if __name__ == '__main__':
    main()
