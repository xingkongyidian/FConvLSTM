import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
import numpy as np


class iLayer(Layer):
    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(iLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        initial_weight_value = np.random.random(input_shape[1:])
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return x * self.W

    def get_output_shape_for(self, input_shape):
        return input_shape

    def compute_output_shape(self, input_shape):
        return input_shape


def main():
    input = Input(shape=(5,))
    dense = Dense(units=4)(input)
    dense2 = Dense(units=3)(dense)
    output = iLayer()(dense2)
    model = Model(input, output)
    model.summary()

if __name__ == '__main__':
    main()