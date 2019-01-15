'''
stolen from https://github.com/tensorflow/models/blob/master/official/transformer/model/transformer.py
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints


class LayerNormalization(Layer):

    def __init__(self,
                 epsilon=1e-6,
                 scale_initializer='ones',
                 scale_regularizer=None,
                 scale_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):

        self.epsilon = epsilon

        self.scale_initializer = initializers.get(scale_initializer)
        self.scale_regularizer = regularizers.get(scale_regularizer)
        self.scale_constraint = constraints.get(scale_constraint)

        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_dim = input_shape[-1].value
        shape = (input_dim, )

        self.scale = self.add_weight(
            name='scale', 
            shape=shape,
            initializer='uniform',
            trainable=True)

        self.bias = self.add_weight(
            name='bias', 
            shape=shape,
            initializer='uniform',
            trainable=True)

        super(LayerNormalization, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + self.epsilon)
        return norm_x * self.scale + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'epsilon': self.epsilon,
            'scale_initializer': initializers.serialize(self.scale_initializer),
            'scale_regularizer': regularizers.serialize(self.scale_regularizer),
            'scale_constraint':constraints.serialize(self.scale_constraint),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'bias_constraint':constraints.serialize(self.bias_constraint),
        }

        base_config = super(LayerNormalization, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def get_class_name(cls):
        return cls.__name__



