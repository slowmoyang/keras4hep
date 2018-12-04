from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Layer


class Gather(Layer):
    def __init__(self, **kwargs):
        super(Gather, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Gather, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        seq, idx = inputs

        # FIXME it seems Keras standardize input shape
        # (batch_size, ) --> (batch_size, 1)
        # check if it is right
        idx = tf.reshape(idx, shape=(-1, ))

        batch_range = tf.range(tf.shape(seq)[0])
        indices = tf.stack([batch_range, idx], axis=1)
        return tf.gather_nd(seq, indices)

    def compute_output_shape(self, input_shape):
        seq_shape = input_shape[0]
        batch_size, _, input_dim = K.int_shape(seq_shape)

        return (batch_size, input_dim)

