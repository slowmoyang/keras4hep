'''
stolen from https://github.com/tensorflow/models/blob/master/official/transformer/model/attention_layer.py
'''

from __future__ import division

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers


class MultiHeadAttention(layers.Layer):

    def __init__(self,
                 output_dim,
                 num_heads,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 **kwargs):
        assert output_dim % num_heads == 0

        self.output_dim = output_dim
        self.num_heads = num_heads
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.depth = int(output_dim / num_heads)
        self.scale_factor = self.depth ** -0.5

        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        input_name = ['key', 'value', 'query']

        # Create a trainable weight variable for this layer.
        for shape, name in zip(input_shape, input_name):
            kernel_shape = tf.TensorShape((shape[2], self.output_dim))
            kernel_name = 'kernel_' + name

            kernel = self.add_weight(
                name=kernel_name,
                shape=kernel_shape,
                initializer='uniform',
                trainable=True)
            setattr(self, kernel_name, kernel)

        self.kernel_attention = self.add_weight(
            name="kernel_attention",
            shape=(self.output_dim, self.output_dim),
            initializer='uniform',
            trainable=True)

        # Make sure to call the `build` method at the end
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs):
        key, value, query = inputs

        key =  K.dot(key, self.kernel_key)
        value =  K.dot(value, self.kernel_value)
        query =  K.dot(query, self.kernel_query)

        key = self.split_heads(key)
        value = self.split_heads(value)
        query = self.split_heads(query)

        query *= self.scale_factor

        logits = tf.matmul(a=query, b=key, transpose_b=True)
        weights = tf.nn.softmax(logits)

        attention_output = tf.matmul(weights, value)
        attention_output = self.combine_heads(attention_output)
        attention_output = K.dot(attention_output, self.kernel_attention)

        return attention_output

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MultiHeadAttention, self).get_config()

        base_config.update({
            'output_dim': self.output_dim,
            'num_heads': self.num_heads,
            'kernel_initializer': self.kernel_initializer,
            'kernel_regularizer': self.kernel_regularizer
        })
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def split_heads(self, x):
        with tf.name_scope("split_heads"):
            batch_size, length, self.output_dim = x.shape.as_list()
            x = tf.reshape(x, [batch_size, length, self.num_heads, self.depth])
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        with tf.name_scope("combine_heads"):
            batch_size, _, length, _ = x.shape.as_list()
            x = tf.transpose(x, perm=[0, 2, 1, 3])
            return tf.reshape(tensor=x, shape=[batch_size, length, self.output_dim])



class MultiHeadSelfAttention(MultiHeadAttention):
  """Multiheaded self-attention layer."""

  def call(self, x):
    return super(MultiHeadSelfAttention, self).call([x, x, x])
