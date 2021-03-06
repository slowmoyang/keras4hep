'''
stolen from https://github.com/tensorflow/models/blob/master/official/transformer/model/attention_layer.py

TODO bias
TODO cache
'''

from __future__ import division

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints


class MultiHeadAttention(Layer):

    def __init__(self,
                 output_dim,
                 num_heads,
                 padding_value=0.0,
                 negative_infinity=-1e9,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        assert output_dim % num_heads == 0

        self.output_dim = output_dim
        self.num_heads = num_heads
        self.padding_value = padding_value
        self.negative_infinity = negative_infinity

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        self.depth = int(output_dim / num_heads)
        self.scale_factor = self.depth ** -0.5

        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list) and len(input_shape) == 3

        input_name = ['key', 'value', 'query']

        # Create a trainable weight variable for this layer.
        for shape, name in zip(input_shape, input_name):
            # FIXME
            # kernel_shape = tf.TensorShape((shape[2], self.output_dim))
            kernel_shape = (shape[2].value, self.output_dim)
            kernel_name = 'kernel_' + name

            kernel = self.add_weight(
                name=kernel_name,
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True)
            setattr(self, kernel_name, kernel)

        self.kernel_attention = self.add_weight(
            name="kernel_attention",
            shape=(self.output_dim, self.output_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True)

        # Make sure to call the `build` method at the end
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs):
        assert isinstance(inputs, list)
        key, value, query = inputs

        key =  K.dot(key, self.kernel_key)
        value =  K.dot(value, self.kernel_value)
        query =  K.dot(query, self.kernel_query)

        key = self.split_heads(key)
        value = self.split_heads(value)
        query = self.split_heads(query)

        query *= self.scale_factor

        logits = tf.matmul(a=query, b=key, transpose_b=True)

        logits += self.get_bias(value)

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
            'negative_infinity': self.negative_infinity,
            'padding_value': self.padding_value,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        })
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def split_heads(self, x):
        with tf.name_scope("split_heads"):
            batch_size, length, self.output_dim = x.shape.as_list()
            # FIXME
            # shape = tf.TensorShape([batch_size, length, self.num_heads, self.depth])
            # x = tf.reshape(tensor=x, shape=shape)
            # x = K.reshape(x, [batch_size, length, self.num_heads, self.depth])
            if batch_size is None:
                batch_size = -1
            target_shape = (batch_size, length, self.num_heads, self.depth)
            x = tf.reshape(tensor=x, shape=target_shape)
            return tf.transpose(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        with tf.name_scope("combine_heads"):
            # FIXME
            # batch_size, _, length, _ = x.shape.as_list()
            length = x.shape.as_list()[2]
            x = tf.transpose(x, perm=[0, 2, 1, 3])

            target_shape = [-1, length, self.output_dim]
            return tf.reshape(tensor=x, shape=target_shape)

    def get_bias(self, x):
        with tf.name_scope("mask"):
            is_padding_value = tf.equal(x, self.padding_value)
            is_padding = tf.reduce_all(is_padding_value, axis=-1, keepdims=True)
            is_padding = tf.to_float(is_padding)
            attention_bias = is_padding * self.negative_infinity
            return attention_bias

    @classmethod
    def get_class_name(cls):
        return cls.__name__


class MultiHeadSelfAttention(MultiHeadAttention):
    """Multiheaded self-attention layer."""
    def build(self, input_shape):
        return super(MultiHeadSelfAttention, self).build([input_shape] * 3)

    def call(self, x):
        return super(MultiHeadSelfAttention, self).call([x] * 3)


