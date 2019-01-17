'''
stolen from https://github.com/tensorflow/models/blob/master/official/transformer/model/transformer.py
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints


class PosWiseFFN(Dropout):

    def __init__(self,
                 filter_size,
                 hidden_size,
                 rate,
                 activation='relu',
                 padding_value=0.0,
                 epsilon=1e-6,
                 noise_shape=None,
                 seed=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 **kwargs):

        self.filter_size = filter_size
        self.hidden_size = hidden_size
        self.activation = activations.get(activation)
        self.padding_value = padding_value
        self.epsilon = epsilon

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        super(PosWiseFFN, self).__init__(
            rate=rate,
            noise_shape=noise_shape,
            seed=seed,
            **kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_dim = input_shape[-1].value

        loop_seq = [
            ('filter', self.filter_size),
            ('hidden', self.hidden_size)
        ]

        for suffix, output_dim in loop_seq: 
            kernel_name = 'kernel_' + suffix
            kernel_shape = (input_dim, output_dim)

            kernel = self.add_weight(
                name=kernel_name,
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                trainable=True)

            setattr(self, kernel_name, kernel)

            bias_name = 'bias_' + suffix
            bias_shape = (output_dim, )

            bias = self.add_weight(
                name=bias_name,
                shape=bias_shape,
                initializer=self.bias_initializer,
                trainable=True)
           
            setattr(self, bias_name, bias)

        super(PosWiseFFN, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, training=None):

        batch_size, length, input_dim = x.shape.as_list()

        # NOTE Get padding
        is_padding_value = tf.equal(x, self.padding_value)
        is_padding = tf.reduce_all(is_padding_value, axis=-1, keepdims=True)
        is_padding = tf.to_float(is_padding)

        pad_mask = tf.reshape(is_padding, [-1])
        non_pad_indices = tf.to_int32(tf.where(pad_mask < self.epsilon))

        # Reshape x to [batch_size*length, hidden_size] to remove padding
        x = tf.reshape(x, [-1, input_dim])
        x = tf.gather_nd(x, indices=non_pad_indices)

        # Reshape x from 2 dimensions to 3 dimensions.
        x.set_shape([None, input_dim])
        x = tf.expand_dims(x, axis=0)

        #
        output = K.dot(x, self.kernel_filter) + self.bias_filter

        if self.activation is not None:
            output = self.activation(output)

        # Dropout
        if 0.0 < self.rate < 1.0:
            noise_shape = self._get_noise_shape(output)

            def dropped_inputs():
                return K.dropout(
                    x=output,
                    level=self.rate,
                    noise_shape=noise_shape,
                    seed=self.seed)

            output = K.in_train_phase(
                dropped_inputs,
                output,
                training=training)

        # Dense
        output = K.dot(output, self.kernel_hidden) + self.bias_hidden
        if self.activation is not None:
            output = self.activation(output)

        output = tf.squeeze(output, axis=0)

        scatter_shape = (-1, self.hidden_size)
        output = tf.scatter_nd(
            indices=non_pad_indices,
            updates=output,
            shape=scatter_shape)

        out_shape = (-1, length, self.hidden_size)
        output = tf.reshape(output, out_shape)

        return output


    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.hidden_size
        return tf.TensorShape(shape)

    def get_config(self):
        config = {
            'filter_size': self.filter_size,
            'hidden_size': self.hidden_size,
            'activation': activations.serialize(self.activation),
            'padding_value': self.padding_value,
            'epsilon': self.epsilon,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint':constraints.serialize(self.kernel_constraint),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'bias_constraint':constraints.serialize(self.bias_constraint),
        }

        base_config = super(PosWiseFFN, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def get_class_name(cls):
        return cls.__name__


def main():
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    x = Input((30, 8))
    h = PosWiseFFN(filter_size=128, hidden_size=32, rate=0.6)(x)

    model = Model(x, h)
    model.summary()


if __name__ == '__main__':
    main()
