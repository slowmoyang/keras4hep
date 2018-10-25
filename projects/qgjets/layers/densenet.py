"""
ref. arXiv:1608.06993 [cs.CV]
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import Layer

from keras4hep.projects.qgjets.layers import layer_utils
from keras4hep.projects.qgjets.layers.layer_utils import conv_unit


"""
composite_fn is conv_unit(order=["bn", "activation", "conv3x3"])
"""

def bottleneck_layer(x, filters, activation="relu", **kargs):
    """
    Args:
      x: 
      filters: 'int', growth_rate.
      activation: 'str', Default is relu.

    Returns:
      ''

    It has been noted in [36, 11] that a 1X1 convolution
    can be introduced as bottleneck layer before each 3X3
    convolution to reduce the number of input feature-maps,
    and thus to improve computational efficiency. We find
    this design especially effective for DenseNet and we
    refer to our network with such a bottleneck layer, i.e.,
    to the BN-ReLU-Conv(1X1)-BN-ReLU-Conv(3X3) version of
    H_l(.), as DenseNet-B. In our experiments, we let each
    1X1 convolution produce 4k feature-maps
    """        
    x = conv_unit(
        x,
        filters=4*filters,
        kernel_size=1,
        activation=activation,
        order=["bn", "activation", "conv"],
        **kargs)

    x = conv_unit(
        x,
        filters=filters,
        kernel_size=3,
        activation=activation,
        order=["bn", "activation", "conv"],
        **kargs)

    return x

def transition_layer(x, theta):
    """
    Args:
        x:
        theta:

    Returns:

    Raises:
        ValueError

    The transition layers used in out experiments consist
    of a batch normalization layer and an 1x1 convolutional
    layer followed by a 2X2 average pooling layer.

    To further improve model compactness, we can reduce the
    number of feature-maps at transition layers. If a dense
    block contains m feature maps, we let the following
    transition layer generate theta*m output feature maps,
    where 0 < theta <= 1 is referred to as the compression
    factor.
    """
    if (theta < 0) and (theta > 1):
        raise ValueError

    channel_axis = layer_utils.get_channel_axis()
    in_channels = K.int_shape(x)[channel_axis]

    out_channels = int(in_channels * theta)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Conv2D(filters=out_channels, kernel_size=1, strides=1)(x) 
    x = AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)

    return x


def dense_block(x,
                num_layers,
                growth_rate=12,
                use_bottleneck=True,
                activation="relu",
                **kargs):
    channel_axis = layer_utils.get_channel_axis()

    for i in range(num_layers):
        if use_bottleneck:
            x_i = bottleneck_layer(
                x,
                filters=growth_rate,
                activation=activation,
                **kargs)
        else:
            x_i = conv_unit(
                x=x,
                filters=growth_rate,
                kernel_size=(3, 3),
                activation=activation,
                order=["bn", "activation", "conv"],
                **kargs)
                            
        x = Concatenate(axis=channel_axis)([x, x_i])

    return x

def _test():
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.utils import plot_model

    inputs = Input((3, 224, 224))

    growth_rate = 12
    theta = 0.5
    num_layers_list = [4, 4, 4]

    # Before entering the first dense block, a convolution
    # with 16 (or twice the growth rate for DenseNet-BC)
    # output channels is performed on the input images.
    out = Conv2D(filters=2*growth_rate, kernel_size=5, strides=2)(inputs)
    for i, n in enumerate(num_layers_list[:-1]):
        out = dense_block(out, num_layers=n, growth_rate=growth_rate)
        out = transition_layer(out, theta)

    out = dense_block(out, num_layers=num_layers_list[-1], growth_rate=growth_rate)

    out = BatchNormalization(axis=1)(out)
    out = Activation("relu")(out)
    out = GlobalAveragePooling2D()(out)
    out = Dense(units=2)(out)

    model = Model(inputs=inputs, outputs=out)

    plot_model(model, to_file="/tmp/densenet.png", show_shapes=True)  
    return model
