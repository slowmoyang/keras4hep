from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.utils.generic_utils import get_custom_objects

from keras4hep.layers.gather import Gather
from keras4hep.layers.multi_head_attention import MultiHeadAttention
from keras4hep.layers.multi_head_attention import MultiHeadSelfAttention
from keras4hep.layers.layer_normalization import LayerNormalization
from keras4hep.layers.pos_wise_ffn import PosWiseFFN


_LOCAL_CUSTOM_OBJECTS = [
    Gather,
    MultiHeadAttention,
    MultiHeadSelfAttention,
    LayerNormalization,
    PosWiseFFN,
]

# It returns the global dictionary of names to classes (_GLOBAL_CUSTOM_OBJECTS).
for each in _LOCAL_CUSTOM_OBJECTS:
    name = each.get_class_name()
    get_custom_objects()[name] = each
