from __future__ import absolute_import

from tensorflow.python.keras.utils.generic_utils import get_custom_objects

from keras4hep.metrics.roc_auc import roc_auc
from keras4hep.metrics import tf_metrics_wrapper


_LOCAL_CUSTOM_OBJECTS = [
    roc_auc,
]

# It returns the global dictionary of names to classes (_GLOBAL_CUSTOM_OBJECTS).
for each in _LOCAL_CUSTOM_OBJECTS:
    key = type(each).__name__
    get_custom_objects()[key] = each
