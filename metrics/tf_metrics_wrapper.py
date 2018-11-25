"""
borrowed fromthe following links
https://stackoverflow.com/questions/45947351/how-to-use-tensorflow-metrics-in-keras/50527423#50527423
https://github.com/keras-team/keras/issues/6050#issuecomment-385541045
"""
import functools
import tensorflow as tf
import tensorflow.keras.backend as K

def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

@as_keras_metric
def streaming_roc_auc(y_true, y_pred):
    return tf.metrics.auc(labels=y_true, predictions=y_pred, curve="ROC")
