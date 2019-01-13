from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def roc_auc(y_true, y_pred):
    y_true_bool = tf.cast(y_true, dtype=tf.bool)
    y_true_2d = tf.reshape(y_true_bool, [1, -1])

    y_pred_2d = tf.reshape(y_pred, [-1, 1])

    num_pred = tf.shape(y_pred_2d)[0]

    num_thresholds = 200
    kepsilon = 1e-7  # to account for floating point imprecisions
    thresholds = [(i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)]
    thresholds = [0.0 - kepsilon] + thresholds + [1.0 + kepsilon]
    thresholds = tf.constant(thresholds)
    thresholds = tf.expand_dims(thresholds, [1])
    thresholds_tiled = tf.tile(thresholds, tf.stack([1, num_pred]))

    # Tile the y_pred after thresholding them across different thresholds.
    y_pred_tiled = tf.tile(tf.transpose(y_pred_2d), [num_thresholds, 1])
    pred_is_pos = tf.greater(y_pred_tiled, thresholds_tiled)
    pred_is_neg = tf.logical_not(pred_is_pos)

    label_is_pos = tf.tile(y_true_2d, [num_thresholds, 1])
    label_is_neg = tf.logical_not(label_is_pos)

    is_true_positive = tf.logical_and(label_is_pos, pred_is_pos)
    is_true_positive = tf.to_float(is_true_positive)
    true_positive = tf.reduce_sum(is_true_positive, 1)

    is_true_negative = tf.logical_and(label_is_neg, pred_is_neg)
    is_true_negative = tf.to_float(is_true_negative)
    true_negative = tf.reduce_sum(is_true_negative, 1)

    is_false_positive = tf.logical_and(label_is_neg, pred_is_pos)
    is_false_positive = tf.to_float(is_false_positive)
    false_positive = tf.reduce_sum(is_false_positive, 1)

    is_false_negative = tf.logical_and(label_is_pos, pred_is_neg)
    is_false_negative = tf.to_float(is_false_negative)
    false_negative = tf.reduce_sum(is_false_negative, 1)

    epsilon = 1.0e-6
    true_pos_rate = tf.div(true_positive + epsilon,
                           true_positive + false_negative + epsilon)
    true_neg_rate = tf.div(true_negative + epsilon,
                           true_negative + false_positive + epsilon)

    x = true_pos_rate
    y = true_neg_rate
    return tf.reduce_sum(
        tf.multiply(
            x[:num_thresholds - 1] - x[1:],
            (y[:num_thresholds - 1] + y[1:]) / 2.),
            name="roc_auc")




def _test():
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    train_set = np.load("/store/slowmoyang/TopTagging/npz4bdt/toptagging-training.npz") 
    x = train_set["x"]
    y_true = train_set["y"]

    params = {
        "booster": "gbtree",
        "objective": "binary:logistic",
        "base_score": 0.5,
        "eval_metric": ["error", "auc", "logloss"],
        "learning_rate": 0.3,
        "max_depth": 5,
        "n_estimator": 2000,
        "min_split_loss": 0,
        "gpu_id": 0,
        # "tree_method": "exact"
        "max_bin": 16,
        "tree_method": "hist"
        #"tree_method": "gpu_hist"
        
    }

    model = xgb.XGBClassifier(**params)
    model.fit(x, y_true)

    y_score = model.predict_proba(x)[:, 1]

    auc_skl = roc_auc_score(y_true=y_true, y_score=y_score)

    y_true_tensor = tf.convert_to_tensor(y_true)
    y_score_tensor = tf.convert_to_tensor(y_score)
    op_auc_tf = roc_auc(y_true=y_true_tensor, y_pred=y_score_tensor)

    with tf.Session() as sess:
        auc_tf = sess.run(op_auc_tf)

    print("Scikit-Learn: {}".format(auc_skl))
    print("TensorFlow: {}".format(auc_tf))
