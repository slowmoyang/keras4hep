from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras.engine.training import Model


class BayesianModel(Model):
    def __init__(self, inputs, outputs, tau, name=None):
        super(BayesianModel, self).__init__(inputs, outputs, name)
        self._tau = tau
        self._inv_tau = np.power(self._tau, -1)

    def stochastic_forward_pass(self, x, num_iter=1000):
        probs = np.array([self.predict_on_batch(x) for _ in xrange(num_iter)])

        pred_mean = np.mean(probs, axis=0)
        pred_var = np.var(probs, axis=0) + self._inv_tau

        return (pred_mean, pred_var)
