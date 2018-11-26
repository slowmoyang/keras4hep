from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import ROOT
from tensorflow.keras.callbacks import Callback

# TODO smooth option
# TODO best point marker color and size

class LearningCurve(Callback):
    def __init__(self, directory, booking_list=[]):
        super(LearningCurve, self).__init__()
        self._directory = directory

        self._train_csv_path = os.path.join(directory, "training.csv")
        self._valid_csv_path = os.path.join(directory, "validation.csv")
        self._path_format_str = os.path.join(directory, "{name}.{ext}")

        self._train = []
        self._valid = []

        self._epoch = 0
        self._global_step = 0

        self._booking_list = booking_list

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        self._train = pd.DataFrame(self._train)
        self._valid = pd.DataFrame(self._valid)

        self._train.to_csv(self._train_csv_path)
        self._valid.to_csv(self._valid_csv_path)

        for x, y, best in self._booking_list:
            self._draw(x, y, best)

    def on_batch_begin(self, batch, logs=None):
        self._global_step += 1

    def on_batch_end(self, batch, logs=None):
        logs["step"] = self._global_step
        logs["epoch"] = self._epoch

        self._train.append(logs)

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch += 1

    def on_epoch_end(self, epoch, logs=None):
        """
        logs contains
            metrics on training
            metrics on validation
        """

        my_logs = {}

        for key, value in logs.iteritems():
            if key.startswith("val_"):
                key = key.replace("val_", "")
                my_logs[key] = value
            

        my_logs["step"] = self._global_step
        my_logs["epoch"] = self._epoch

        self._valid.append(my_logs)

    def book(self, x, y, best):
        self._booking_list.append([x, y, best])

    def _draw(self, x, y, best):
        x_train = self._train[x].values
        y_train = self._train[y].values

        x_valid = self._valid[x].values
        y_valid = self._valid[y].values



        # NOTE LOWESS (locally weighted scatterplot smoothing)
        # https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
        smooth_train_curve = lowess(
            endog=y_train,
            exog=x_train,        
            frac=0.075,
            it=0,
            is_sorted=True)

        fig, ax = plt.subplots(figsize=(12, 8))

        # NOTE plt.plot interface doesn't work well..
        train_curve = Line2D(xdata=x_train, ydata=y_train,
                label="Training",
                color="navy", alpha=0.33, lw=2)
        ax.add_line(train_curve)

        smooth_train_curve = Line2D(
            xdata=smooth_train_curve[:, 0],
            ydata=smooth_train_curve[:, 1],
            label="Training (LOWESS)",
            color="navy", lw=3)
        ax.add_line(smooth_train_curve)

        valid_curve = Line2D(
            xdata=x_valid,
            ydata=y_valid,
            label="Validation",
            color="orange",
            ls="--", lw=3,
            marker="^", markersize=10)
        ax.add_line(valid_curve)

        if best == "max":
            index = np.argmax(y_valid)
        else: # min
            index = np.argmin(y_valid)

        x_best = x_valid[index]
        y_best = y_valid[index]
        best_epoch = self._valid["epoch"][index]

        ax.text(x=x_best, y=y_best,
                s="{:.4f} @ Epoch {:d}".format(y_best, best_epoch),
                color="black")

        vertical_line = Line2D(xdata=[x_best, x_best],
                               ydata=[0, y_best],
                               color="lightcoral", alpha=0.3, ls=":", lw=3)
        ax.add_line(vertical_line)

        horizontal_line = Line2D(xdata=[0, x_best],
                                  ydata=[y_best, y_best],
                                  color="lightcoral", alpha=0.3, ls=":", lw=3)
        ax.add_line(horizontal_line)
        


        ax.set_xlabel(x, fontdict={"size": 20})
        ax.set_ylabel(y, fontdict={"size": 20})
        ax.legend(fontsize=20)

        ax.autoscale()

        ax.grid()

        fig.savefig(self._path_format_str.format(name=y, ext='png'))
        fig.savefig(self._path_format_str.format(name=y, ext='png'))
