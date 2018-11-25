import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

class ROCCurve(object):
    def __init__(self, directory, title, name):
        self.directory = directory

        self._y_true = np.empty((0, ), dtype=np.int64)
        self._y_score = np.empty((0, ), dtype=np.float32)

        self._title = title
        self._name = name

    @property
    def y_score(self):
        return self._y_score

    @y_score.setter
    def y_score(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self._y_score = data

    @property
    def y_true(self):
        return self._y_true

    @y_true.setter
    def y_true(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self._y_true = data
       
    def append(self, y_true, y_score):
        self._y_true = np.append(self._y_true, y_true)
        self._y_score = np.append(self._y_score, y_score)

    def compute(self):
        fpr, self._tpr, _ = metrics.roc_curve(self._y_true, self._y_score)
        # True negative rate = background rejection
        self._tnr = 1 - fpr
        self._auc = metrics.auc(x=self._tpr, y=self._tnr)

    def _draw(self, tpr, tnr, path):
        fig, ax = plt.subplots(figsize=(16, 12))

        label = 'ROC Curve (AUC = {:0.3f})'.format(self._auc)
        roc_curve = Line2D(
            xdata=tpr, ydata=tnr,
            label=label,
            color='darkorange', lw=3)
        ax.add_line(roc_curve)

        horizontal_line = Line2D(
            xdata=[0, 1], ydata=[1, 1],
            color='navy', lw=2, linestyle='--')
        ax.add_line(horizontal_line)

        vertical_line = Line2D(
            xdata=[1, 1], ydata=[0, 1],
            color='navy', lw=2, linestyle='--')
        ax.add_line(vertical_line)

        ax.set_xlim([0.0, 1.1])
        ax.set_ylim([0.0, 1.1])

        ax.set_xlabel('Singal Efficiency')
        ax.set_ylabel('Background Rejection)')

        ax.set_title(self._title)
        ax.legend(loc='lower left')
        ax.grid()

        fig.savefig(path)
        plt.close()

    def finish(self):
        self.compute()

        name = '{name}_auc-{auc:.3f}.{{ext}}'.format(
            name=self._name,
            auc=self._auc)

        npz_path = os.path.join(self.directory, name.format(ext='npz'))
        plot_path = os.path.join(self.directory, name.format(ext='png'))

        np.savez(file=npz_path,
                 y_true=self._y_true,
                 y_score=self._y_score,
                 tpr=self._tpr,
                 tnr=self._tnr)

        self._draw(tpr=self._tpr, tnr=self._tnr, path=plot_path)
