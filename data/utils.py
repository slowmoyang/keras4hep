from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import ROOT

from keras4hep.data.dataset import BaseTreeDataset 
from keras4hep.data.data_iter import DataIterator

def get_batch_time(data_iter, num_iter=10, verbose=True):
    elapsed_times = []
    for _ in range(num_iter):
        start_time = time.time()
        data_iter.next()
        elapsed_times.append(time.time() - start_time)
    elapsed_times = np.array(elapsed_times)
    if verbose:
        print("[Elapsed time] mean: {mean:.5f} / stddev {stddev:5f}".format(
            mean=elapsed_times.mean(),
            stddev=elapsed_times.std()))
        print("Batch Size: {}".format(data_iter.batch_size))
        print("Iteration: {}".format(num_iter))
    return elapsed_times


def get_class_weight(source, class_weight="balanced", label="label",
                     tree_name=None, verbose=True):
    if verbose:
        print("Start to compute class weight")

    if isinstance(source, BaseTreeDataset):
        tree = source._tree
    elif isinstance(source, DataIterator):
        tree = source._dataset._tree
    elif isinstance(source, ROOT.TTree):
        tree = source
    elif isinstance(source, str):
        root_file = ROOT.TFile(source)
        tree = root_file.Get(tree_name)
    else:
        tree = None

    if tree is not None:
        y = source
    else:
        y = [getattr(entry, label) for entry in tree]

    classes = np.unique(y)

    class_weight = compute_class_weight(class_weight=class_weight, classes=classes, y=y)

    if verbose:
        print("Finish off computing class weight")
        for i, w in enumerate(class_weight):
            print("{} th class: {:.4f}".format(i, w))

    return class_weight
            

 
