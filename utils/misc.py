from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import json
import argparse
import numpy as np
import warnings

import tensorflow.keras.backend as K
from tensorflow.python.client import device_lib

class Directory(object):
    def __init__(self, path, creation=True):
        self.path = path
        self._creation = creation
        if self._creation:
            os.makedirs(self.path)

    def make_subdir(self, name):
        path = os.path.join(self.path, name)
        setattr(self, name, Directory(path, creation=self._creation))

    def get_entries(self, full_path=True):
        entries = os.listdir(self.path)
        if full_path:
            entries = map(lambda each: os.path.join(self.path, each), entries)
        return entries

def get_log_dir(path, creation=True):
    # mkdir
    log = Directory(path, creation=creation)
    log.make_subdir('validation')
    log.make_subdir('saved_models')
    log.make_subdir('roc')
    log.make_subdir('output_histogram')
    log.make_subdir("heatmap")
    return log

def get_dataset_paths(dpath):
    entries = os.listdir(dpath)
    datasets = {}
    for each in entries:
        key, _ = os.path.splitext(each)
        datasets[key] = os.path.join(dpath, each)

    return datasets


class Config(object):
    def __init__(self, dpath, mode="w"):
        self.path = os.path.join(dpath, "config.json")
        if self._mode == "w":
            self.log = {}
        elif self._mode == "r":
            self.log = self.load(self.path)
            for key, value in self.log.iteritems():
                setattr(self, key, value)
        else:
            raise ValueError("Expected 'r' or 'w' but found '{}'".format(self._mode))


    def update(self, data):
        if isinstance(data, argparse.Namespace):
            data = vars(data)
        self.log.update(data)
        for key, value in data.iteritems():
            setattr(self, key, value)

    def __setitem__(self, key, item):
        self.log[key] = item
        setattr(self, key, item)

    def __getitem__(self, key):
        return self.log[key]

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.log, f, indent=4, sort_keys=True)

    def load(self, path):
        log = open(path).read()
        log = json.loads(log)
        log = dict(log)
        return log

    def finish(self):
        self.save() 


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_size_of_model(model):
    return sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])


def get_filename(path):
    return os.path.splitext(os.path.split(path)[-1])[0]


def convert_str_to_number(string):
    float_case = float(string)
    int_case = int(float_case)
    output = int_case if float_case == int_case else float_case
    return output
