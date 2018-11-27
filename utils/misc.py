from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime
import json
import argparse
import numpy as np
import warnings
import glob
import pandas as pd
import logging
import sys
import re


import tensorflow.keras.backend as K
from tensorflow.python.client import device_lib


class Directory(object):
    def __init__(self, path, create=True):
        self._path = path
        self._create = create
        if self._create:
            os.makedirs(self.path)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path_):
        if not isinstance(path_, str):
            raise TypeError
        self._path = path_

    def mkdir(self, name):
        path = os.path.join(self._path, name)
        setattr(self, name, Directory(path, create=self._create))

    def get_entries(self, full_path=True):
        entries = os.listdir(self.path)
        if full_path:
            entries = [os.path.join(self._path, each) for each in entries]
        return entries

    def concat(self, name):
        return os.path.join(self._path, name)


def get_dataset_paths(directory):
    entries = os.listdir(directory)
    datasets = {}
    for each in entries:
        key, _ = os.path.splitext(each)
        datasets[key] = os.path.join(directory, each)

    return datasets


class Config(object):
    def __init__(self, directory, mode="w"):
        self.path = os.path.join(directory, "config.json")
        if mode == "w":
            self.log = {}
        elif mode == "r":
            self.log = self.load(self.path)
            for key, value in self.log.iteritems():
                setattr(self, key, value)
        else:
            raise ValueError("Expected 'r' or 'w' but found '{}'".format(mode))

    def append(self, data):
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
        with open(self.path, "w") as json_file:
            json.dump(self.log, json_file, indent=4, sort_keys=True)

    def load(self, path):
        with open(path) as json_file:
            log = json_file.read()
        log = json.loads(log)
        log = dict(log)
        return log



def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_size_of_model(model):
    return sum([np.prod(K.get_value(w).shape) for w in model.trainable_weights])


def get_filename(path):
    return os.path.splitext(os.path.split(path)[-1])[0]


def is_float(string):
    if not isinstance(string, str):
        raise TypeError
    return re.match("^\d+?\.\d+?$", string) is not None


def convert_str_to_number(string, warning=True):
    if not isinstance(string, str):
        raise TypeError

    if is_float(string):
        return float(string)
    elif string.isdigit():
        return int(string)
    else:
        if warning:
            warnings.warn("'{}' is neither 'float' nor 'int'".format(string),
                          UserWarning)
        return string


def get_logger(name, path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    format_str = '[%(asctime)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(format_str, date_format)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def parse_str(string, target=None):
    data = string.split("_")
    data = [each.split("-") for each in data if "-" in each]
    data = {key: convert_str_to_number(value) for key, value in data}
    if target is None:
        return data
    else:
        return data[target]


def parse_stringized_data(data):
    data = [each.split("-") for each in data]
    data = {key: convert_str_to_number(value, warning=False) for (key, value) in data}
    return data


def find_good_checkpoint(directory,
                    which={"max": ["auc"], "min": ["loss"]},
                    verbose=True,
                    extension=".hdf5"):
    """
    path: '/path/to/directory/<MODEL NAME>_loss-0.2345_<KEY>-<VALUE>.pth.tar'
    """
    def _parse_path(path): 
        basename = os.path.basename(path)
        basename = basename.rstrip(".hdf5")
        metadata = basename.split("_")[1:]
        metadata = [each.split("-") for each in metadata]
        metadata = {key: convert_str_to_number(value, warning=False) for (key, value) in metadata}
        metadata["path"] = path
        return metadata

    entries = glob.glob(os.path.join(directory, "*{}".format(extension)))
    entries = [_parse_path(each) for each in entries if each.endswith(extension)]
    df = pd.DataFrame(entries)

    good_models = []
    for each in which["max"]:
        path = df.loc[df[each].idxmax()]["path"]
        good_models.append(path)
        if verbose:
            print("Max {}: {}".format(each, path))
    for each in which["min"]:
        good_models.append(df.loc[df[each].idxmin()]["path"])
        if verbose:
            print("Min {}: {}".format(each, path))
    good_models = list(set(good_models))
    return good_models        
