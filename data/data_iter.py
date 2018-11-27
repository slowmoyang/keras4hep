from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import warnings

import numpy as np

import ROOT

# TODO next method with cyclic and random shuffling

class DataIterator(object):
    def __init__(self,
                 dataset,
                 batch_size,
                 fit_generator_input=None, # use {"x": ["x"], "y": ["y"]}
                 fit_generator_mode=False,
                 class_weight=None,
                 cycle=False,
                 shuffle=False,
                 drop_last=True):

        self._dataset = dataset
        self.batch_size = batch_size
        self.fit_generator_input = fit_generator_input

        self._cycle = cycle or fit_generator_mode
        self._shuffle = shuffle
        self._fit_generator_mode = fit_generator_mode
        self._set_mode_of_next()

        self._class_weight = class_weight

        self._drop_last = drop_last



        self._num_examples = len(self._dataset)
        self._start = 0



    def __len__(self):
        # TODO
        if self._cycle:
            warnings.warn(
                "cycle mode... length is # of batches per a epoch",
                Warning)

        num_batches = self._num_examples / self._batch_size
        num_batches = np.floor(num_batches) if self._drop_last else np.ceil(num_batches)
        return int(num_batches)

    def __next__(self):
        batch = self._next()
        if self._fit_generator_mode:
            return self._get_fit_generator_batch(batch)
        else:
            return batch

    # NOTE python2 support
    next = __next__

    def _acyclic_next(self):
        if self._start + 1 >= self._num_examples:
            raise StopIteration

        end = self._start + self._batch_size
        if end >= self._num_examples and self._drop_last:
            raise StopIteration

        slicing = slice(self._start, end)
        self._start = end

        batch = self._dataset[slicing]
        return batch

    def _acyclic_shuffle_next(self):
        if self._start + 1 >= self._num_examples:
            raise StopIteration

        end = self._start + self._batch_size
        if end >= self._num_examples:
            if self._drop_last:
                raise StopIteration
            end = self._num_examples - 1

        slicing = [self._indices.pop() for _ in range(self._start, end)]
        self._start = end

        batch = self._dataset[slicing]
        return batch

    def _cyclic_next(self):
        if self._start < (self._num_examples - 1):
            end = self._start + self._batch_size
            slicing = slice(self._start, end)

            if end <= self._num_examples:
                self._start = end
                batch = self._dataset[slicing]
                return batch
            else:
                batch = self._dataset[slicing]
                self._start = 0
                end = self._batch_size - len(batch)
                batch1 = self._dataset[slice(self._start, end)]
                self._start = end
                batch += batch1
                return batch
        else:
            self._start = 0
            end = self._start + self._batch_size
            batch = self._dataset[slice(self._start, end)]
            self._start = end
            return batch

    def __iter__(self):
        self._start = 0
        self._indices = self._get_shuffled_indices()
        return self

    def _get_shuffled_indices(self):
        indices = np.arange(self._num_examples)
        np.random.shuffle(indices)
        indices = list(indices)
        return indices

    # FIXME Move this method to utils or dataset
    def get_shape(self, key, batch_shape=False):
        shape = self._dataset[:1][key].shape[1:]
        if batch_shape:
            return (self._batch_size, ) + shape
        else:
            return shape 

    def _get_fit_generator_batch(self, batch):
        x = [batch[each] for each in self._fit_generator_input["x"]]
        y = [batch[each] for each in self._fit_generator_input["y"]]
        if self._class_weight is None:
            return (x, y)
        else:
            return (x, y, self._class_weight)

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size):
        if size <= 0:
            raise ValueError
        self._batch_size = size

    @property
    def fit_generator_input(self):
        return self._fit_generator_input

    @fit_generator_input.setter
    def fit_generator_input(self, input_names):
        if input_names is None:
            input_names = {"x": ["x"], "y": ["y"]}
        elif isinstance(input_names, dict):
            for key in ["x", "y"]:
                if not input_names.has_key(key):
                    raise ValueError

                if isinstance(input_names[key], str):
                    input_names[key] = [input_names.pop(key)]
                elif not isinstance(input_names[key], collections.Sequence):
                    raise TypeError
        else:
            # TODO allow sequence
            raise TypeError

        self._fit_generator_input = input_names

    @property
    def fit_generator_mode(self):
        return self._fit_generator_mode

    @fit_generator_mode.setter
    def fit_generator_mode(self, mode):
        if not isinstance(mode, bool):
            raise TypeError
        self._fit_generator_mode = mode
        if mode:
            warnings.warn(
                "fit_generator_mode: cycle=True",
                Warning)
            self.cycle = True

    @property
    def cycle(self):
        return self._cycle

    @cycle.setter
    def cycle(self, cycle_):
        if not isinstance(cycle_, bool):
            raise TypeError
        self._cycle = cycle_
        self._set_mode_of_next()

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle_):
        if not isinstance(shuffle_, bool):
            raise TypeError
        self._shuffle = shuffle_
        self._set_mode_of_next()
       
    def _set_mode_of_next(self):
        if self._cycle:
            self._next = self._cyclic_next
        else:
            if self._shuffle:
                self._indices = self._get_shuffled_indices()
                self._next = self._acyclic_shuffle_next
            else:
                self._next = self._acyclic_next

