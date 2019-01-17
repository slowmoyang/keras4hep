from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import argparse
from datetime import datetime

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks

import keras4hep as kh
from keras4hep.data import DataIterator
from keras4hep.data import get_class_weight
from keras4hep.metrics import roc_auc
from keras4hep.utils.misc import Directory
from keras4hep.utils.misc import Config
from keras4hep.utils.misc import find_good_checkpoint
from keras4hep.projects.qgjets.utils import get_dataset_paths
from keras4hep.projects.toptagging import ROCCurve
from keras4hep.projects.toptagging import BinaryClassifierResponse
from keras4hep.projects.toptagging import LearningCurve


class QGJetsExperiment(object):
    '''
    Experimental feature
    '''
    def __init__(self):
        parser = self.make_argument_parser()
        args = parser.parse_args()
        self.make_log_dir(args)
        self.make_config(args)

        self.class_weight = None

    def make_argument_parser(self):
        '''
        make_argument_parser
        '''
        parser = argparse.ArgumentParser()

        # logs
        default_name = "run_{}".format(datetime.now().strftime("%y%m%d_%H%M%S"))
        parser.add_argument("--name", default=default_name)
        parser.add_argument("--directory", default="./logs")
        parser.add_argument("--keep-all-ckpt", action="store_true")
        parser.add_argument("--num-gpus", default=1, type=int)

        # data
        parser.add_argument("--pt", dest="pt", default=100, type=int)
        parser.add_argument("--batch-size", dest="batch_size", default=128, type=int)
        parser.add_argument("--test-batch-size", dest="test_batch_size", default=256, type=int)

        # optimizer
        parser.add_argument("--optimizer", default="Adam", type=str)
        parser.add_argument("--lr", dest="lr", default=0.001, type=float)
        parser.add_argument("--clipnorm", dest="clipnorm", default=-1, type=float)
        parser.add_argument("--clipvalue", dest="clipvalue", default=-1, type=float)

        # training
        parser.add_argument("--epoch", dest="num_epochs", default=200, type=int)
        parser.add_argument("--use-class-weight", dest="use_class_weight",
                            default=False, action="store_true")

        return parser

    def make_log_dir(self, args):
        '''
        make_log_dir
        '''
        path = os.path.join(args.directory, args.name)

        log_dir = Directory(path=path)
        log_dir.mkdir("checkpoint")
        log_dir.mkdir("learning_curve")

        log_dir.mkdir("roc_curve")
        log_dir.mkdir("model_response")

        self.log_dir = log_dir

    def make_config(self, args):
        '''
        make_config
        '''
        config = Config(self.log_dir.path, "w")
        config.append(args)
        config["hostname"] = os.environ["HOSTNAME"]
        config.save()

        self.config = config

    def backup_script(self, script):
        for each in script:
            shutil.copy2(each, self.log_dir.script.path)

    def get_dataset(self):
        raise NotImplementedError

    def get_data_iter(self,
                      path,
                      batch_size,
                      cycle,
                      fit_generator_mode):
        dataset = self.get_dataset(path)

        data_iter = DataIterator(
            dataset,
            batch_size=batch_size,
            cycle=cycle,
            fit_generator_mode=fit_generator_mode) 

        return data_iter

    def set_data_iter(self):
        dataset_paths = get_dataset_paths(self.config.pt)
        self.config.append(dataset_paths)

        self.train_iter = self.get_data_iter(
            path=dataset_paths["training"],
            batch_size=self.config.batch_size,
            fit_generator_mode=True,
            cycle=True)

        self.valid_iter = self.get_data_iter(
            path=dataset_paths["validation"],
            batch_size=self.config.test_batch_size,
            fit_generator_mode=True,
            cycle=True)

        self.test_iter = self.get_data_iter(
            path=dataset_paths["test"],
            batch_size=self.config.test_batch_size,
            fit_generator_mode=True,
            cycle=False)

    def get_optimizer(self):
        '''
        get_optimizer
        '''
        optim_kwargs = {"lr": self.config.lr}
        if self.config.clipnorm > 0:
            optim_kwargs['clipnorm'] = self.config.clipnorm
        if self.config.clipvalue > 0:
            optim_kwargs['clipvalue'] = self.config.clipvalue

        optimizer = getattr(tf.keras.optimizers, self.config.optimizer)(**optim_kwargs)
        return optimizer

    def build_model(self):
        '''
        build_model
        '''
        raise NotImplementedError

    # TODO
    def compile_model(self):
        '''
        compile_model
        '''
        if self.config.num_gpus > 1:
            self.model = tf.keras.utils.multi_gpu_model(
                model=self.model,
                gpus=self.config.num_gpus)

        loss = "categorical_crossentropy"
        optimizer = self.get_optimizer()
        metric_list = ["accuracy", roc_auc]

        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metric_list)

    def set_callback(self):
        '''
        set_callback
        '''
        ckpt_format_str = "weights_epoch-{epoch:02d}_loss-{val_loss:.4f}_acc-{val_acc:.4f}_auc-{val_roc_auc:.4f}.hdf5"
        ckpt_path = self.log_dir.checkpoint.concat(ckpt_format_str)

        csv_log_path = self.log_dir.concat("log_file.csv")

        learning_curve = LearningCurve(directory=self.log_dir.learning_curve.path)
        learning_curve.book(x="step", y="roc_auc", best="max")
        learning_curve.book(x="step", y="acc", best="max")
        learning_curve.book(x="step", y="loss", best="min")

        callback_list = [
            callbacks.ModelCheckpoint(filepath=ckpt_path),
            callbacks.ReduceLROnPlateau(verbose=1),
            callbacks.CSVLogger(csv_log_path),
            learning_curve,
        ]

        self.callback_list = callback_list

    def train(self):
        '''
        train
        '''
        self.model.fit_generator(
            generator=self.train_iter,
            steps_per_epoch=len(self.train_iter),
            epochs=self.config.num_epochs,
            validation_data=self.valid_iter,
            validation_steps=len(self.valid_iter),
            callbacks=self.callback_list,
            shuffle=True,
            class_weight=self.class_weight)

    def evaluate(self, filepath, custom_objects=None):
        '''
        evaluate
        '''
        model = tf.keras.models.load_model(
            filepath=filepath,
            custom_objects=custom_objects)

        ckpt_name = os.path.basename(filepath).replace(".hdf5", "")
        epoch = kh.utils.misc.parse_str(ckpt_name, target="epoch")
        name = "model_epoch-{:02d}".format(epoch)

        title = "Epoch {}".format(epoch)

        roc_curve = ROCCurve(
            name=name,
            title=title,
            directory=self.log_dir.roc_curve.path)

        model_response = BinaryClassifierResponse(
            name=name,
            title=title,
            directory=self.log_dir.model_response.path)

        ##########################
        # training data
        ###########################
        print("TRAINING SET")
        for_loop_seq = tqdm(self.train_iter) if HAS_TQDM else self.train_iter
        for x, y in for_loop_seq:
            y_true = y[0]

            y_score = model.predict_on_batch(x)
            model_response.append(is_train=True,
                                  y_true=y_true,
                                  y_score=y_score)

        #############################
        # test data
        ########################
        print("TEST SET")
        for_loop_seq = tqdm(self.test_iter) if HAS_TQDM else self.test_iter
        for x, y in for_loop_seq:
            y_true = y[0]

            y_score = model.predict_on_batch(x)
            model_response.append(is_train=False,
                                  y_true=y_true,
                                  y_score=y_score)
            roc_curve.append(y_true=y_true, y_score=y_score)

        roc_curve.finish()
        model_response.finish()

    def run(self):
        '''
        run
        '''
        self.set_data_iter()

        self.build_model()
        self.compile_model()

        ###################
        # Training
        ##################
        self.set_callback()

        if self.config.use_class_weight:
            self.class_weight = get_class_weight(self.train_iter)
            # NumPy array to list
            self.config["class_weight"] = list(self.class_weight)

        self.train()

        ###################
        # Evaluation
        ###################
        self.train_iter.cycle = False

        good_ckpt_condition = {
            "max": ['auc', 'acc'],
            'min': ['loss']
        }

        good_ckpt = find_good_checkpoint(
            self.log_dir.checkpoint.path,
            which=good_ckpt_condition
        )
        if not self.config.keep_all_ckpt:
            all_ckpt = set(self.log_dir.checkpoint.get_entries())
            useless_ckpt = all_ckpt.difference(good_ckpt)
            for each in useless_ckpt:
                os.remove(each)

        for idx, each in enumerate(good_ckpt, 1):
            K.clear_session()
            self.evaluate(each)

        self.config.save()


