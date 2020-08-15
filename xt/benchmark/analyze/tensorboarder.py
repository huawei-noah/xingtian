# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE
"""vision by tensorboard
   usage:
   1. tensorboard --logdir=/tmp/.xt_data/tensorboard
   2. and then, open chrome with url: http://YOUR.SERVER.IP:6006

   3. if multi-scaler, you NEED re-run step-1 above!!!  bugs
"""
import os
import subprocess
from datetime import datetime
from time import sleep
import shutil

import numpy as np
import tensorflow as tf
from absl import logging
from xt.model.tf_compat import summary_scalar


def is_board_running(pro_name="tensorboard"):
    """check if process running."""
    cmd = (
        'ps aux | grep "'
        + pro_name
        + '" | grep -v grep | grep -v tail | grep -v keepH5ssAlive'
    )
    try:
        process_num = len(os.popen(cmd).readlines())
        if process_num >= 1:
            return True
        else:
            return False
    except BaseException as err:
        logging.warning("check process failed with {}.".format(err))
        return False


def variable_summaries(var):
    """
    tensorboard utils
    Args:
        var:

    Returns:

    """
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)

        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))

        tf.summary.histogram("histogram", var)


def clean_board_dir(_to_deleta_dir):
    """re-clear tensorboard dir"""
    if os.path.isdir(_to_deleta_dir):
        shutil.rmtree(_to_deleta_dir, ignore_errors=True)
        print("will clean path: {} for board...".format(_to_deleta_dir))
        sleep(0.01)


class Summary(object):
    """
    summary base class, init the tensorflow local graph for tensorboard
    """

    def __init__(self, logdir_root, fixed_path=None):
        """
        :param logdir_root:
        :param fixed_path:
        """
        self._fixed_path = fixed_path  # record for reuse, with different agent
        self._root_path = logdir_root
        self.scalar_dict = dict()
        self.summary_dict = dict()

        if not os.path.isdir(logdir_root):
            os.makedirs(logdir_root)
        if not fixed_path:
            self.logdir = os.path.join(
                logdir_root, datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        else:
            self.logdir = os.path.join(logdir_root, str(fixed_path))

        self.graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        self.writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        if not is_board_running():
            self.vision_call = subprocess.Popen(
                "tensorboard --logdir={}".format(logdir_root), shell=True
            )
        else:
            self.vision_call = None

    @property
    def fix_path(self):
        """
        fix path for different plot
        """
        return self._fixed_path

    @fix_path.setter
    def fix_path(self, path_value):
        """
        fix path for different plot
        :param path_value:
        """
        # fixme: check path, make new ones
        self.logdir = os.path.join(self._root_path, path_value)
        self._fixed_path = path_value

    def add_scalar(self, name, value, index):
        """
        add new scalar to tensorboard file
        :param name: display name
        :param value: y value
        :param index: x value
        """
        with self.graph.as_default():
            if name not in self.scalar_dict:
                self.scalar_dict[name] = tf.placeholder(tf.float32, [])
                self.summary_dict[name] = summary_scalar(
                    name, self.scalar_dict[name]
                )
            scalar_tensor = self.scalar_dict[name]
            summary_tensor = self.summary_dict[name]

            summary = self.sess.run(
                summary_tensor, feed_dict={scalar_tensor: np.float32(value)}
            )
            self.writer.add_summary(summary, index)
            self.writer.flush()

    def __del__(self):
        """ close object """
        self.writer.close()
        self.sess.close()
        
        os.system("pkill tensorboard -g " + str(os.getpgid(0)))
