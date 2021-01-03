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
# THE SOFTWARE.
"""Model base."""

import os
import glob
from xt.model.tf_compat import tf, K, get_sess_graph
from xt.model.pb_format import pb_model

os.environ["KERAS_BACKEND"] = "tensorflow"


class XTModel(object):
    """
    Model Base class for model module.

    Owing to the same name to Keras.Model, set `XTModel` as the base class.
    User could inherit the XTModel, to implement their model.
    """

    def __init__(self, model_info):
        """
        Initialize XingTian Model.

        To avoid the compatibility problems about tensorflow's versions.
        Model class will hold their graph&session within itself.
        Now, we used the keras's API to create models.
        :param model_info:
        """
        sess, self.graph = get_sess_graph()
        # User Could assign it within create model.
        self.actor_var = None
        self._summary = model_info.get("summary", False)

        with self.graph.as_default():
            # init sess within the graph without assign the graph into sess.
            self.sess = sess
            K.set_session(self.sess)
            self.model_format = model_info.get('model_format')
            self.max_to_keep = model_info.get("max_to_keep", 100)
            self.model = self.create_model(model_info)
            if 'init_weights' in model_info:
                model_name = model_info['init_weights']
                try:
                    self.load_model(model_name)
                    print("load weight: {} success.".format(model_name))
                except BaseException:
                    print("load weight: {} failed!".format(model_name))

    def create_model(self, model_info):
        """Abstract method for creating model."""
        raise NotImplementedError

    def predict(self, state):
        """
        Do predict use the newest model.

        :param state:
        :return: output tensor ref to policy.model
        """
        with self.graph.as_default():
            K.set_session(self.sess)
            return self.model.predict(state)

    def train(self, state, label):
        """Train the model."""
        with self.graph.as_default():
            K.set_session(self.sess)
            loss = self.model.train_on_batch(state, label)
            return loss

    def set_weights(self, weights):
        """Set weight with memory tensor."""
        with self.graph.as_default():
            self.actor_var.set_weights(weights)

    def get_weights(self):
        """Get the weights."""
        with self.graph.as_default():
            return self.actor_var.get_weights()

    def get_grad(self, data):
        with self.graph.as_default():
            K.set_session(self.sess)
            self.model.get_grad(data)

    def save_model(self, file_name):
        """Save weights into .h5 file."""
        # check max model file to keep
        if self.max_to_keep > -1:
            check_keep_model(os.path.dirname(file_name), self.max_to_keep)

        if self.actor_var:
            self.actor_var.save_weights(file_name + ".npz")
        else:
            with self.graph.as_default():  # keras
                K.set_session(self.sess)
                self.model.save_weights(file_name)

        if self.model_format == 'pb':
            pb_model(self.model, file_name)
        return file_name + ".npz"

    def load_model(self, model_name):
        if self.actor_var:
            self.actor_var.set_weights_with_npz(model_name)
        else:
            with self.graph.as_default():
                K.set_session(self.sess)
                self.model.load_weights(model_name, by_name=True)


def check_keep_model(model_path, keep_num):
    """Check model saved count under path."""
    target_file = glob.glob(os.path.join(model_path, "actor*".format(model_path)))
    if len(target_file) > keep_num:
        to_rm_model = sorted(target_file, reverse=True)[keep_num:]
        for item in to_rm_model:
            os.remove(item)
