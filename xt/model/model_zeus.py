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
from xt.model.model import XTModel, check_keep_model
from zeus.common.class_factory import ClassFactory, ClassType


class XTModelZeus(XTModel):
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

    def train(self, state, label):
        """Train the model."""
        loss = self.model.train([state], [label])
        return loss

    def predict(self, state):
        """Do predict use the latest model."""
        return self.model.predict(state)

    def save_model(self, file_name):
        check_keep_model(os.path.dirname(file_name), self.max_to_keep)
        return self.model.save(file_name)

    def load_model(self, model_name, by_name=False):
        return self.model.load(model_name, by_name)

    def set_weights(self, weights):
        """Set weight with memory tensor."""
        self.model.set_weights(weights)

    def get_weights(self):
        """Get the weights."""
        return self.model.get_weights()
