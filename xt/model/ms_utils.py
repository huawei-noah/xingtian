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
"""Create tf utils for assign weights between learner and actor\
    and model utils for universal usage."""

import numpy as np
from mindspore import nn
import mindspore as ms
import copy
from collections import OrderedDict


class MSVariables:
    def __init__(self, net: nn.Cell) -> None:
        self.net = net

    def get_weights(self) -> OrderedDict:
        _weights = OrderedDict((par_name, par.data.asnumpy())
                               for par_name, par in
                               self.net.parameters_and_names())
        return _weights

    def save_weights(self, save_name: str):
        _weights = OrderedDict((par_name, par.data.asnumpy())
                               for par_name, par in
                               self.net.parameters_and_names())
        np.savez(save_name, **_weights)

    def set_weights(self, to_weights):
        for _, param in self.net.parameters_and_names():
            if param.name in to_weights:
                new_param_data = ms.Tensor(
                    copy.deepcopy(to_weights[param.name]))
                param.set_data(new_param_data, param.sliced)
        return
    def read_weights(weight_file: str):
        """Read weights with numpy.npz"""
        np_file = np.load(weight_file)
        return OrderedDict(**np_file)

    def set_weights_with_npz(self, npz_file: str):
        """Set weight with numpy file."""
        weights = self.read_weights(npz_file)
        self.set_weights(weights)

    def save_weight_with_checkpoint(self, filename: str):
        ms.save_checkpoint(self._weights, filename)

    def load_weight_with_checkpoint(self, filename: str):
        param_dict = ms.load_checkpoint(filename, self.net)
        param_not_load = ms.load_param_into_net(self.net, param_dict)
