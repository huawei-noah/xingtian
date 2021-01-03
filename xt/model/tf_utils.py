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
"""Create tf utils for assign weights between learner and actor and model utils for universal usage."""

import os
import numpy as np
from collections import OrderedDict, deque
from absl import logging

from xt.model.tf_compat import tf


def restore_tf_variable(tf_sess, target_paras, model_name):
    """Restore explorer variable with tf.train.checkpoint."""
    reader = tf.train.NewCheckpointReader(model_name)
    var_names = reader.get_variable_to_shape_map().keys()
    result = dict()
    for _name in var_names:
        result[_name] = reader.get_tensor(_name)
        logging.debug("read variable-{} from model file: {}".format(_name, model_name))
    with tf_sess.as_default():  # must work with sess
        for var_key in target_paras:
            try:
                var_key.load(result[var_key.name])
                logging.debug("load {} success".format(var_key.name))
            except BaseException as err:
                raise KeyError("update {} encounter error:{}".format(var_key.name, err))


class TFVariables:
    """Set & Get weights for TF networks with actor's route."""

    def __init__(self, output_op, session):
        """Extract variables, makeup the TFVariables class."""
        self.session = session
        if not isinstance(output_op, (list, tuple)):
            output_op = [output_op]

        track_explored_ops = set(output_op)
        to_process_queue = deque(output_op)
        to_handle_node_list = list()

        # find the dependency variables start with inputs with BFS.
        while len(to_process_queue) != 0:
            tf_object = to_process_queue.popleft()
            if tf_object is None:
                continue

            if hasattr(tf_object, "op"):
                tf_object = tf_object.op
            for input_op in tf_object.inputs:
                if input_op not in track_explored_ops:
                    to_process_queue.append(input_op)
                    track_explored_ops.add(input_op)

            # keep track of explored operations,
            for control in tf_object.control_inputs:
                if control not in track_explored_ops:
                    to_process_queue.append(control)
                    track_explored_ops.add(control)

            # process the op with 'Variable' or 'VarHandle' attribute
            if "VarHandle" in tf_object.node_def.op or "Variable" in tf_object.node_def.op:
                to_handle_node_list.append(tf_object.node_def.name)

        self.node_hub_with_order = OrderedDict()
        # go through whole global variables
        for _val in tf.global_variables():
            if _val.op.node_def.name in to_handle_node_list:
                self.node_hub_with_order[_val.op.node_def.name] = _val

        self._ph, self._to_assign_node_dict = dict(), dict()

        for node_name, variable in self.node_hub_with_order.items():
            self._ph[node_name] = tf.placeholder(variable.value().dtype,
                                                 variable.get_shape().as_list(),
                                                 name="ph_{}".format(node_name))
            self._to_assign_node_dict[node_name] = variable.assign(self._ph[node_name])

        logging.debug("layers_with_order: \n{}".format(self.node_hub_with_order.keys()))

    def get_weights(self):
        """Get weights with dict type."""
        _weights = self.session.run(self.node_hub_with_order)
        return _weights

    def set_weights(self, to_weights):
        """Set weights with dict type."""
        nodes_to_assign = [
            self._to_assign_node_dict[node_name] for node_name in to_weights.keys()
            if node_name in self._to_assign_node_dict
        ]
        # unused_nodes = [_node for _node in to_weights.keys()
        #                 if _node not in self._to_assign_node_dict]
        # assert not unused_nodes, "weights: {} not assign!".format(unused_nodes)

        if not nodes_to_assign:
            print("to_weights: ", to_weights)
            raise KeyError("NO node's weights could assign in self.graph {} vs {}".format(
                self.node_hub_with_order.keys(), to_weights.keys()
            ))

        assign_feed_dict = {
            self._ph[node_name]: value
            for (node_name, value) in to_weights.items() if node_name in self._ph
        }

        self.session.run(
            nodes_to_assign,
            feed_dict=assign_feed_dict,
        )

    def save_weights(self, save_name: str):
        """Save weights with numpy io."""
        _weights = self.session.run(self.node_hub_with_order)
        np.savez(save_name, **_weights)

    @staticmethod
    def read_weights(weight_file: str):
        """Read weights with numpy.npz"""
        np_file = np.load(weight_file)
        return OrderedDict(**np_file)

    def set_weights_with_npz(self, npz_file: str):
        """Set weight with numpy file."""
        weights = self.read_weights(npz_file)
        self.set_weights(weights)


def norm_initializer(std=0.5):
    """Build customized norm initializer."""
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def gelu(x, name=None):
    """
    Implemente OpenAI GPT's gelu activation function.

    See https://arxiv.org/abs/1606.08415 for details.
    """
    with tf.name_scope(name, 'GELU', [x]):
        cdf = 0.5 * (1.0 + tf.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
        x = x * cdf
    return x
