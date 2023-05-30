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

import os
import copy
import typing
import math
import numpy as np

from collections import OrderedDict
from typing import List
from mindspore import nn, ops, ParameterTuple
from xt.model.ms_compat import ms, Tensor, Adam, Cell, TrainOneStepCell, FixedLossScaleUpdateCell
from xt.model.model_ms import XTModel_MS, check_keep_model
from xt.model.muzero.default_config import LR, td_step
from xt.model.muzero.muzero_utils_ms import value_compression_ms,\
    value_decompression_ms, cross_entropy_ms, scale_gradient_ms
from zeus.common.util.common import import_config
from xt.model.pb_format import pb_model
from zeus.common.util.register import Registers
from mindspore import set_context
from xt.model.dqn.dqn_cnn_ms import MyTrainOneStepCell
set_context(runtime_num_threads=3)

# pylint: disable=W0201

@Registers.model
class MuzeroModelMS(XTModel_MS):
    """Docstring for ActorNetwork."""

    class InitInferNet(Cell):
        def __init__(self, representation_network, policy_network):
            super(MuzeroModelMS.InitInferNet, self).__init__()
            self.representation_network = representation_network
            self.policy_network = policy_network

        def construct(self, obs):
            hidden = self.representation_network(obs)
            policy, value = self.policy_network(hidden)
            return value, policy, hidden

    class RecurInferNet(Cell):
        def __init__(self, dynamic_network, policy_network):
            super(MuzeroModelMS.RecurInferNet, self).__init__()
            self.dynamic_network = dynamic_network
            self.policy_network = policy_network

        def construct(self, conditioned_hidden):
            hidden, reward = self.dynamic_network(conditioned_hidden)
            policy, value = self.policy_network(hidden)
            return value, reward, policy, hidden

    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)
        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        self.reward_min = model_config.get('reward_min', -300)
        self.reward_max = model_config.get('reward_max', 300)
        self.reward_support_size = math.ceil(
            value_compression_ms(
                self.reward_max - self.reward_min)) + 1
        self.value_min = model_config.get('value_min', 0)
        self.value_max = model_config.get('value_max', 60000)
        self.value_support_size = math.ceil(
            value_compression_ms(
                self.value_max - self.value_min)) + 1
        self.obs_type = model_config.get('obs_type', 'float32')
        self.td_step = td_step
        self.weight_decay = 1e-4
        self.representation_network = self.create_rep_network()
        self.policy_network = self.create_policy_network()
        self.dynamic_network = self.create_dyn_network()
        super().__init__(model_info)
        self.trainable_parameter = self.model.trainable_params()
        self.net_with_loss = NetWithLoss(
            self.model,
            self.model.rnet,
            self.model.pnet,
            self.model.dnet,
            td_step,
            self.action_dim,
            self.weight_decay)
        self.adam = Adam(params=self.trainable_parameter, learning_rate=LR)
        self.init_infer_net = self.InitInferNet(
            self.model.rnet, self.model.pnet)
        self.recur_infer_net = self.RecurInferNet(
            self.model.dnet, self.model.pnet)
        device_target = ms.get_context("device_target")
        if device_target == 'Ascend':
            manager = FixedLossScaleUpdateCell(loss_scale_value=2**14)
            self.train_net = MyTrainOneStepCell(self.net_with_loss, self.adam, manager)
        elif device_target == "GPU" or device_target == "CPU" :
            self.train_net = myTrainOneStepCell(self.net_with_loss, optimizer=self.adam)
        else:
            raise Exception("Target error, GPU or Ascend is supported.")
        super(MuzeroModelMS, self).__init__(model_info)
        self.recur_infer_net.compile(ms.Tensor(np.zeros((1, 260))).astype(ms.float32))
        self.init_infer_net.compile(ms.Tensor(np.zeros((1, 84, 84, 4))).astype(ms.float32))

    def create_model(self, model_info):
        self.full_model = MuzeroBaseMS(self.representation_network,
                                       self.dynamic_network,
                                       self.policy_network)

        return self.full_model

    def initial_inference(self, input_data):
        obs = Tensor.from_numpy(input_data)
        value, policy, hidden = self.init_infer_net(obs)
        hidden = hidden.asnumpy()
        policy = policy.asnumpy()
        value = value.asnumpy()
        value = self.value_transform(
            value[0],
            self.value_support_size,
            self.value_min,
            self.value_max)
        return NetworkOutput(value, 0, policy[0], hidden[0])

    """这里的变量使用还要考虑一下"""

    def recurrent_inference(self, hidden_state, action):
        action = np.expand_dims(np.eye(self.action_dim)[action], 0)
        hidden_state = np.expand_dims(hidden_state, 0)
        conditioned_hidden = np.hstack((hidden_state, action))
        conditioned_hidden = Tensor(conditioned_hidden, ms.float32)
        value, reward, policy, hidden = self.recur_infer_net(
            conditioned_hidden)

        hidden = hidden.asnumpy()
        reward = reward.asnumpy()
        policy = policy.asnumpy()
        value = value.asnumpy()
        value = self.value_transform(
            value[0],
            self.value_support_size,
            self.value_min,
            self.value_max)
        reward = self.value_transform(
            reward[0],
            self.reward_support_size,
            self.reward_min,
            self.reward_max)
        return NetworkOutput(value, reward, policy[0], hidden[0])

    def train(self, state, label):
        target_value = self.conver_value(
            label[0],
            self.value_support_size,
            self.value_min,
            self.value_max)
        target_reward = self.conver_value(
            label[1],
            self.reward_support_size,
            self.reward_min,
            self.reward_max)
        obs = Tensor.from_numpy(state[0])
        action = Tensor.from_numpy(state[1])
        loss_weights = Tensor.from_numpy(state[2])
        target_value = Tensor.from_numpy(target_value)
        target_reward = Tensor.from_numpy(target_reward)
        target_policy = Tensor.from_numpy(label[2])
        loss = self.train_net(
            obs,
            action,
            loss_weights,
            target_value,
            target_reward,
            target_policy).asnumpy()
        return np.mean(loss)

    def get_weights(self):
        """return the weights of the model"""
        _weights = OrderedDict([(par_name, par.data.asnumpy())
                                for par_name, par in
                                self.model.parameters_and_names()])
        return _weights

    def set_weights(self, weights):
        """set the new weights"""
        for _, param in self.model.parameters_and_names():
            if param.name in weights:
                new_param_data = Tensor.from_numpy(copy.deepcopy(weights[param.name]))
                param.set_data(new_param_data, param.sliced)

    def save_model(self, file_name):
        """save weights into .h5 file"""
        # check max model file to keep
        check_keep_model(os.path.dirname(file_name), self.max_to_keep)
        _weights = OrderedDict([(par_name, par.data.asnumpy())
                                for par_name, par in
                                self.model.parameters_and_names()])
        np.savez(file_name + ".h5", **_weights)
        if self.model_format == 'pb':
            pb_model(self.model, file_name)
        return file_name + ".h5"

    def load_model(self, model_name, by_name=False):
        np_file = np.load(model_name)
        weights = OrderedDict(**np_file)
        self.set_weights(weights)

    def conver_value(self, target_value, support_size, min, max):
        # MSE in board games, cross entropy between categorical values in
        # Atari.
        targets = np.zeros(target_value.shape[0:2] + (support_size,))
        target_value = np.clip(target_value, min, max) - min
        batch_size = target_value.shape[0]
        td_size = target_value.shape[1]

        for i in range(batch_size):
            value = value_compression_ms(target_value[i])
            floor_value = np.floor(value).astype(int)
            rest = value - floor_value

            index = floor_value.astype(int)
            targets[i, range(td_size), index] = 1 - rest
            targets[i, range(td_size), index + 1] = rest

        return targets

    def value_transform(self, value_support, support_size, min, max):
        """
        The value is obtained by first computing the expected value
        from the discrete support.
        Second, the inverse transform is then apply (the square function).
        """
        value = np.dot(value_support, range(0, support_size))
        value = value_decompression_ms(value) + min
        value = np.clip(value, min, max)
        return np.asscalar(value)

    def value_inference(self, input_data):
        obs = Tensor.from_numpy(input_data)
        value, _, _ = self.init_infer_net(obs)
        value = value.asnumpy()
        value_list = []
        for value_data in value:
            value_list.append(
                self.value_transform(
                    value_data,
                    self.value_support_size,
                    self.value_min,
                    self.value_max))
        return np.asarray(value_list)


class myTrainOneStepCell(TrainOneStepCell):
    def __init__(self, network, optimizer):
        super(myTrainOneStepCell, self).__init__(network, optimizer)
        self.depend = ops.Depend()
        self.network = network
        self.grad_fn = ops.value_and_grad(
            self.network, grad_position=None, weights=self.weights)

    def construct(self, *inputs):
        loss, grads = self.grad_fn(*inputs)
        grads = self.grad_reducer(grads)
        loss = self.depend(loss, self.optimizer(grads))
        return loss


class NetWithLoss(nn.Cell):
    def __init__(
            self,
            full_model,
            representation_network,
            policy_network,
            dynamic_network,
            td_step,
            action_dim,
            weight_decay):
        super(NetWithLoss, self).__init__(auto_prefix=False)
        self.full_model = full_model
        self.representation_network = representation_network
        self.policy_network = policy_network
        self.dynamic_network = dynamic_network
        self.params = list(self.full_model.parameters_and_names())
        self.on_value, self.off_value = Tensor(1.0, ms.float32),\
                                        Tensor(0.0, ms.float32)
        self.td_step = td_step
        self.action_dim = action_dim
        self.weight_decay = weight_decay
        self.l2_loss = ops.L2Loss()
        self.one_hot = ops.OneHot()
        self.reshape = ops.Reshape()
        self.concat = ops.Concat(-1)

    def construct(
            self,
            obs,
            action,
            loss_weights,
            target_value,
            target_reward,
            target_policy):
        hidden_state = self.representation_network(obs)
        policy_logits, value = self.policy_network(hidden_state)
        loss = cross_entropy_ms(
            policy_logits, target_policy[:, 0], loss_weights)
        loss += cross_entropy_ms(value, target_value[:, 0], loss_weights)
        gradient_scale = 1.0 / self.td_step
        for i in range(self.td_step):
            action_change = self.one_hot(
                action[:, i], self.action_dim, self.on_value, self.off_value)
            action_change = self.reshape(action_change, (-1, self.action_dim,))
            conditioned_state = self.concat((hidden_state, action_change))
            hidden_state, reward = self.dynamic_network(conditioned_state)
            policy_logits, value = self.policy_network(hidden_state)
            hidden_state = scale_gradient_ms(hidden_state, 0.5)
            l = cross_entropy_ms(reward, target_reward[:, i], loss_weights)
            l += cross_entropy_ms(policy_logits,
                                  target_policy[:, i + 1], loss_weights)
            l += cross_entropy_ms(value, target_value[:, i + 1], loss_weights)
            loss += scale_gradient_ms(l, gradient_scale)

        for _, param in self.params:
            loss += self.weight_decay * self.l2_loss(param)
        return loss


class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy: List[int]
    hidden_state: List[float]


class MuzeroBaseMS(Cell):
    """Model that combine the representation and prediction
    (value+policy) network.
    """

    def __init__(
            self,
            representation_network: Cell,
            dynamic_network: Cell,
            policy_network: Cell):
        super().__init__()
        self.representation_network = representation_network
        self.dynamic_network = dynamic_network
        self.policy_network = policy_network

    @property
    def rnet(self):
        return self.representation_network
    @property
    def dnet(self):
        return self.dynamic_network
    @property
    def pnet(self):
        return self.policy_network
