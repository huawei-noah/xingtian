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
import typing
from typing import List
import math
import numpy as np

from xt.model.tf_compat import tf
from xt.model.tf_compat import K, Dense, MSE, Model, Sequential, Input, Lambda

from xt.model.model import XTModel, check_keep_model
from xt.model.muzero.default_config import LR, td_step, max_value
from xt.model.muzero.muzero_utils import value_compression, value_decompression, cross_entropy, scale_gradient
from zeus.common.util.common import import_config

from zeus.common.util.register import Registers


# pylint: disable=W0201
@Registers.model
class MuzeroModel(XTModel):
    """Docstring for ActorNetwork."""

    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        self.reward_min = model_config.get('reward_min', -300)
        self.reward_max = model_config.get('reward_max', 300)
        self.reward_support_size = math.ceil(value_compression(self.reward_max - self.reward_min)) + 1
        self.value_min = model_config.get('value_min', 0)
        self.value_max = model_config.get('value_max', 60000)
        self.value_support_size = math.ceil(value_compression(self.value_max - self.value_min)) + 1
        self.obs_type = model_config.get('obs_type', 'float32')

        super(MuzeroModel, self).__init__(model_info)

    def create_model(self, model_info):
        self.td_step = td_step
        self.weight_decay = 1e-4
        self.optimizer = tf.train.AdamOptimizer(LR)

        self.representation_network = self.create_rep_network()
        self.policy_network = self.create_policy_network()
        self.dynamic_network = self.create_dyn_network()

        self.full_model = MuzeroBase(self.representation_network,
                                     self.dynamic_network,
                                     self.policy_network)

        self.build_graph()

        return self.full_model

    def initial_inference(self, input_data):
        with self.graph.as_default():
            K.set_session(self.sess)

            feed_dict = {self.obs: input_data}
            policy, value, hidden = self.sess.run(self.init_infer, feed_dict)
            value = self.value_transform(value[0], self.value_support_size, self.value_min, self.value_max)

        return NetworkOutput(value, 0, policy[0], hidden[0])

    def recurrent_inference(self, hidden_state, action):
        with self.graph.as_default():
            K.set_session(self.sess)
            action = np.expand_dims(np.eye(self.action_dim)[action], 0)
            hidden_state = np.expand_dims(hidden_state, 0)
            conditioned_hidden = np.hstack((hidden_state, action))
            feed_dict = {self.conditioned_hidden: conditioned_hidden}
            hidden, reward, policy, value = self.sess.run(self.rec_infer, feed_dict)

            value = self.value_transform(value[0], self.value_support_size, self.value_min, self.value_max)
            reward = self.value_transform(reward[0], self.reward_support_size, self.reward_min, self.reward_max)

        return NetworkOutput(value, reward, policy[0], hidden[0])

    def build_graph(self):
        self.build_train_graph()
        self.build_infer_graph()
        self.sess.run(tf.initialize_all_variables())

    def build_train_graph(self):
        self.obs = tf.placeholder(self.obs_type, name="obs",
                                  shape=(None, ) + tuple(self.state_dim))
        self.action = tf.placeholder(tf.int32, name="action",
                                     shape=(None, self.td_step))
        target_value_shape = (None, ) + (1 + self.td_step, self.value_support_size)
        self.target_value = tf.placeholder(tf.float32, name="value",
                                           shape=target_value_shape)
        self.target_reward = tf.placeholder(tf.float32, name="reward",
                                            shape=(None, ) + (1 + self.td_step, self.reward_support_size))
        self.target_policy = tf.placeholder(tf.float32, name="policy",
                                            shape=(None, ) + (1 + self.td_step, self.action_dim))
        self.loss_weights = tf.placeholder(tf.float32, name="loss_weights", shape=(None, 1))

        hidden_state = self.representation_network(self.obs)
        policy_logits, value = self.policy_network(hidden_state)

        loss = cross_entropy(policy_logits, self.target_policy[:, 0], self.loss_weights)
        loss += cross_entropy(value, self.target_value[:, 0], self.loss_weights)

        gradient_scale = 1.0 / self.td_step
        for i in range(self.td_step):
            action = tf.one_hot(self.action[:, i], self.action_dim)
            action = tf.reshape(action, (-1, self.action_dim,))
            conditioned_state = tf.concat((hidden_state, action), axis=-1)
            hidden_state, reward = self.dynamic_network(conditioned_state)
            policy_logits, value = self.policy_network(hidden_state)
            hidden_state = scale_gradient(hidden_state, 0.5)

            l = cross_entropy(reward, self.target_reward[:, i], self.loss_weights)
            l += cross_entropy(policy_logits, self.target_policy[:, i + 1], self.loss_weights)
            l += cross_entropy(value, self.target_value[:, i + 1], self.loss_weights)
            loss += scale_gradient(l, gradient_scale)

        for weights in self.full_model.get_weights():
            loss += self.weight_decay * tf.nn.l2_loss(weights)
        self.loss = loss
        self.train_op = self.optimizer.minimize(loss)

    def build_infer_graph(self):
        self.infer_obs = tf.placeholder(tf.float32, name="infer_obs",
                                        shape=(None, ) + tuple(self.state_dim))
        init_infer_h = self.representation_network(self.obs)
        init_infer_p, init_infer_v = self.policy_network(init_infer_h)
        self.init_infer = [init_infer_p, init_infer_v, init_infer_h]

        self.conditioned_hidden = self.dynamic_network.inputs[0]
        rec_infer_h, rec_infer_r = self.dynamic_network(self.conditioned_hidden)
        rec_infer_p, rec_infer_v = self.policy_network(rec_infer_h)
        self.rec_infer = [rec_infer_h, rec_infer_r, rec_infer_p, rec_infer_v]

    def train(self, state, label):
        with self.graph.as_default():
            K.set_session(self.sess)

            target_value = self.conver_value(label[0], self.value_support_size, self.value_min, self.value_max)
            target_reward = self.conver_value(label[1], self.reward_support_size, self.reward_min, self.reward_max)

            feed_dict = {self.obs: state[0],
                         self.action: state[1],
                         self.loss_weights: state[2],
                         self.target_value: target_value,
                         self.target_reward: target_reward,
                         self.target_policy: label[2]}
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict)

            return np.mean(loss)

    def get_weights(self):
        """return the weights of the model"""
        with self.graph.as_default():
            K.set_session(self.sess)
            return self.model.get_weights()

    def set_weights(self, weights):
        """set the new weights"""
        with self.graph.as_default():
            K.set_session(self.sess)
            self.model.set_weights(weights)

    def save_model(self, file_name):
        """save weights into .h5 file"""
        # check max model file to keep
        check_keep_model(os.path.dirname(file_name), self.max_to_keep)

        with self.graph.as_default():
            K.set_session(self.sess)
            self.model.save_weights(file_name + ".h5", overwrite=True)
        if self.model_format == 'pb':
            pb_model(self.model, file_name)
        return file_name + ".h5"

    def load_model(self, model_name, by_name=False):
        with self.graph.as_default():
            K.set_session(self.sess)
            self.model.load_weights(model_name, by_name)

    def conver_value(self, target_value, support_size, min, max):
        # MSE in board games, cross entropy between categorical values in Atari.
        targets = np.zeros(target_value.shape[0:2] + (support_size, ))
        target_value = np.clip(target_value, min, max) - min
        batch_size = target_value.shape[0]
        td_size = target_value.shape[1]

        for i in range(batch_size):
            value = value_compression(target_value[i])
            floor_value = np.floor(value).astype(int)
            rest = value - floor_value

            index = floor_value.astype(int)
            targets[i, range(td_size), index] = 1 - rest
            targets[i, range(td_size), index + 1] = rest

        return targets

    def value_transform(self, value_support, support_size, min, max,):
        """
        The value is obtained by first computing the expected value from the discrete support.
        Second, the inverse transform is then apply (the square function).
        """
        value = np.dot(value_support, range(0, support_size))
        value = value_decompression(value) + min
        value = np.clip(value, min, max)
        return np.asscalar(value)

    def value_inference(self, input_data):
        with self.graph.as_default():
            K.set_session(self.sess)

            feed_dict = {self.obs: input_data}
            policy, value, hidden = self.sess.run(self.init_infer, feed_dict)

            value_list = []
            for value_data in value:
                value_list.append(self.value_transform(value_data, self.value_support_size, self.value_min, self.value_max))

        return np.asarray(value_list)


class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy: List[int]
    hidden_state: List[float]


class MuzeroBase(Model):
    """Model that combine the representation and prediction (value+policy) network."""
    def __init__(self, representation_network: Model, dynamic_network: Model, policy_network: Model):
        super().__init__()
        self.representation_network = representation_network
        self.dynamic_network = dynamic_network
        self.policy_network = policy_network
