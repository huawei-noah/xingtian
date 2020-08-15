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
"""dqn algorithm."""
import os
import numpy as np

from xt.algorithm import Algorithm
from xt.algorithm.dqn.default_config import BUFFER_SIZE, GAMMA, TARGET_UPDATE_FREQ, BATCH_SIZE
from xt.algorithm.replay_buffer import ReplayBuffer
from xt.framework.register import Registers
from xt.model import model_builder
from xt.util.common import import_config

os.environ["KERAS_BACKEND"] = "tensorflow"


@Registers.algorithm
class DQN(Algorithm):
    """Deep Q learning algorithm.
    """

    def __init__(self, model_info, alg_config, **kwargs):
        """Initialize DQN algorithm. it's contains four steps:
        1. override the default config, with user's configuration;
        2. create the default actor with Algorithm.__init__;
        3. create once more actor, named by target_actor;
        4. create the replay buffer for training.
        :param model_info:
        :param alg_config:
        """
        import_config(globals(), alg_config)
        model_info = model_info["actor"]
        super(DQN, self).__init__(
            alg_name="dqn", model_info=model_info, alg_config=alg_config
        )

        self.target_actor = model_builder(model_info)
        self.buff = ReplayBuffer(BUFFER_SIZE)

    def train(self, **kwargs):
        """
        train process for DQN algorithm.
        1. predict the newest state with actor & target actor;
        2. calculate TD error;
        3. train operation;
        4. update target actor if need.
        :return: loss of this train step.
        """
        batch_size = BATCH_SIZE

        batch = self.buff.get_batch(batch_size)
        states = np.asarray([e[0] for e in batch])
        actions = np.asarray([e[1] for e in batch])
        rewards = np.asarray([e[2] for e in batch])
        new_states = np.asarray([e[3] for e in batch])
        dones = np.asarray([e[4] for e in batch])
        y_t = self.actor.predict(states)
        target_q_values = self.target_actor.predict(new_states)
        max_q_val = np.max(target_q_values, 1)
        for k in range(len(batch)):
            if dones[k]:
                q_value = rewards[k]
            else:
                q_value = rewards[k] + GAMMA * max_q_val[k]
            y_t[k][actions[k]] = q_value

        loss = self.actor.train(states, y_t)

        self.train_count += 1
        if self.train_count % TARGET_UPDATE_FREQ == 0:
            self.update_target()

        return loss

    def restore(self, model_name, model_weights=None):
        """
        DQN will restore two model weights, actor & target actor
        :param model_name:
        :param model_weights:
        :return:
        """
        self.actor.load_model(model_name)
        self.target_actor.load_model(model_name)

    def prepare_data(self, train_data, **kwargs):
        """
        prepare the train data for dqn,
        here, just add once new data into replay buffer.
        :param train_data:
        :return:
        """
        buff = self.buff
        data_len = len(train_data["done"])
        for index in range(data_len):
            data = (
                train_data["cur_state"][index],
                train_data["action"][index],
                train_data["reward"][index],
                train_data["next_state"][index],
                train_data["done"][index],
            )
            buff.add(data)  # Add replay buffer

    def update_target(self):
        """
        sync the actor's weight to target.

        :return:
        """
        weights = self.actor.get_weights()
        self.target_actor.set_weights(weights)
