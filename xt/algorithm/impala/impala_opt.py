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
"""Bulid optimized impala algorithm by merging the data process and inferencing into tf.graph."""

import os
import threading

import numpy as np

from xt.algorithm import Algorithm
from xt.algorithm.impala.default_config import BATCH_SIZE
from zeus.common.ipc.uni_comm import UniComm
from zeus.common.util.register import Registers
from xt.model.tf_compat import loss_to_val
from zeus.common.util.common import import_config
from xt.algorithm.alg_utils import DivideDistPolicy, FIFODistPolicy, EqualDistPolicy


@Registers.algorithm
class IMPALAOpt(Algorithm):
    """Build IMPALA algorithm."""

    def __init__(self, model_info, alg_config, **kwargs):
        import_config(globals(), alg_config)
        super().__init__(alg_name="impala",
                         model_info=model_info["actor"],
                         alg_config=alg_config)
        self.states = list()
        self.behavior_logits = list()
        self.actions = list()
        self.dones = list()
        self.rewards = list()
        self.async_flag = False

        # update to divide model policy
        self.dist_model_policy = FIFODistPolicy(
            alg_config["instance_num"],
            prepare_times=self._prepare_times_per_train)

        self.use_train_thread = False
        if self.use_train_thread:
            self.send_train = UniComm("LocalMsg")
            train_thread = threading.Thread(target=self._train_thread)
            train_thread.setDaemon(True)
            train_thread.start()

    def _train_thread(self):
        while True:
            data = self.send_train.recv()
            batch_state, batch_logit, batch_action, batch_done, batch_reward = data
            actor_loss = self.actor.train(
                batch_state,
                [batch_logit, batch_action, batch_done, batch_reward],
            )

    def train(self, **kwargs):
        """Train impala agent by calling tf.sess."""
        states = np.concatenate(self.states)
        behavior_logits = np.concatenate(self.behavior_logits)
        actions = np.concatenate(self.actions)
        dones = np.concatenate(self.dones)
        rewards = np.concatenate(self.rewards)

        nbatch = len(states)
        count = (nbatch + BATCH_SIZE - 1) // BATCH_SIZE
        loss_list = []

        for start in range(count):
            start_index = start * BATCH_SIZE
            env_index = start_index + BATCH_SIZE
            batch_state = states[start_index:env_index]
            batch_logit = behavior_logits[start_index:env_index]
            batch_action = actions[start_index:env_index]
            batch_done = dones[start_index:env_index]
            batch_reward = rewards[start_index:env_index]

            actor_loss = self.actor.train(
                batch_state,
                [batch_logit, batch_action, batch_done, batch_reward],
            )
            loss_list.append(loss_to_val(actor_loss))

        # clear states for next iter
        self.states.clear()
        self.behavior_logits.clear()
        self.actions.clear()
        self.dones.clear()
        self.rewards.clear()
        return np.mean(loss_list)

    def save(self, model_path, model_index):
        """Save model."""
        actor_name = "actor" + str(model_index).zfill(5)
        actor_name = self.actor.save_model(os.path.join(model_path, actor_name))
        actor_name = actor_name.split("/")[-1]

        return [actor_name]

    def prepare_data(self, train_data, **kwargs):
        """Prepare the data for impala algorithm."""
        state, logit, action, done, reward = self._data_proc(train_data)
        self.states.append(state)
        self.behavior_logits.append(logit)
        self.actions.append(action)
        self.dones.append(done)
        self.rewards.append(reward)

    def predict(self, state):
        """Predict with actor inference operation."""
        pred = self.actor.predict(state)

        return pred

    @staticmethod
    def _data_proc(episode_data):
        """
        Process data for impala.

        Agent will record the follows:
            states, behavior_logits, actions, dones, rewards
        """
        states = episode_data["cur_state"]

        behavior_logits = episode_data["logit"]
        actions = episode_data["action"]
        dones = np.asarray(episode_data["done"], dtype=np.bool)

        rewards = np.asarray(episode_data["reward"])

        return states, behavior_logits, actions, dones, rewards
