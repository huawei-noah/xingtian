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
"""Build dqn algorithm for information flow."""

from __future__ import division, print_function

import os
import time
import numpy as np

from xt.algorithm import Algorithm

from xt.algorithm.dqn.default_config import (
    BUFFER_SIZE,
    GAMMA,
    TARGET_UPDATE_FREQ,
    BATCH_SIZE,
)

from xt.algorithm.replay_buffer import ReplayBuffer
from zeus.common.util.register import Registers
from xt.model import model_builder

os.environ["KERAS_BACKEND"] = "tensorflow"


@Registers.algorithm
class DQNInfoFlowAlg(Algorithm):
    """Build Deep Q learning algorithm for info flow."""
    def __init__(self, model_info, alg_config, **kwargs):
        """
        Initialize DQN algorithm.

        It contains four steps:
        1. override the default config, with user's configuration;
        2. create the default actor with Algorithm.__init__;
        3. create once more actor, named by target_actor;
        4. create the replay buffer for training.
        :param model_info:
        :param alg_config:
        """
        model_info = model_info["actor"]
        super(DQNInfoFlowAlg, self).__init__(
            alg_name="info_flow_dqn",
            model_info=model_info,
            alg_config=alg_config)

        self.target_actor = model_builder(model_info)
        self.buff = ReplayBuffer(alg_config.get("buffer_size", BUFFER_SIZE))
        self.batch_size = alg_config.get("batch_size", BATCH_SIZE)
        self.target_update_freq = alg_config.get("target_update_freq", TARGET_UPDATE_FREQ)
        self.gamma = alg_config.get("gamma", GAMMA)

        self.item_dim = alg_config.get("item_dim")
        self.user_dim = alg_config.get("user_dim")
        self.async_flag = False
        self._times = list()

    def train(self, **kwargs):
        """
        Train process for DQN algorithm.

        1. predict the newest state with actor & target actor;
        2. calculate TD error;
        3. train operation;
        4. update target actor if need.
        :return: loss of this train step.
        """
        # _t = time.time()
        minibatch = self.buff.get_batch(self.batch_size)
        # print("minbatch: ", len(minibatch))
        # self._times.append(time.time() - _t)
        # if len(self._times) % 10 == 9:
        #     print("get last-10 min batch time: {}, as {}".format(
        #         np.nanmean(self._times[-10:]), self._times[-10:]))
        #     print("buffer size: {}".format(self.buff.size()))

        user_input = []
        history_click = []
        history_no_click = []
        item_input = []
        target_batch = []

        next_user_input = []
        next_history_click = []
        next_history_no_click = []
        next_item_input = []

        rewards = []
        dones = []
        cand_length = []
        for state_info, item, reward, next_state_info, done in minibatch:
            # print("in train: ", np.shape(state_info["candidate_items"]),
            #       np.shape(next_state_info["candidate_items"]))

            rewards.append(reward)
            dones.append(done)
            cand_length.append(len(next_state_info["candidate_items"]))
            user_input.append(state_info["user"])
            history_click.append(state_info["clicked_items"])
            history_no_click.append(state_info["viewed_items"])
            item_input.append(item)

            num_candidates = len(next_state_info["candidate_items"])
            tmp_next_user_input = np.tile(next_state_info["user"], (num_candidates, 1))
            tmp_next_history_click = np.tile(
                np.array(next_state_info["clicked_items"]).reshape(-1, self.item_dim * 5),
                (num_candidates, 1),
            )
            tmp_next_history_no_click = np.tile(
                np.array(next_state_info["viewed_items"]).reshape(-1, self.item_dim * 5),
                (num_candidates, 1),
            )
            tmp_next_item_input = np.array(next_state_info["candidate_items"])
            next_user_input.append(tmp_next_user_input)
            next_history_click.append(tmp_next_history_click)
            next_history_no_click.append(tmp_next_history_no_click)
            next_item_input.append(tmp_next_item_input)
        q_input_batch = {
            "user_input": np.array(user_input).reshape(-1, self.user_dim),
            "history_click": np.array(history_click).reshape(-1, self.item_dim * 5),
            "history_no_click": np.array(history_no_click).reshape(-1, self.item_dim * 5),
            "item_input": np.array(item_input).reshape(-1, self.item_dim),
        }
        next_q_input_batch = {
            "user_input": np.concatenate(next_user_input),
            "history_click": np.concatenate(next_history_click),
            "history_no_click": np.concatenate(next_history_no_click),
            "item_input": np.concatenate(next_item_input),
        }

        # q_values = np.array(self.model.predict_on_batch(next_q_input_batch)).reshape(-1)
        q_values = np.array(self.actor.predict(next_q_input_batch)).reshape(-1)
        # target_q_values = self.target_model.predict_on_batch(next_q_input_batch)

        new_q_values = []
        # new_target_q_values = []
        curr_idx = 0
        for length in cand_length:
            new_q_values.append(q_values[curr_idx:curr_idx + length])
            # new_target_q_values.append(target_q_values[curr_idx : curr_idx + length])
            curr_idx += length
        for i in range(self.batch_size):
            if dones[i]:
                target_batch.append(rewards[i])
            else:
                max_idx = np.argmax(new_q_values[i])
                target_batch.append(new_q_values[i][max_idx] * self.gamma + rewards[i])
        target_batch = np.array(target_batch)

        loss = self.actor.train(q_input_batch, target_batch,
                                batch_size=self.batch_size, verbose=False)

        if kwargs["episode_num"] % self.target_update_freq == 0:
            self.update_target()

        return loss

    def restore(self, model_name=None, model_weights=None):
        """
        Restore model weights.

        DQN will restore two model weights, actor & target.
        :param model_name:
        :param model_weights:
        :return:
        """
        if model_weights:
            self.actor.set_weights(model_weights)
            self.target_actor.set_weights(model_weights)
        else:
            self.actor.load_model(model_name)
            self.target_actor.load_model(model_name)

    def prepare_data(self, train_data, **kwargs):
        """
        Prepare the train data for DQN.

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

            # print("in prepare_data: ", np.shape(data[0]["candidate_items"]),
            #       np.shape(data[3]["candidate_items"]))
            buff.add(data)  # Add replay buffer

    def update_target(self):
        """
        Synchronize the actor's weight to target.

        :return:
        """
        weights = self.actor.get_weights()
        self.target_actor.set_weights(weights)

    def train_ready(self, elapsed_episode, **kwargs):
        """
        Support custom train logic.

        :return: train ready flag
        """
        # we set train ready as default
        self._train_ready = True
        if elapsed_episode < self.learning_starts:
            self._train_ready = False

            if not kwargs.get("dist_dummy_model"):
                raise KeyError("rec need to dist dummy model.")
            # dist dummy model
            kwargs["dist_dummy_model"]()
        else:
            self._train_ready = True

        return self._train_ready
