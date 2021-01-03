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
"""muzero algorithm """
import numpy as np

from xt.algorithm import Algorithm
from xt.algorithm.muzero.default_config import BATCH_SIZE, BUFFER_SIZE, GAMMA, TD_STEP, UNROLL_STEP
from xt.algorithm.replay_buffer import ReplayBuffer
from xt.algorithm.prioritized_replay_buffer_muzero import PrioritizedReplayBuffer
from zeus.common.util.register import Registers
from zeus.common.util.common import import_config


@Registers.algorithm
class Muzero(Algorithm):
    """ muzero algorithm """
    def __init__(self, model_info, alg_config, **kwargs):
        """
        Algorithm instance, will create their model within the `__init__`.
        :param model_info:
        :param alg_config:
        :param kwargs:
        """
        import_config(globals(), alg_config)
        super().__init__(
            alg_name=kwargs.get("name") or "muzero",
            model_info=model_info["actor"],
            alg_config=alg_config,
        )
        # self.buff = ReplayBuffer(BUFFER_SIZE)
        self.buff = PrioritizedReplayBuffer(BUFFER_SIZE, alpha=1)
        self.discount = GAMMA
        self.unroll_step = UNROLL_STEP
        self.td_step = TD_STEP
        self.async_flag = False

    def train(self, **kwargs):
        """ muzero train process."""
        if self.buff.len() < BATCH_SIZE:
            return 0
            
        trajs, traj_weights, traj_indexs = self.buff.sample(BATCH_SIZE, 1)

        traj_pos = [(t,) + tuple(self.sample_position(t)) for t in trajs]

        traj_data = [(g["cur_state"][i], g["action"][i:i + self.unroll_step],
                      self.make_target(i, g)) for (g, i, w) in traj_pos]
        image = np.asarray([e[0] for e in traj_data])
        actions = np.asarray([e[1] for e in traj_data])
        targets = [e[2] for e in traj_data]

        target_values = []
        target_rewards = []
        target_policys = []
        for target in targets:
            target_values.append([e[0] for e in target])
            target_rewards.append([e[1] for e in target])
            target_policys.append([e[2] for e in target])

        target_values = np.asarray(target_values)
        target_rewards = np.asarray(target_rewards)
        target_policys = np.asarray(target_policys)
        traj_weights = np.expand_dims(traj_weights, -1)
        loss = self.actor.train([image, actions, traj_weights],
                                [target_values, target_rewards, target_policys])

        self.update_pri(traj_pos, image, target_values[:, 0])
        return loss

    def prepare_data(self, train_data, **kwargs):
        if len(train_data["reward"]) > self.unroll_step + 1:
            priorities = self.calc_pri(train_data)
            # print('priorities', priorities.shape)
            pos_buff = PrioritizedReplayBuffer(len(priorities), alpha=1)
            for i in range(len(priorities) - self.unroll_step):
                pos_buff.add(0, priorities[i])

            train_data.update({"pos_buff": pos_buff})
            self.buff.add(train_data, pos_buff.weight())

    def sample_position(self, traj):
        pos_buff = traj["pos_buff"]
        value, weight, index = pos_buff.sample(1, 1)
        return index[0], weight[0]

    def make_target(self, state_index, traj):
        """Generate targets to learn from during the network training."""

        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        targets = []
        root_values = traj["root_value"]
        rewards = traj["reward"]
        child_visits = traj["child_visits"]
        target_value = traj["target_value"]
        obs = traj["cur_state"]

        for current_index in range(state_index, state_index + self.unroll_step + 1):

            if current_index < len(root_values):
                targets.append((target_value[current_index], rewards[current_index], child_visits[current_index]))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, 0, []))
        return targets

    def calc_pri(self, train_data):
        state = np.asarray(train_data["cur_state"])
        # value = np.asarray(train_data["root_value"])
        target_value = np.asarray(train_data["target_value"])

        value = self.actor.value_inference(state)

        return np.abs(value - target_value)

    def update_pri(self, traj_pos, state, target_value):
        value = self.actor.value_inference(state)
        # print(value.shape, target_value.shape)
        new_pri = np.abs(value - np.squeeze(target_value))
        new_pri = np.maximum(new_pri, 1e-5)

        for i, (g, pos, pos_pri) in enumerate(traj_pos):
            pos_buff = g["pos_buff"]

            pos_buff.update_priorities([pos], [new_pri[i]])
            self.buff.update_priorities([i], [pos_buff.weight()])
