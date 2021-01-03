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
"""Build CartPole agent for ppo algorithm."""

import numpy as np

from xt.agent import Agent
from xt.agent.ppo.default_config import GAMMA, LAM
from zeus.common.util.register import Registers


@Registers.agent
class CartpolePpo(Agent):
    """Build Cartpole Agent with PPO algorithm."""

    def __init__(self, env, alg, agent_config, **kwargs):
        super().__init__(env, alg, agent_config, **kwargs)
        self.next_state = None
        self.next_action = None
        self.next_value = None

    def infer_action(self, state, use_explore):
        """
        Infer an action with `state`.

        :param state:
        :param use_explore:
        :return: action value
        """
        if self.next_state is None:
            # print("multi preidict")
            s_t = state
            predict_val = self.alg.predict(s_t)
            action = predict_val[0][0]
            value = predict_val[1][0]
        else:
            s_t = self.next_state
            action = self.next_action
            value = self.next_value

        real_action = np.random.choice(self.alg.action_dim, p=np.nan_to_num(action))

        # update transition data
        self.transition_data.update({
            "cur_state": s_t,
            "action": action,
            "value": value,
            "real_action": real_action
        })

        return real_action

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        predict_val = self.alg.predict(next_raw_state)
        self.next_action = predict_val[0][0]
        self.next_value = predict_val[1][0]
        self.next_state = next_raw_state
        self.transition_data.update({
            "reward": reward,
            "next_value": self.next_value,
            "done": done,
            "info": info
        })

        return self.transition_data

    def get_trajectory(self, last_pred=None):
        self.data_proc()
        return super().get_trajectory()

    def data_proc(self):
        traj = self.trajectory
        action = np.asarray(traj["real_action"])
        action_label = np.eye(self.action_dim)[action.reshape(-1)]

        value = np.asarray(traj["value"])
        next_value = np.asarray(traj["next_value"])
        dones = np.asarray(traj["done"])
        dones[-1] = dones[-1] and not traj["info"][-1]
        dones = np.expand_dims(dones, 1)
        rewards = np.asarray(traj["reward"])
        rewards = np.expand_dims(rewards, 1)
        state = np.asarray(self.trajectory["cur_state"])
        real_action = np.asarray(self.trajectory["action"])

        discounts = ~dones * GAMMA
        deltas = rewards + discounts * next_value - value
        adv = deltas
        for j in range(len(adv) - 2, -1, -1):
            adv[j] += adv[j + 1] * discounts[j] * LAM

        self.trajectory["adv"] = adv
        self.trajectory["target_value"] = adv + value
        self.trajectory["target_action"] = action_label
        self.trajectory["cur_state"] = state
        self.trajectory["action"] = real_action
        self.trajectory["value"] = value

        del self.trajectory["next_value"]
        del self.trajectory["real_action"]
