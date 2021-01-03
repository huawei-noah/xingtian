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
"""Build base agent for PPO algorithm."""

import numpy as np

from xt.agent import Agent
from xt.agent.ppo.default_config import GAMMA, LAM
from zeus.common.util.register import Registers


@Registers.agent
class PPO(Agent):
    """Build base agent with PPO algorithm."""
    def __init__(self, env, alg, agent_config, **kwargs):
        super().__init__(env, alg, agent_config, **kwargs)

    def infer_action(self, state, use_explore):
        """
        Infer an action with `state`.

        :param state:
        :param use_explore:
        :return: action value
        """
        predict_val = self.alg.predict(state)
        return self.handel_predict_value(state, predict_val)

    def handel_predict_value(self, state, predict_val):
        action = predict_val[0][0]
        logp = predict_val[1][0]
        value = predict_val[2][0]

        # update transition data
        self.transition_data.update({
            'cur_state': state,
            'action': action,
            'logp': logp,
            'value': value,
        })

        return action

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):

        self.transition_data.update({
            'reward': reward,
            'done': done,
            'info': info
        })

        return self.transition_data

    def get_trajectory(self, last_pred=None):
        last_val = last_pred[2][0]
        self.trajectory['value'].append(last_val)
        self.data_proc()
        return super().get_trajectory()

    def data_proc(self):
        """Process data."""
        traj = self.trajectory
        state = np.asarray(traj['cur_state'])
        action = np.asarray(traj['action'])
        logp = np.asarray(traj['logp'])
        value = np.asarray(traj['value'])
        reward = np.asarray(traj['reward'])
        done = np.asarray(traj['done'])

        next_value = value[1:]
        value = value[:-1]

        done = np.expand_dims(done, axis=1)
        reward = np.expand_dims(reward, axis=1)
        discount = ~done * GAMMA
        delta_t = reward + discount * next_value - value
        adv = delta_t

        for j in range(len(adv) - 2, -1, -1):
            adv[j] += adv[j + 1] * discount[j] * LAM

        self.trajectory['cur_state'] = state
        self.trajectory['action'] = action
        self.trajectory['logp'] = logp
        self.trajectory['adv'] = adv
        self.trajectory['old_value'] = value
        self.trajectory['target_value'] = adv + value

        del self.trajectory['value']
