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
"""Build Atari agent for ppo algorithm."""

import numpy as np

from absl import logging
from xt.agent.ppo.ppo import PPO
from xt.agent.ppo.default_config import GAMMA, LAM
from zeus.common.util.register import Registers


@Registers.agent
class AtariPpo(PPO):
    """Atari Agent with PPO algorithm."""

    def __init__(self, env, alg, agent_config, **kwargs):
        super().__init__(env, alg, agent_config, **kwargs)
        self.keep_seq_len = True
        self.next_state = None
        self.next_action = None
        self.next_value = None
        self.next_log_p = None

    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        info.update({'eval_reward': reward})

        self.transition_data.update({
            "reward": np.sign(reward) if use_explore else reward,
            "done": done,
            "info": info
        })

        return self.transition_data
