#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
"""
Make multi-agents environment on catch pigs.

There are two agents and a pig in the pigsty,
the goal as two agents are trying to catch a moving pig in this pigsty.
More detail information could be found in `docs/third/CatchPigs.pdf`
"""

from xt.environment.ma.env_CatchPigs import EnvCatchPigs
from xt.environment.environment import Environment
from zeus.common.util.register import Registers


@Registers.env
class MaEnvCatchPigs(Environment):
    """Encapsulates multi-agents on catch pigs."""

    def init_env(self, env_info):
        """
        Create a atari environment instance.

        :param: the config information of environment.
        :return: the instance of environment
        """
        pigsty_size = env_info.get("size", 7)
        self.init_state = None
        self.vision = env_info.get("vision", False)
        self.action_type = "Categorical"

        # update the agents number and env api type.
        self.n_agents = 2
        self.api_type = "unified"

        return EnvCatchPigs(pigsty_size, True)

    def reset(self):
        """
        Reset the environment, if done is true, must clear obs array.

        :return: the observation of gym environment
        """
        self.env.reset()
        obs_list = self.env.get_obs()
        self.init_state = {"0": obs_list[0], "1": obs_list[1]}
        return self.init_state

    def step(self, action, agent_index=0):
        """
        Run one timestep of the environment's dynamics.

        Accepts an action and returns a tuple (state, reward, done, info).

        :param action: action
        :param agent_index: the index of agent
        :return: state, reward, done, info
        """
        if self.vision:
            self.env.render()

        reward, done = self.env.step([action["0"], action["1"]])
        obs_list = self.env.get_obs()
        state = {"0": obs_list[0], "1": obs_list[1]}
        multi_done = {"0": done, "1": done}
        info = {"0": dict(), "1": dict()}
        return state, {"0": reward[0], "1": reward[1]}, multi_done, info
