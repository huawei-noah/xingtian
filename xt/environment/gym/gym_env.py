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
"""Make gym env for simulation."""
import sys
import time

import gym
from gym_minigrid.register import register

from xt.environment.environment import Environment
from xt.environment.gym import infer_action_type
from zeus.common.util.register import Registers

register(id='MiniGrid-Ant-v0', entry_point='xt.environment.MiniGrid.ant:AntEnv')
register(id='MiniGrid-Dog-v0', entry_point='xt.environment.MiniGrid.dog:DogEnv')
register(id='MiniGrid-TrafficControl-v0', entry_point='xt.environment.MiniGrid.traffic_control:TrafficControlEnv')

@Registers.env
class GymEnv(Environment):
    """Encapsulate an openai gym environment."""

    def init_env(self, env_info):
        """
        Create a gym environment instance.

        :param: the config information of environment
        :return: the instance of environment
        """
        env = gym.make(env_info["name"])
        gym.Wrapper.__init__(self, env)

        self.action_type = infer_action_type(self.action_space)
        self.vision = env_info.get("vision", False)
        self.init_state = None
        return env

    def reset(self):
        """
        Reset the environment, if visionis true, must close environment first.

        :return: the observation of gym environment
        """
        if self.vision:
            self.env.close()
            time.sleep(0.05)
        state = self.env.reset()

        self.init_state = state
        return state

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

        state, reward, done, info = self.env.step(action)

        return state, reward, done, info
