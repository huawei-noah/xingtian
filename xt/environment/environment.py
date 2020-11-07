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
"""Make Environment base class, User could inherit it."""

from __future__ import division, print_function


class Environment(object):
    """Make environment basic class."""

    def __init__(self, env_info, **kwargs):
        """
        Initialize environment.

        :param env_info: the config info of environment
        """
        # default agent number is one, multi_agent environment must assign this value
        self.n_agents = 1
        # Support two API_TYPE:
        # 1) standalone, each agent could do interaction with their environment
        # 2) unified, Agents within group do interaction with environment in one step.
        self.api_type = "standalone"
        self.action_type = None

        self.env = self.init_env(env_info)
        self.id = kwargs.get("env_id", 0)
        self.init_state = None

    def init_env(self, env_info):
        """
        Create an environment instance.

        # NOTE: User must assign the `api_type` value on multi_agent.
        :param: the config information of environment
        :return: the instance of environment
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the environment.

        :return: the observation of environment
        """
        state = self.env.reset()
        self.init_state = state

        return state

    def step(self, action, agent_index=0):
        """
        Send action  to running agent in this environment.

        :param action: action
        :param agent_index: the index of agent
        :return: state, reward
        """
        return self.env.step(action)

    def get_init_state(self, agent_index=0):
        """
        Get reset observation of one agent.

        :param agent_index: the index of agent
        :return: the reset observation of agent
        """
        return self.init_state

    def stop_agent(self, agent_index):
        """
        Stop one agent running.

        :param agent_index: the index of agent
        :return:
        """
        return self.env.stop_agent(agent_index)

    def get_env_info(self):
        """Return environment's basic information."""
        self.reset()
        env_info = {
            "n_agents": self.n_agents,
            "api_type": self.api_type,
            "action_type": self.action_type
        }
        # update the agent ids, will used in the weights map.
        # default work well with the sumo multi-agents
        agent_ids = list(self.get_init_state().keys()) if self.n_agents > 1 else [0]
        env_info.update({"agent_ids": agent_ids})

        return env_info

    def close(self):
        """Close environment."""
        try:  # do some thing you need
            self.env.close()
        except AttributeError:
            print("please complete your env close function")
