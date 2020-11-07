#!/usr/bin/env python
"""
DESC: The agent module is used to explore and test in the environment for a specialized task.

The module receives the raw data from the environment as the
input, and transfers the raw data into the training state for RL model , and
then outputs an action by some exploration policy to the environment,
finally the next training state and corresponding reward is obtained.
During this process, the tuples needed for RL training have to be returned.
You can also define your specialized reward function, explosion policy,
training state and so on.
"""

import zeus.common.util.common as common
from xt.agent.agent import Agent

__ALL__ = ["Agent", "AsyncAgent", "agent_builder"]
from zeus.common.util.register import Registers


def agent_builder(agent_name, env, alg, agent_config, **kwargs):
    """
    Build an agent instance.

    :param agent_name:
    :param env:
    :param alg:
    :param agent_config:
    :param kwargs:
    :return:
    """
    return Registers.agent[agent_name](env, alg, agent_config, **kwargs)
