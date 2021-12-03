#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build environment module.

Do encapsulation for different simulations.
Unify the single and multi-agents.
"""

from __future__ import division, print_function

from xt.framework import Registers

from gym_minigrid.register import register

register(id='MiniGrid-Ant-v0', entry_point='xt.environment.MiniGrid.ant:AntEnv')
register(id='MiniGrid-Dog-v0', entry_point='xt.environment.MiniGrid.dog:DogEnv')
register(id='MiniGrid-TrafficControl-v0', entry_point='xt.environment.MiniGrid.traffic_control:TrafficControlEnv')

def env_builder(env_name, env_info, **kwargs):
    """
    Build the interface func for creating environment.

    :param env_name：the name of environment
    :param env_info: the config info of environment
    :return：environment instance
    """
    return Registers.env[env_name](env_info, **kwargs)
