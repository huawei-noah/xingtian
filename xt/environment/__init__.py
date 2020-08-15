#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""environment module, do encapsulation for different simulations.
unify the single and multi-agents."""
from __future__ import division, print_function

from xt.framework import Registers


def env_builder(env_name, env_info, **kwargs):
    """
    the interface func for creating environment

    :param env_name：the name of environment
    :param env_info: the config info of environment
    :return：environment instance
    """
    return Registers.env[env_name](env_info, **kwargs)
