#!/usr/bin/env python
"""DESC: This module Contains the train and prepare data operations within special algorithm."""

from __future__ import division, print_function

from xt.algorithm.algorithm import Algorithm
from xt.algorithm.algorithm import AGENT_PREFIX, MODEL_PREFIX

__ALL__ = [
    "Algorithm",
    "alg_builder",
    "AGENT_PREFIX",
    "MODEL_PREFIX",
]

from xt.framework import Registers


def alg_builder(alg_name, model_info, alg_config, **kwargs):
    """
    DESC: The API to build a algorithm instance.

    :param alg_name:
    :param model_info:
    :param alg_config:
    :return:
    """
    return Registers.algorithm[alg_name](model_info, alg_config, **kwargs)
