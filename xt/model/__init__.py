# from .model_creator import model_creator
"""
model module work for the NN architecture define
"""
from __future__ import division, print_function
from xt.model.model import XTModel
import xt.util.common as common
from xt.framework import Registers

__ALL__ = ["model_builder", "Model"]


def model_builder(model_info):
    """the interface func for creating model"""
    model_name = model_info['model_name']
    model_ = Registers.model[model_name](model_info)
    return model_
