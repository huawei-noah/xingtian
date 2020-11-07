# from .model_creator import model_creator
"""Create model module to define the NN architecture."""

from __future__ import division, print_function

from xt.framework import Registers
from xt.model.model import XTModel
from xt.model.tf_compat import tf
from xt.model.tf_utils import gelu
import zeus.common.util.common as common


__ALL__ = ['model_builder', 'Model']


ACTIVATION_MAP = {
    'sigmoid': tf.math.sigmoid,
    'tanh': tf.math.tanh,
    'softsign': tf.nn.softsign,
    'softplus': tf.nn.softplus,
    'relu': tf.nn.relu,
    'leaky_relu': tf.nn.leaky_relu,
    'elu': tf.nn.elu,
    'selu': tf.nn.selu,
    'swish': tf.nn.swish,
    'gelu': gelu
}


def model_builder(model_info):
    """Create the interface func for creating model."""
    model_name = model_info['model_name']
    model_ = Registers.model[model_name](model_info)
    return model_
