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
"""Make available with multi version tensorflow."""

import os
import sys


if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def import_tf_compact():
    """Import tensorflow with compact behavior."""
    if "tensorflow" not in sys.modules:
        # if 3 > int(tf.__version__.split('.')[0]) > 1:
        try:
            import tensorflow.compat.v1 as tf
            tf.disable_v2_behavior()
        except ImportError:
            import tensorflow as tf

        tf.logging.set_verbosity(tf.logging.ERROR)
        # tf.disable_resource_variables()
        return tf
    else:
        return sys.modules["tensorflow"]


tf = import_tf_compact()


# pylint: disable=W0611
if tf.__version__ in ("1.15.0", "2.0.0", "2.3.1"):
    from tensorflow.compat.v1.keras import backend as K
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Lambda
    from tensorflow.keras.layers import concatenate, Activation, Concatenate
    from tensorflow.keras.layers import GRU, Reshape, Embedding
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.python.keras.callbacks import History
    from tensorflow.python.keras.losses import MSE
    from tensorflow.compat.v1 import global_variables_initializer
    from tensorflow.compat.v1.train import AdamOptimizer, Saver
    from tensorflow.compat.v1.summary import scalar as summary_scalar
    from tensorflow.compat.v1.train import linear_cosine_decay, piecewise_constant

elif tf.__version__ in ("1.12.0", "1.13.1", "1.14.0",):
    from tensorflow.keras import backend as K
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, Lambda
    from tensorflow.keras.layers import concatenate, Activation, Concatenate
    from tensorflow.keras.layers import GRU, Reshape, Embedding
    from tensorflow.keras.models import Model
    from tensorflow.python.keras.callbacks import History
    from tensorflow import global_variables_initializer
    from tensorflow.train import AdamOptimizer, Saver
    from tensorflow.summary import scalar as summary_scalar
    from tensorflow.train import linear_cosine_decay, piecewise_constant

elif tf.__version__ in ("1.8.0", "1.4.1", "1.4.0"):
    from tensorflow.python.keras._impl.keras import backend as K
    from tensorflow.python.keras._impl.keras.optimizers import Adam
    from tensorflow.python.keras._impl.keras.layers import (
        Conv2D,
        Dense,
        Flatten,
        Input,
        Lambda,
        Concatenate,
    )
    from tensorflow.python.keras._impl.keras.layers import concatenate
    from tensorflow.python.keras._impl.keras.layers import Activation
    from tensorflow.python.keras._impl.keras.models import Model, Sequential
    from tensorflow.python.keras._impl.keras.callbacks import History
    from tensorflow import global_variables_initializer
    from tensorflow.train import AdamOptimizer, Saver, linear_cosine_decay, \
        piecewise_constant
else:
    raise ValueError("non-support tensorflow version: {}".format(tf.__version__))


def loss_to_val(loss):
    """Make keras instance into value."""
    if isinstance(loss, History):
        loss = loss.history.get("loss")[0]
    return loss


DTYPE_MAP = {
    "float32": tf.float32,
    "float16": tf.float16,
}


def get_sess_graph():
    """Get tf.graph and session."""
    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config, graph=graph)
    return sess, graph


def get_graph():
    """Get tf.Graph."""
    return tf.Graph()


def get_tf_major():
    """Get major of tensorflow version."""
    return int(tf.__version__.split(".")[0])
