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
import numpy as np
from xt.model.tf_compat import tf

def scale_gradient(tensor, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + tf.stop_gradient(tensor) * (1 - scale)


def hidden_normlize(hidden):
    hidden_max = tf.reduce_max(hidden, axis=-1, keepdims=True)
    hidden_min = tf.reduce_min(hidden, axis=-1, keepdims=True)
    hidden_norm = (hidden - hidden_min) / (hidden_max - hidden_min + 1e-10)
    return hidden_norm


def cross_entropy(pred_p, target_p, loss_weights):
    _cross_entropy = tf.reduce_mean(-target_p * tf.log(pred_p + 1e-10), axis=-1, keepdims=True)
    return tf.reduce_mean(_cross_entropy * 1.0)


def value_compression(value):
    return np.sign(value) * (np.sqrt(np.abs(value) + 1) - 1) + 0.001 * value


def value_decompression(value):
    return np.sign(value) * (
        ((np.sqrt(1 + 4 * 0.001 * (np.abs(value) + 1 + 0.001)) - 1) / (2 * 0.001)) ** 2 - 1
    )
