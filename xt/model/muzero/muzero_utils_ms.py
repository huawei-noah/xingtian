import numpy as np
from mindspore import ops
from xt.model.ms_compat import ReduceMax, ReduceMin, ReduceMean


def scale_gradient_ms(tensor, scale):
    """Scales the gradient for the backward pass."""
    return tensor * scale + ops.stop_gradient(tensor) * (1 - scale)


def hidden_normlize_ms(hidden):
    reduce_max = ReduceMax(keep_dims=True)
    reduce_min = ReduceMin(keep_dims=True)
    hidden_max = reduce_max(hidden, -1)
    hidden_min = reduce_min(hidden, -1)
    hidden_norm = (hidden - hidden_min) / (hidden_max - hidden_min + 1e-10)
    return hidden_norm


def cross_entropy_ms(pred_p, target_p, loss_weights):
    log = ops.Log()
    reduce_mean = ReduceMean(keep_dims=True)
    _cross_entropy = reduce_mean(-target_p * log(pred_p + 1e-10), -1)
    return reduce_mean(_cross_entropy * 1.0)


def value_compression_ms(value):
    return np.sign(value) * (np.sqrt(np.abs(value) + 1) - 1) + 0.001 * value


def value_decompression_ms(value):
    return np.sign(value) * (
        (
            (np.sqrt(1 + 4 * 0.001 * (np.abs(value) + 1 + 0.001)) - 1)
            / (2 * 0.001)
        ) ** 2 - 1
    )
