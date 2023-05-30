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

import sys


def import_ms_compact():
    """Import mindspore with compact behavior."""
    if "mindspore" not in sys.modules:
        try:
            import mindspore.compat.v1 as ms
            ms.disable_v2_behavior()
        except ImportError:
            import mindspore as ms
        return ms
    else:
        return sys.modules["mindspore"]


ms = import_ms_compact()


# pylint: disable=W0611
if ms.__version__ in ("1.9.0"):
    from mindspore.nn import Adam
    from mindspore.nn import Conv2d, Dense, Flatten, ReLU
    from mindspore.nn import MSELoss
    from mindspore.train import Model
    from mindspore.nn import WithLossCell, TrainOneStepCell, SoftmaxCrossEntropyWithLogits, SequentialCell
    from mindspore.nn import Cell, WithLossCell, DynamicLossScaleUpdateCell, get_activation, LossBase, FixedLossScaleUpdateCell
    from mindspore import Model, Tensor
    from mindspore.ops import Cast, MultitypeFuncGraph, ReduceSum, ReduceMax, ReduceMin, ReduceMean, Reciprocal
    from mindspore.ops import Depend, value_and_grad, clip_by_global_norm, Minimum, Maximum, Exp, Square, clip_by_value
    from mindspore import History


def loss_to_val(loss):
    """Make keras instance into value."""
    if isinstance(loss, History):
        loss = loss.history.get("loss")[0]
    return loss


DTYPE_MAP = {
    "float32": ms.float32,
    "float16": ms.float16,
}
