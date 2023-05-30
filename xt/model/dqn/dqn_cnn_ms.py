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

from zeus.common.util.register import Registers
from xt.model.model_ms import XTModel_MS
from xt.model.ms_utils import MSVariables
from xt.model.dqn.default_config import LR
from xt.model.ms_compat import ms
from xt.model.ms_compat import Conv2d, Dense, Flatten, ReLU, Adam, MSELoss, WithLossCell, MultitypeFuncGraph, \
    DynamicLossScaleUpdateCell, Cast, Cell, Tensor
from zeus.common.util.common import import_config
import mindspore.ops as ops
import numpy as np

@Registers.model
class DqnCnnMS(XTModel_MS):
    """Docstring for DqnCnn."""

    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        self.learning_rate = LR
        self.dueling = model_config.get('dueling', False)
        self.net = DqnCnnNet(state_dim=self.state_dim, action_dim=self.action_dim, dueling=self.dueling)
        super().__init__(model_info)
        self.net.compile(ms.Tensor(np.zeros((1, 84, 84, 4))).astype(ms.float32))

    def create_model(self, model_info):
        """Create Deep-Q CNN network."""
        loss_fn = MSELoss()
        adam = Adam(params=self.net.trainable_params(), learning_rate=self.learning_rate, use_amsgrad=True)
        loss_net = WithLossCell(self.net, loss_fn)
        device_target = ms.get_context("device_target")
        if device_target == 'Ascend':
            manager = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 12, scale_factor=2, scale_window=1000)
            model = MyTrainOneStepCell(loss_net, adam, manager, grad_clip=True, clipnorm=10.)
        else:
            model = MyTrainOneStepCell(loss_net, adam, grad_clip=True, clipnorm=10.)
        self.actor_var = MSVariables(self.net)
        return model

    def predict(self, state):
        state = Tensor(state, dtype=ms.float32)
        return self.net(state).asnumpy()


class DqnCnnNet(Cell):
    def __init__(self, **descript):
        super(DqnCnnNet, self).__init__()
        self.state_dim = descript.get("state_dim")
        action_dim = descript.get("action_dim")
        self.dueling = descript.get("dueling")
        self.convlayer1 = Conv2d(self.state_dim[2], 32, kernel_size=8, stride=4, pad_mode='valid',
                                 weight_init="xavier_uniform")
        self.convlayer2 = Conv2d(32, 64, kernel_size=4, stride=2, pad_mode='valid', weight_init="xavier_uniform")
        self.convlayer3 = Conv2d(64, 64, kernel_size=3, stride=1, pad_mode='valid', weight_init="xavier_uniform")
        self.relu = ReLU()
        self.flattenlayer = Flatten()
        _dim = (
                (((self.state_dim[0] - 4) // 4 - 2) // 2 - 2)
                * (((self.state_dim[1] - 4) // 4 - 2) // 2 - 2)
                * 64
        )
        self.denselayer1 = Dense(_dim, 256, activation='relu', weight_init="xavier_uniform")
        self.denselayer2 = Dense(256, action_dim, weight_init="xavier_uniform")
        self.denselayer3 = Dense(256, 1, weight_init="xavier_uniform")

    def construct(self, x):
        out = Cast()(x.transpose((0, 3, 1, 2)), ms.float32) / 255.
        out = self.convlayer1(out)
        out = self.relu(out)
        out = self.convlayer2(out)
        out = self.relu(out)
        out = self.convlayer3(out)
        out = self.relu(out)
        out = self.flattenlayer(out)
        out = self.denselayer1(out)
        value = self.denselayer2(out)
        if self.dueling:
            adv = self.denselayer3(out)
            mean = value.sub(value.mean(axis=1, keep_dims=True))
            value = adv.add(mean)
        return value


_grad_scale = MultitypeFuncGraph("grad_scale")


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.cast(ops.Reciprocal()(scale), ops.dtype(grad))


class MyTrainOneStepCell(ms.nn.TrainOneStepWithLossScaleCell):
    def __init__(self, network, optimizer, scale_sense=1, grad_clip=False, clipnorm=1.):
        self.clipnorm = clipnorm
        if isinstance(scale_sense, (int, float)):
            scale_sense = Tensor(scale_sense, dtype=ms.float32)
        super(MyTrainOneStepCell, self).__init__(network, optimizer, scale_sense)
        self.grad_clip = grad_clip

    def construct(self,*inputs ):
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = ops.ones_like(loss) * ops.cast(scaling_sens, ops.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(ops.partial(_grad_scale, scaling_sens), grads)
        if self.grad_clip:
            grads = ops.clip_by_global_norm(grads, self.clipnorm)
        grads = self.grad_reducer(grads)
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        if not overflow:
            loss = ops.depend(loss, self.optimizer(grads))
        return loss