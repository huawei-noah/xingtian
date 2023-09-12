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
from xt.model.dqn.default_config import LR
from xt.model.dqn.dqn_cnn_ms import DqnCnnMS
from xt.model.ms_utils import MSVariables
from xt.model.ms_compat import ms, Adam, MSELoss, WithLossCell, DynamicLossScaleUpdateCell, Tensor
from zeus.common.util.register import Registers
from xt.model.dqn.dqn_cnn_ms import MyTrainOneStepCell


@Registers.model
class DqnCnnPongMS(DqnCnnMS):
    """Docstring for DqnPong."""

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