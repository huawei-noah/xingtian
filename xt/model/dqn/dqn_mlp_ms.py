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
from xt.model.dqn.default_config import HIDDEN_SIZE, NUM_LAYERS, LR
from xt.model.model_ms import XTModel_MS
from zeus.common.util.common import import_config
from zeus.common.util.register import Registers
from xt.model.ms_compat import Dense, Adam, DynamicLossScaleUpdateCell, MSELoss, Cell, Model, ms
import mindspore.ops as ops
from xt.model.ms_utils import MSVariables
from xt.model.dqn.dqn_cnn_ms import MyTrainOneStepCell


@Registers.model
class DqnMlpMS(XTModel_MS):

    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        self.learning_rate = LR
        self.dueling = model_config.get('dueling', False)
        self.net = DqnMlpNet(state_dim=self.state_dim, action_dim=self.action_dim, dueling=self.dueling)
        super().__init__(model_info)

    def create_model(self, model_info):
        """Create Deep-Q CNN network."""
        loss_fn = MSELoss()
        adam = Adam(params=self.net.trainable_params(), learning_rate=self.learning_rate)
        loss_net = ms.nn.WithLossCell(self.net, loss_fn)
        device_target = ms.get_context("device_target")
        if device_target == 'Ascend':
            manager = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 12, scale_factor=2, scale_window=1000)
            model = MyTrainOneStepCell(loss_net, adam, manager, grad_clip=True, clipnorm=10.)
        else:
            model = MyTrainOneStepCell(loss_net, adam, grad_clip=True, clipnorm=10.)
        self.actor_var = MSVariables(self.net)
        return model

    def predict(self, state):
        state = ms.Tensor(state, dtype=ms.float32)
        return self.net(state).asnumpy()


class DqnMlpNet(Cell):
    def __init__(self, **descript):
        super(DqnMlpNet, self).__init__()
        self.state_dim = descript.get("state_dim")
        self.action_dim = descript.get("action_dim")
        self.dueling = descript.get("dueling")
        self.denselayer1 = Dense(self.state_dim[-1], HIDDEN_SIZE, activation='relu', weight_init='xavier_uniform')
        self.denselayer2 = Dense(HIDDEN_SIZE, HIDDEN_SIZE, activation='relu', weight_init='xavier_uniform')
        self.denselayer3 = Dense(HIDDEN_SIZE, self.action_dim, weight_init='xavier_uniform')
        self.denselayer4 = Dense(HIDDEN_SIZE, 1, weight_init='xavier_uniform')

    def construct(self, x):
        out = self.denselayer1(x.astype("float32"))
        for _ in range(NUM_LAYERS - 1):
            out = self.denselayer2(out)
        value = self.denselayer3(out)
        if self.dueling:
            adv = self.denselayer4(out)
            mean = value.sub(value.mean(axis=1, keep_dims=True))
            value = adv.add(mean)
        return value