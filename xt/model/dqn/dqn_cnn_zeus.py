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
from xt.model.tf_compat import tf, MSE
from xt.model.dqn.default_config import LR
from xt.model.model_zeus import XTModelZeus

from zeus import set_backend
from zeus.common.util.common import import_config
from zeus.common.util.register import Registers
from zeus.trainer.trainer_api import Trainer
from zeus.common.class_factory import ClassFactory, ClassType
from zeus.trainer.modules.conf.loss import LossConfig
from zeus.trainer.modules.conf.optim import OptimConfig
from zeus.modules.module import Module
from zeus.modules.operators.ops import Relu, Linear, Conv2d, View, Lambda

set_backend(backend='tensorflow', device_category='GPU')


@Registers.model
class DqnCnnZeus(XTModelZeus):
    """Docstring for DqnCnn."""
    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        self.learning_rate = LR

        super().__init__(model_info)

    def create_model(self, model_info):
        """Create Deep-Q network."""
        zeus_model = DqnCnnNet(state_dim=self.state_dim, action_dim=self.action_dim)

        LossConfig.type = 'mse_loss'
        OptimConfig.type = 'Adam'
        OptimConfig.params.update({'lr': self.learning_rate})

        loss_input = dict()
        loss_input['inputs'] = [{"name": "input_state", "type": "float32", "shape": self.state_dim}]
        loss_input['labels'] = [{"name": "target_value", "type": "float32", "shape": self.action_dim}]

        model = Trainer(model=zeus_model, backend='tensorflow', device='GPU',
                        loss_input=loss_input, lazy_build=False)
        return model


class DqnCnnNet(Module):
    """Create DQN net with FineGrainedSpace."""
    def __init__(self, **descript):
        """Create layers."""
        super().__init__()
        state_dim = descript.get("state_dim")
        action_dim = descript.get("action_dim")

        self.lambda1 = Lambda(lambda x: tf.cast(x, dtype='float32') / 255.)
        self.conv1 = Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, bias=False)
        self.ac1 = Relu()
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=False)
        self.ac2 = Relu()
        self.conv3 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, bias=False)
        self.ac3 = Relu()
        self.view = View()
        self.fc1 = Linear(64, 256)
        self.ac4 = Relu()
        self.fc2 = Linear(256, action_dim)


@ClassFactory.register(ClassType.LOSS, 'mse_loss')
def mse_loss(logits, labels):
    return tf.reduce_mean(MSE(logits, labels))
