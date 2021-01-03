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

from xt.model.tf_compat import Dense, Input, Model, Adam, K, tf
from xt.model.tf_utils import TFVariables
from xt.model.impala.default_config import ENTROPY_LOSS, HIDDEN_SIZE, LR, NUM_LAYERS
from xt.model.model_zeus import XTModelZeus

from zeus.common.util.common import import_config
from zeus.common.util.register import Registers
from zeus import set_backend
from zeus.trainer_api import Trainer
from zeus.common.class_factory import ClassFactory, ClassType
from zeus.trainer.modules.conf.loss import LossConfig
from zeus.trainer.modules.conf.optim import OptimConfig
from zeus.modules.module import Module
from zeus.modules.operators.ops import Relu, Linear, Lambda, softmax
from zeus.modules.connections import MultiOutput, Sequential

set_backend(backend='tensorflow', device_category='GPU')


@Registers.model
class ImpalaMlpZeus(XTModelZeus):
    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        self._batch_size = 128
        super().__init__(model_info)

    def create_model(self, model_info):
        """Create zeus network."""
        zeus_model = ImpalaMlpNet(state_dim=self.state_dim, action_dim=self.action_dim)

        LossConfig.type = 'impala_loss'
        OptimConfig.type = 'Adam'
        OptimConfig.params.update({'lr': LR})

        loss_input = dict()
        loss_input['inputs'] = [{"name": "input_state", "type": "float32", "shape": self.state_dim}]
        loss_input['labels'] = [{"name": "target_p", "type": "float32", "shape": self.action_dim}]
        loss_input['labels'].append({"name": "target_v", "type": "float32", "shape": 1})
        loss_input['labels'].append({"name": "adv", "type": "float32", "shape": 1})
        model = Trainer(model=zeus_model, backend='tensorflow', device='GPU',
                        loss_input=loss_input, lazy_build=False)

        return model

    def train(self, state, label):
        nbatch = state[0].shape[0]
        inds = np.arange(nbatch)
        loss_val = []

        np.random.shuffle(inds)
        for start in range(0, nbatch, self._batch_size):
            end = start + self._batch_size
            mbinds = inds[start:end]

            inputs = [state[0][mbinds]]
            labels = [label[0][mbinds], label[1][mbinds], state[1][mbinds]]
            ret_loss = self.model.train(inputs, labels)
            loss_val.append(ret_loss)

        return np.mean(loss_val)

    def predict(self, state):
        """Do predict use the latest model."""
        # print(state)
        return self.model.predict(state[0])


class ImpalaMlpNet(Module):
    """Create DQN net with FineGrainedSpace."""
    def __init__(self, **descript):
        """Create layers."""
        super().__init__()
        state_dim = descript.get("state_dim")
        action_dim = descript.get("action_dim")

        self.fc2 = Sequential(Linear(state_dim, HIDDEN_SIZE), Linear(HIDDEN_SIZE, action_dim),
                              Lambda(lambda x: softmax(x)))
        self.fc3 = Sequential(Linear(state_dim, HIDDEN_SIZE), Linear(HIDDEN_SIZE, 1))

        # self.out = MultiOutput(fc2, fc3)

    def __call__(self, inputs):
        """Override compile function, conect models into a seq."""

        outputs = []
        outputs.append(self.fc2(inputs))
        outputs.append(self.fc3(inputs))
        return outputs


@ClassFactory.register(ClassType.LOSS, 'impala_loss')
def impala_loss(inputs, lables):
    """Compute loss for IMPALA."""
    policy, value = inputs
    target_p, target_v, adv = lables

    log_policy = tf.log(policy + 1e-10)
    entropy = (-policy * log_policy)
    cross_entropy = (-target_p * log_policy)
    p_loss = tf.reduce_mean(adv * cross_entropy - ENTROPY_LOSS * entropy)

    v_loss = 0.5 * tf.reduce_mean(tf.square(value - target_v))

    return p_loss + v_loss
