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
from xt.model.tf_compat import Dense, Input, Model, Adam

from xt.model.dqn.default_config import HIDDEN_SIZE, NUM_LAYERS, LR
from xt.model import XTModel
from xt.util.common import import_config

from xt.framework.register import Registers


@Registers.model
class DqnMlp(XTModel):
    """docstring for ."""
    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        self.learning_rate = LR
        super().__init__(model_info)

    def create_model(self, model_info):
        """method for creating DQN Q network"""
        state = Input(shape=self.state_dim)
        denselayer = Dense(HIDDEN_SIZE, activation='relu')(state)
        for _ in range(NUM_LAYERS - 1):
            denselayer = Dense(HIDDEN_SIZE, activation='relu')(denselayer)
        value = Dense(self.action_dim, activation='linear')(denselayer)
        model = Model(inputs=state, outputs=value)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model
