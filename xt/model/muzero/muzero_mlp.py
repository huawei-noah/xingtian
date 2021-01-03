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

from xt.model.tf_compat import tf
from xt.model.tf_compat import K, Dense, MSE, Model, Sequential, Input, Lambda

from xt.model.muzero.muzero_model import MuzeroModel
from xt.model.muzero.muzero_utils import hidden_normlize
from xt.model.muzero.default_config import HIDDEN1_UNITS, HIDDEN2_UNITS, LR, td_step, max_value
from zeus.common.util.common import import_config

from zeus.common.util.register import Registers


# pylint: disable=W0201
@Registers.model
class MuzeroMlp(MuzeroModel):
    """Docstring for ActorNetwork."""

    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        super().__init__(model_info)

    def create_rep_network(self):
        obs = Input(shape=self.state_dim, name='rep_input')
        hidden = Dense(HIDDEN1_UNITS, activation='relu')(obs)
        out_rep = Dense(HIDDEN2_UNITS, activation='relu')(hidden)
        # out_rep = Lambda(hidden_normlize)(out_rep)
        return Model(inputs=obs, outputs=out_rep)

    def create_policy_network(self):
        hidden_input = Input(shape=HIDDEN2_UNITS, name='hidden_input')
        hidden = Dense(HIDDEN1_UNITS, activation='relu')(hidden_input)
        out_v = Dense(self.value_support_size, activation='softmax')(hidden)
        out_p = Dense(self.action_dim, activation='softmax')(hidden)
        return Model(inputs=hidden_input, outputs=[out_p, out_v])

    def create_dyn_network(self):
        conditioned_hidden = Input(shape=HIDDEN2_UNITS + self.action_dim)
        hidden = Dense(HIDDEN1_UNITS, activation='relu')(conditioned_hidden)
        out_h = Dense(HIDDEN2_UNITS, activation='relu')(hidden)
        # out_h = Lambda(hidden_normlize)(out_h)
        out_r = Dense(self.reward_support_size, activation='softmax')(hidden)
        return Model(inputs=conditioned_hidden, outputs=[out_h, out_r])
