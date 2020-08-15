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
import tensorflow as tf
from xt.framework.register import Registers
from xt.model.tf_compat import Dense, Input, Model
from xt.model.ppo.ppo_cnn import PpoCnn
from xt.model.ppo.default_config import HIDDEN_SIZE, NUM_LAYERS


@Registers.model
class PpoMlp(PpoCnn):
    def create_model(self, model_info):
        state_input = Input(shape=self.state_dim, name='state_input')
        advantage = Input(shape=(1, ), name='adv')
        old_prediction = Input(shape=(self.action_dim, ), name='old_p')
        old_value = Input(shape=(1, ), name='old_v')

        denselayer = Dense(HIDDEN_SIZE, activation='relu')(state_input)
        for _ in range(NUM_LAYERS - 1):
            denselayer = Dense(HIDDEN_SIZE, activation='relu')(denselayer)
        out_actions = Dense(self.action_dim, activation='softmax', name='output_actions')(denselayer)
        out_value = Dense(1, name='output_value')(denselayer)
        model = Model(inputs=[state_input], outputs=[out_actions, out_value])
        if model_info.get("summary"):
            model.summary()

        self.build_graph(tf.float32, model)
        return model
