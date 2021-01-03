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
from xt.model.tf_compat import Dense, Input, Model, Adam, tf, Lambda
from xt.model.tf_utils import TFVariables
from xt.model.dqn.default_config import HIDDEN_SIZE, NUM_LAYERS, LR
from xt.model import XTModel
from zeus.common.util.common import import_config

from zeus.common.util.register import Registers


@Registers.model
class DqnMlp(XTModel):
    """Docstring for DqnMlp."""

    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        self.learning_rate = LR
        self.dueling = model_config.get('dueling', False)
        super().__init__(model_info)

    def create_model(self, model_info):
        """Create Deep-Q network."""
        state = Input(shape=self.state_dim)
        denselayer = Dense(HIDDEN_SIZE, activation='relu')(state)
        for _ in range(NUM_LAYERS - 1):
            denselayer = Dense(HIDDEN_SIZE, activation='relu')(denselayer)

        value = Dense(self.action_dim, activation='linear')(denselayer)
        if self.dueling:
            adv = Dense(1, activation='linear')(denselayer)
            mean = Lambda(layer_normalize)(value)
            value = Lambda(layer_add)([adv, mean])

        model = Model(inputs=state, outputs=value)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)

        self.infer_state = tf.placeholder(tf.float32, name="infer_input",
                                          shape=(None, ) + tuple(self.state_dim))
        self.infer_v = model(self.infer_state)
        self.actor_var = TFVariables([self.infer_v], self.sess)

        self.sess.run(tf.initialize_all_variables())
        return model

    def predict(self, state):
        """
        Do predict use the newest model.

        :param state:
        :return:
        """
        with self.graph.as_default():

            feed_dict = {self.infer_state: state}
            return self.sess.run(self.infer_v, feed_dict)

def layer_normalize(x):
    """Normalize data."""
    return tf.subtract(x, tf.reduce_mean(x, axis=1, keep_dims=True))


def layer_add(x):
    """Compute Q given Advantage and V."""
    return x[0] + x[1]
