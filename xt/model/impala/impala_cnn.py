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
"""IMPALA model."""
from xt.model.tf_compat import tf
from xt.model.tf_compat import Dense, Input, Conv2D, \
    Model, Adam, Lambda, Flatten, K

from xt.model.tf_utils import TFVariables
from xt.model.impala.default_config import ENTROPY_LOSS, LR
from xt.model import XTModel
from zeus.common.util.common import import_config
from zeus.common.util.register import Registers


@Registers.model
class ImpalaCnn(XTModel):
    """Create model for ImpalaNetworkCnn."""

    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        super().__init__(model_info)

    def create_model(self, model_info):
        state_input = Input(shape=self.state_dim, name='state_input', dtype='uint8')
        state_input_1 = Lambda(layer_function)(state_input)
        advantage = Input(shape=(1, ), name='adv')

        convlayer = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding='valid')(state_input_1)
        convlayer = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='valid')(convlayer)
        convlayer = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='valid')(convlayer)
        flattenlayer = Flatten()(convlayer)
        denselayer = Dense(256, activation='relu')(flattenlayer)

        out_actions = Dense(self.action_dim, activation='softmax', name='output_actions')(denselayer)
        out_value = Dense(1, name='output_value')(denselayer)
        model = Model(inputs=[state_input, advantage], outputs=[out_actions, out_value])
        losses = {"output_actions": impala_loss(advantage), "output_value": 'mse'}
        lossweights = {"output_actions": 1.0, "output_value": .5}

        decay_value = 0.00000000512
        model.compile(optimizer=Adam(lr=LR, clipnorm=40., decay=decay_value), loss=losses, loss_weights=lossweights)

        self.infer_state = tf.placeholder(tf.uint8, name="infer_state",
                                          shape=(None,) + tuple(self.state_dim))
        self.adv = tf.placeholder(tf.float32, name="adv", shape=(None, 1))
        self.infer_p, self.infer_v = model([self.infer_state, self.adv])

        self.actor_var = TFVariables([self.infer_p, self.infer_v], self.sess)

        self.sess.run(tf.initialize_all_variables())

        return model

    def train(self, state, label):
        with self.graph.as_default():
            # print(type(state[2][0][0]))
            K.set_session(self.sess)
            loss = self.model.fit(x={'state_input': state[0], 'adv': state[1]},
                                  y={"output_actions": label[0],
                                     "output_value": label[1]},
                                  batch_size=128,
                                  verbose=0)
            return loss

    def predict(self, state):
        """Do predict use the latest model."""
        with self.graph.as_default():
            K.set_session(self.sess)
            feed_dict = {self.infer_state: state[0], self.adv: state[1]}
            return self.sess.run([self.infer_p, self.infer_v], feed_dict)


def layer_function(x):
    """Normalize data."""
    return K.cast(x, dtype='float32') / 255.


def impala_loss(advantage):
    """Compute loss for impala."""
    def loss(y_true, y_pred):
        policy = y_pred
        log_policy = K.log(policy + 1e-10)
        entropy = -policy * K.log(policy + 1e-10)
        cross_entropy = -y_true * log_policy
        return K.mean(advantage * cross_entropy - ENTROPY_LOSS * entropy, 1)

    return loss
