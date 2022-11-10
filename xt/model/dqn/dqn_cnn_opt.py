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
from xt.model.tf_compat import Conv2D, Dense, Flatten, Input, Model, Adam, Lambda, K, MSE
from xt.model.dqn.default_config import LR
from xt.model.dqn.dqn_mlp import layer_normalize, layer_add
from xt.model import XTModel
from xt.model.tf_utils import TFVariables
from zeus.common.util.common import import_config

from zeus.common.util.register import Registers
tf.disable_eager_execution()

@Registers.model
class DqnCnnOpt(XTModel):
    """Docstring for DqnCnn."""

    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        self.learning_rate = LR
        self.dueling = model_config.get('dueling', False)

        super().__init__(model_info)

    def create_model(self, model_info):
        self.network = self.create_actor_model(model_info)
        self.target_network = self.create_actor_model(model_info)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.v = self.network.outputs[0]
        self.target_v = self.target_network.outputs[0]

        self.obs = self.network.inputs[0]
        self.tn_obs = self.target_network.inputs[0]

        self.full_model = DQNBase(self.network, self.target_network)

        self.build_graph()
        # self.sess.run(tf.initialize_all_variables())
        return self.full_model

    def create_actor_model(self, model_info):
        """Create Deep-Q CNN network."""
        state = Input(shape=self.state_dim, dtype="uint8")
        state1 = Lambda(lambda x: K.cast(x, dtype='float32') / 255.)(state)
        convlayer = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding='valid')(state1)
        convlayer = Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='valid')(convlayer)
        convlayer = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='valid')(convlayer)
        flattenlayer = Flatten()(convlayer)
        denselayer = Dense(256, activation='relu')(flattenlayer)
        value = Dense(self.action_dim, activation='linear')(denselayer)
        if self.dueling:
            adv = Dense(1, activation='linear')(denselayer)
            mean = Lambda(layer_normalize)(value)
            value = Lambda(layer_add)([adv, mean])
        model = Model(inputs=state, outputs=value)
        return model

    def build_train_graph(self):
        self.obs = tf.placeholder(tf.uint8, name="infer_input",
                                  shape=(None,) + tuple(self.state_dim))

        self.v = self.network(self.obs)
        self.true_v = tf.placeholder(tf.float32, name="true_v",
                                     shape=self.v.shape)

        loss = tf.keras.losses.mean_squared_error(self.true_v, self.v)
        self.loss = loss
        self.train_op = self.optimizer.minimize(loss)

    def build_infer_graph(self):
        self.tn_obs = tf.placeholder(tf.uint8, name="tn_input",
                                     shape=(None,) + tuple(self.state_dim))
        self.target_v = self.target_network(self.tn_obs)
        self.v = self.network(self.obs)

    def build_graph(self):
        self.build_train_graph()
        self.build_infer_graph()
        self.sess.run(tf.initialize_all_variables())
        self.explore_paras = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope="explore_agent")

    def train(self, state, label):
        """Train the model."""
        with self.graph.as_default():
            K.set_session(self.sess)
            feed_dict = {
                self.obs: state,
                self.true_v: label
            }
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
            # loss = self.model.train_on_batch(state, label)
            return loss

    def predict(self, state, tn_state):
        """
        Do predict use the newest model.

        :param state:
        :return:
        """
        with self.graph.as_default():
            K.set_session(self.sess)
            feed_dict = {self.obs: state, self.tn_obs: tn_state}
            return self.sess.run([self.v, self.target_v], feed_dict)

    def update_target(self):
        try:
            with self.graph.as_default():
                weights = self.network.get_weights()
                self.target_network.set_weights(weights)
        except ValueError:
            for t in self.explore_paras:
                print(t.name)
            raise RuntimeError("DEBUG")

    def get_weights(self):
        with self.graph.as_default():
            return self.model.get_weights()

    def set_weights(self, weights):
        try:
            with self.graph.as_default():
                self.model.set_weights(weights)
        except ValueError as e:
            for t in self.explore_paras:
                print(t.name)
            raise RuntimeError("DEBUG")


class DQNBase(Model):
    """Model that combine the representation and prediction (value+policy) network."""

    def __init__(self, actor_network: Model, target_actor_network: Model):
        super().__init__()
        self.actor_network = actor_network
        self.target_actor_network = target_actor_network
