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

import tensorflow as tf
from tensorflow.python.keras import layers

from xt.model.tf_compat import K, Dense, MSE, Model, Conv2D, Input, Flatten, Lambda
from xt.model.muzero.default_config import LR, td_step, max_value
from xt.model.muzero.muzero_model import MuzeroModel, MuzeroBase, NetworkOutput, scale_gradient

from zeus.common.util.register import Registers


@Registers.model
class MuzeroAtari(MuzeroModel):
    """docstring for ActorNetwork."""
    def create_model(self, model_info):
        self.weight_decay = 1e-4
        self.optimizer = tf.train.AdamOptimizer(LR)
        self.td_step = td_step
        self.max_value = max_value
        self.value_support_size = 300
        self.full_support_size = 601

        self.representation_network = self.create_rep_network()
        self.policy_network = self.create_policy_network()
        self.dynamic_network = self.create_dyn_network()

        self.out_v = self.policy_network.outputs[1]
        self.out_p = self.policy_network.outputs[0]
        self.out_h = self.dynamic_network.outputs[0]
        self.out_r = self.dynamic_network.outputs[1]
        self.out_rep = self.representation_network.outputs[0]
        self.hidden = self.policy_network.inputs[0]
        self.conditioned_hidden = self.dynamic_network.inputs[0]
        self.obs = self.representation_network.inputs[0]
        self.full_model = MuzeroBase(self.representation_network,
                                     self.dynamic_network,
                                     self.policy_network)

        self.train_op = self.build_graph()
        self.sess.run(tf.initialize_all_variables())

        return self.full_model

    def create_rep_network(self):
        obs = Input(shape=self.state_dim, name='rep_input')
        obs_1 = Lambda(lambda x: K.cast(x, dtype='float32') / 255.)(obs)
        convlayer = down_sample(obs_1)
        # [convlayer = residual_block(convlayer, 256) for _ in range(16)]
        for _ in range(6):
            convlayer = residual_block(convlayer, 256)
        return Model(inputs=obs, outputs=convlayer)

    def create_policy_network(self):
        hidden_input = Input(shape=(6, 6, 256,), name='hidden_input')
        hidden_v = Conv2D(256, (3, 3), activation='relu', padding='same')(hidden_input)
        hidden_p = Conv2D(256, (3, 3), activation='relu', padding='same')(hidden_input)
        hidden_v = Flatten()(hidden_v)
        hidden_p = Flatten()(hidden_p)
        out_v = Dense(self.full_support_size)(hidden_v)
        out_p = Dense(self.action_dim)(hidden_p)
        return Model(inputs=hidden_input, outputs=[out_p, out_v])

    def create_dyn_network(self):
        conditioned_hidden = Input(shape=(6, 6, 257, ))
        convlayer = Conv2D(256, (3, 3), activation='relu', padding='same')(conditioned_hidden)
        for _ in range(16):
            convlayer = residual_block(convlayer, 256)
        out_h = convlayer
        hidden = Conv2D(256, (3, 3), activation='relu', padding='same')(convlayer)
        hidden = Flatten()(hidden)
        out_r = Dense(1)(hidden)
        return Model(inputs=conditioned_hidden, outputs=[out_h, out_r])

    def recurrent_inference(self, hidden_state, action):
        with self.graph.as_default():
            K.set_session(self.sess)
            action = np.eye(36)[action]
            action = action.reshape(6, 6, 1)
            action = np.expand_dims(action, 0)
            hidden_state = np.expand_dims(hidden_state, 0)
            # print("recu shape", hidden_state.shape, action.shape)
            conditioned_hidden = np.concatenate((hidden_state, action), axis=-1)
            feed_dict = {self.conditioned_hidden: conditioned_hidden}
            hidden, reward = self.sess.run([self.out_h, self.out_r], feed_dict)

            feed_dict = {self.hidden: hidden}
            policy, value = self.sess.run([self.out_p, self.out_v], feed_dict)
            value = self._value_transform(value[0])
            print("value", value, "reward", reward)
        return NetworkOutput(value, reward[0], policy[0], hidden[0])

    def build_graph(self):
        self.image = tf.placeholder(tf.float32, name="obs",
                                    shape=(None, ) + tuple(self.state_dim))
        self.action = tf.placeholder(tf.int32, name="action",
                                     shape=(None, self.td_step))
        self.target_value = tf.placeholder(tf.float32, name="value",
                                           shape=(None, ) + (1 + self.td_step, self.full_support_size))
        self.target_reward = tf.placeholder(tf.float32, name="reward",
                                            shape=(None, ) + (1 + self.td_step, ))
        self.target_policy = tf.placeholder(tf.float32, name="policy",
                                            shape=(None, ) + (1 + self.td_step, self.action_dim))

        hidden_state = self.representation_network(self.image)
        policy_logits, value = self.policy_network(hidden_state)

        loss = tf.math.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
            logits=value, labels=self.target_value[:, 0]))
        loss += tf.math.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
            logits=policy_logits, labels=self.target_policy[:, 0]))

        gradient_scale = 1.0 / self.td_step
        for i in range(self.td_step):
            action = tf.one_hot(self.action[:, i], 36)
            action = tf.reshape(action, (-1, 6, 6, 1,))
            conditioned_state = tf.concat((hidden_state, action), axis=-1)
            hidden_state, reward = self.dynamic_network(conditioned_state)
            policy_logits, value = self.policy_network(hidden_state)
            hidden_state = scale_gradient(hidden_state, 0.5)

            l = MSE(reward, self.target_reward[:, i])
            l += tf.math.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                logits=value, labels=self.target_value[:, i + 1]))
            l += tf.math.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                logits=policy_logits, labels=self.target_policy[:, i + 1]))
            loss += scale_gradient(l, gradient_scale)

        for weights in self.full_model.get_weights():
            loss += self.weight_decay * tf.nn.l2_loss(weights)
        self.loss = loss
        return self.optimizer.minimize(loss)

    def conver_value(self, target_value):
        # MSE in board games, cross entropy between categorical values in Atari.
        targets = np.zeros((target_value.shape + ((self.value_support_size * 2) + 1,)))
        batch_size = targets.shape[0]
        td_size = targets.shape[1]
        for i in range(batch_size):
            sqrt_value = np.sign(target_value[i]) * (np.sqrt(np.abs(target_value[i]) + 1) - 1) + 0.001 * target_value[i]
            floor_value = np.floor(sqrt_value).astype(int)
            rest = sqrt_value - floor_value
            index = floor_value.astype(int) + self.value_support_size
            targets[i, range(td_size), index] = 1 - rest
            targets[i, range(td_size), index + 1] = rest

        return targets

    def _value_transform(self, value_support):
        """
        The value is obtained by first computing the expected value from the discrete support.
        Second, the inverse transform is then apply (the square function).
        """
        value = self._softmax(value_support)
        value = np.dot(value, range(-self.value_support_size, self.value_support_size + 1))
        value = np.sign(value) * (
            ((np.sqrt(1 + 4 * 0.001 * (np.abs(value) + 1 + 0.001)) - 1) / (2 * 0.001)) ** 2 - 1
        )
        return np.asscalar(value)


def down_sample(input_tensor):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor

    # Returns
        Output tensor for the module.
    """
    convlayer = Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(input_tensor)
    for _ in range(2):
        convlayer = residual_block(convlayer, 128)

    convlayer = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same')(convlayer)
    for _ in range(3):
        convlayer = residual_block(convlayer, 256)

    convlayer = layers.AveragePooling2D(pool_size=(2, 2))(convlayer)
    for _ in range(3):
        convlayer = residual_block(convlayer, 256)

    convlayer = layers.AveragePooling2D(pool_size=(2, 2))(convlayer)
    return convlayer


def residual_block(input_tensor, filter, kernel_size=3):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        filters: output channel num
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
    # Returns
        Output tensor for the block.
    """
    x = layers.Conv2D(filter, kernel_size, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filter, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x
