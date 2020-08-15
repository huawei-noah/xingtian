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

import time
import numpy as np
import tensorflow as tf
from xt.model.tf_compat import K, Conv2D, Dense, \
    Flatten, Input, Lambda, Model, Activation

from xt.model.model import XTModel
from xt.model.ppo.default_config import LR, LOSS_CLIPPING, ENTROPY_LOSS, BATCH_SIZE
from xt.util.common import import_config
from xt.framework.register import Registers


@Registers.model
class PpoCnn(XTModel):
    """docstring for ActorNetwork."""
    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        super().__init__(model_info)

    def create_model(self, model_info):
        state_input = Input(shape=self.state_dim, name='state_input', dtype='uint8')
        state_input_1 = Lambda(layer_function)(state_input)
        convlayer = Conv2D(32, (8, 8), strides=(4, 4), activation='relu', padding='valid')(state_input_1)
        convlayer = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='valid')(convlayer)
        convlayer = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='valid')(convlayer)
        flattenlayer = Flatten()(convlayer)
        denselayer = Dense(256, activation='relu', name='dense_1')(flattenlayer)
        out_actions = Dense(self.action_dim, activation='softmax', name='output_actions_raw')(denselayer)
        out_value = Dense(1, name='output_value')(denselayer)
        model = Model(inputs=[state_input], outputs=[out_actions, out_value])

        self.build_graph(np.uint8, model)

        return model

    def build_graph(self, intput_type, model):
        # pylint: disable=W0201
        self.infer_state = tf.placeholder(intput_type, name="infer_input",
                                          shape=(None, ) + tuple(self.state_dim))
        self.state = tf.placeholder(intput_type, name="input",
                                    shape=(None, ) + tuple(self.state_dim))
        self.adv = tf.placeholder(tf.float32, name="adv",
                                  shape=(None, 1))
        self.old_p = tf.placeholder(tf.float32, name="old_p",
                                    shape=(None, self.action_dim))
        self.old_v = tf.placeholder(tf.float32, name="old_v",
                                    shape=(None, 1))
        self.out_p, self.out_v = model(self.state)
        self.infer_p, self.infer_v = model(self.infer_state)

        self.target_v = tf.placeholder(tf.float32, name="target_value",
                                       shape=(None, 1))
        self.target_p = tf.placeholder(tf.float32, name="target_policy",
                                       shape=(None, self.action_dim))

        loss = 0.5 * value_loss(self.target_v, self.out_v, self.old_v)
        loss += ppo_loss(self.adv, self.old_p, self.target_p, self.out_p)
        self.loss = loss

        self.trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss)
        grads, var = zip(*grads_and_var)

        max_grad_norm = .5
        grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads_and_var = list(zip(grads, var))

        self.train_op = self.trainer.apply_gradients(grads_and_var)
        self.sess.run(tf.initialize_all_variables())

    def train(self, state, label):
        with self.graph.as_default():
            K.set_session(self.sess)
            nbatch_train = BATCH_SIZE
            nbatch = state[0].shape[0]

            inds = np.arange(nbatch)
            loss_val = []
            start_time = time.time()
            for _ in range(4):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]

                    feed_dict = {self.state: state[0][mbinds],
                                 self.adv: state[1][mbinds],
                                 self.old_p: state[2][mbinds],
                                 self.old_v: state[3][mbinds],
                                 self.target_p: label[0][mbinds],
                                 self.target_v: label[1][mbinds],}
                    ret_value = self.sess.run([self.train_op, self.loss], feed_dict)

                    loss_val.append(np.mean(ret_value[1]))

            return np.mean(loss_val)

    def predict(self, state):
        """
        Do predict use the latest model.
        """
        with self.graph.as_default():
            K.set_session(self.sess)
            feed_dict = {self.infer_state: state}
            return self.sess.run([self.infer_p, self.infer_v], feed_dict)

def layer_function(x):
    """ normalize data """
    return K.cast(x, dtype='float32') / 255.

def value_loss(target_v, out_v, old_v):
    vpredclipped = old_v + tf.clip_by_value(out_v - old_v, -LOSS_CLIPPING, LOSS_CLIPPING)
    # Unclipped value
    vf_losses1 = tf.square(out_v - target_v)
    # Clipped value
    vf_losses2 = tf.square(vpredclipped - target_v)

    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

    return vf_loss

def ppo_loss(adv, old_p, target_p, out_p):
    """loss for ppo"""
    neglogpac = -target_p * tf.log(out_p + 1e-10)
    old_neglog = -target_p * tf.log(old_p + 1e-10)
    ratio = tf.exp(old_neglog - neglogpac)

    pg_losses = -adv * ratio
    pg_losses2 = -adv * tf.clip_by_value(ratio, 1.0 - LOSS_CLIPPING, 1.0 + LOSS_CLIPPING)
    pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))

    entropy = tf.reduce_mean(-out_p * tf.log(out_p + 1e-10))
    return pg_loss - ENTROPY_LOSS * entropy
