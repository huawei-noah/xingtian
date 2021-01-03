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
"""
Implement the impala cnn network with tensorflow.

The Implement of Vtrace_loss refers to deepmind/scalable_agent.
https://github.com/deepmind/scalable_agent

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

See https://arxiv.org/abs/1802.01561 for the full paper.
"""

import numpy as np

from functools import partial
import xt.model.impala.vtrace as vtrace
from tensorflow.python.util import deprecation
from zeus.common.util.register import Registers
from xt.model import XTModel
from xt.model.impala.default_config import GAMMA, LR
from xt.model.tf_compat import (
    DTYPE_MAP,
    AdamOptimizer,
    Conv2D,
    Flatten,
    Lambda,
    Saver,
    global_variables_initializer,
    linear_cosine_decay,
    tf,
)
from xt.model.atari_model import get_atari_filter
from xt.model.tf_utils import TFVariables, restore_tf_variable
from xt.model.model_utils import state_transform, custom_norm_initializer
from zeus.common.util.common import import_config
from absl import logging

deprecation._PRINT_DEPRECATION_WARNINGS = False


@Registers.model
class ImpalaCnnOpt(XTModel):
    """Docstring for ActorNetwork."""

    def __init__(self, model_info):
        model_config = model_info.get("model_config", dict())
        import_config(globals(), model_config)
        self.dtype = DTYPE_MAP.get(model_info.get("default_dtype", "float32"))
        self.input_dtype = model_info.get("input_dtype", "float32")
        self.sta_mean = model_info.get("state_mean", 0.)
        self.sta_std = model_info.get("state_std", 255.)

        self._transform = partial(state_transform,
                                  mean=self.sta_mean,
                                  std=self.sta_std,
                                  input_dtype=self.input_dtype)

        self.state_dim = model_info["state_dim"]
        self.action_dim = model_info["action_dim"]
        self.filter_arch = get_atari_filter(self.state_dim)

        # lr schedule with linear_cosine_decay
        self.lr_schedule = model_config.get("lr_schedule", None)
        self.opt_type = model_config.get("opt_type", "adam")
        self.lr = None

        self.ph_state = None
        self.ph_adv = None
        self.out_actions = None
        self.pi_logic_outs, self.baseline = None, None

        # placeholder for behavior policy logic outputs
        self.ph_bp_logic_outs = None
        self.ph_actions = None
        self.ph_dones = None
        self.ph_rewards = None
        self.loss, self.optimizer, self.train_op = None, None, None

        self.grad_norm_clip = model_config.get("grad_norm_clip", 40.0)
        self.sample_batch_steps = model_config.get("sample_batch_step", 50)

        self.saver = None
        self.explore_paras = None
        self.actor_var = None  # store weights for agent

        super().__init__(model_info)

    def create_model(self, model_info):
        self.ph_state = tf.placeholder(self.input_dtype,
                                       shape=(None, *self.state_dim),
                                       name="state_input")

        with tf.variable_scope("explore_agent"):
            state_input = Lambda(self._transform)(self.ph_state)
            last_layer = state_input

            for (out_size, kernel, stride) in self.filter_arch[:-1]:
                last_layer = Conv2D(
                    out_size,
                    (kernel, kernel),
                    strides=(stride, stride),
                    activation="relu",
                    padding="same",
                )(last_layer)

            # last convolution
            (out_size, kernel, stride) = self.filter_arch[-1]
            convolution_layer = Conv2D(
                out_size,
                (kernel, kernel),
                strides=(stride, stride),
                activation="relu",
                padding="valid",
            )(last_layer)

            self.pi_logic_outs = tf.squeeze(
                Conv2D(self.action_dim, (1, 1), padding="same")(convolution_layer),
                axis=[1, 2],
            )

            baseline_flat = Flatten()(convolution_layer)
            self.baseline = tf.squeeze(
                tf.layers.dense(
                    inputs=baseline_flat,
                    units=1,
                    activation=None,
                    kernel_initializer=custom_norm_initializer(0.01),
                ),
                1,
            )
            self.out_actions = tf.squeeze(
                tf.multinomial(self.pi_logic_outs, num_samples=1, output_dtype=tf.int32),
                1,
                name="out_action",
            )

        # create learner
        self.ph_bp_logic_outs = tf.placeholder(self.dtype, shape=(None, self.action_dim),
                                               name="ph_b_logits")

        self.ph_actions = tf.placeholder(tf.int32, shape=(None, ), name="ph_action")
        self.ph_dones = tf.placeholder(tf.bool, shape=(None, ), name="ph_dones")
        self.ph_rewards = tf.placeholder(self.dtype, shape=(None, ), name="ph_rewards")

        # Split the tensor into batches at known episode cut boundaries.
        # [batch_count * batch_step] -> [batch_step, batch_count]
        batch_step = self.sample_batch_steps

        def split_batches(tensor, drop_last=False):
            batch_count = tf.shape(tensor)[0] // batch_step
            reshape_tensor = tf.reshape(
                tensor,
                tf.concat([[batch_count, batch_step], tf.shape(tensor)[1:]], axis=0),
            )

            # swap B and T axes
            res = tf.transpose(
                reshape_tensor,
                [1, 0] + list(range(2, 1 + int(tf.shape(tensor).shape[0]))),
            )

            if drop_last:
                return res[:-1]
            return res

        self.loss = vtrace_loss(
            bp_logic_outs=split_batches(self.ph_bp_logic_outs, drop_last=True),
            tp_logic_outs=split_batches(self.pi_logic_outs, drop_last=True),
            actions=split_batches(self.ph_actions, drop_last=True),
            discounts=split_batches(tf.cast(~self.ph_dones, tf.float32) * GAMMA, drop_last=True),
            rewards=split_batches(tf.clip_by_value(self.ph_rewards, -1, 1), drop_last=True),
            values=split_batches(self.baseline, drop_last=True),
            bootstrap_value=split_batches(self.baseline)[-1],
        )

        global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        if self.opt_type == "adam":
            if self.lr_schedule:
                learning_rate = self._get_lr(global_step)
            else:
                learning_rate = LR
            optimizer = AdamOptimizer(learning_rate)
        elif self.opt_type == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(LR, decay=0.99, epsilon=0.1, centered=True)
        else:
            raise KeyError("invalid opt_type: {}".format(self.opt_type))

        grads_and_vars = optimizer.compute_gradients(self.loss)

        # global norm
        grads, var = zip(*grads_and_vars)
        grads, _ = tf.clip_by_global_norm(grads, self.grad_norm_clip)
        clipped_gvs = list(zip(grads, var))

        self.train_op = optimizer.apply_gradients(clipped_gvs, global_step=global_step)

        # fixme: help to show the learning rate among training processing
        self.lr = optimizer._lr

        self.actor_var = TFVariables(self.out_actions, self.sess)

        self.sess.run(global_variables_initializer())

        self.explore_paras = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope="explore_agent")

        self.saver = Saver({t.name: t for t in self.explore_paras}, max_to_keep=self.max_to_keep)

        return True

    def _get_lr(self, global_step, decay_step=20000.):
        """Make decay learning rate."""
        lr_schedule = self.lr_schedule
        if len(lr_schedule) != 2:
            logging.warning("Need 2 elements in lr_schedule!\n, "
                            "likes [[0, 0.01], [20000, 0.000001]]")
            logging.fatal("lr_schedule invalid: {}".format(lr_schedule))

        if lr_schedule[0][0] != 0:
            logging.info("lr_schedule[0][1] could been init learning rate")

        learning_rate = linear_cosine_decay(lr_schedule[0][1],
                                            global_step, decay_step,
                                            beta=lr_schedule[1][1] / float(decay_step))

        return learning_rate

    def train(self, state, label):
        """Train with sess.run."""
        bp_logic_outs, actions, dones, rewards = label
        with self.graph.as_default():
            _, loss = self.sess.run(
                [self.train_op, self.loss],
                feed_dict={
                    self.ph_state: state,
                    self.ph_bp_logic_outs: bp_logic_outs,
                    self.ph_actions: actions,
                    self.ph_dones: dones,
                    self.ph_rewards: rewards,
                },
            )
        return loss

    def predict(self, state):
        """
        Do predict use the newest model.

        :param: state
        :return: action_logits, action_val, value
        """
        with self.graph.as_default():
            feed_dict = {self.ph_state: state}
            return self.sess.run([self.pi_logic_outs, self.baseline, self.out_actions],
                                 feed_dict)

    def save_model(self, file_name):
        """Save model without meta graph."""
        ck_name = self.saver.save(self.sess, save_path=file_name, write_meta_graph=False)
        return ck_name

    def load_model(self, model_name, by_name=False):
        """Load model with inference variables."""
        restore_tf_variable(self.sess, self.explore_paras, model_name)

    def set_weights(self, weights):
        """Set weight with memory tensor."""
        with self.graph.as_default():
            self.actor_var.set_weights(weights)

    def get_weights(self):
        """Get weights."""
        with self.graph.as_default():
            return self.actor_var.get_weights()


def calc_baseline_loss(advantages):
    """Calculate the baseline loss."""
    return 0.5 * tf.reduce_sum(tf.square(advantages))


def calc_entropy_loss(logic_outs):
    """Calculate entropy loss."""
    pi = tf.nn.softmax(logic_outs)
    log_pi = tf.nn.log_softmax(logic_outs)
    entropy_per_step = tf.reduce_sum(-pi * log_pi, axis=-1)
    return -tf.reduce_sum(entropy_per_step)


def calc_pi_loss(logic_outs, actions, advantages):
    """Calculate policy gradient loss."""
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=actions, logits=logic_outs)
    advantages = tf.stop_gradient(advantages)
    pg_loss_per_step = cross_entropy * advantages
    return tf.reduce_sum(pg_loss_per_step)


def vtrace_loss(
        bp_logic_outs, tp_logic_outs, actions,
        discounts, rewards, values, bootstrap_value):
    """
    Compute vtrace loss for impala algorithm.

    :param bp_logic_outs: behaviour_policy_logic_outputs
    :param tp_logic_outs: target_policy_logic_outputs
    :param actions:
    :param discounts:
    :param rewards:
    :param values:
    :param bootstrap_value:
    :return: total loss
    """
    with tf.device("/cpu"):
        value_of_state, pg_advantages = vtrace.from_logic_outputs(
            behaviour_policy_logic_outputs=bp_logic_outs,
            target_policy_logic_outputs=tp_logic_outs,
            actions=actions,
            discounts=discounts,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
        )

    pi_loss = calc_pi_loss(tp_logic_outs, actions, pg_advantages)
    val_loss = calc_baseline_loss(value_of_state - values)
    entropy_loss = calc_entropy_loss(tp_logic_outs)

    return pi_loss + 0.5 * val_loss + 0.01 * entropy_loss
