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

from xt.model import XTModel
from xt.model.ppo import actor_loss_with_entropy, critic_loss
from xt.model.ppo.default_config import \
    LR, BATCH_SIZE, CRITIC_LOSS_COEF, ENTROPY_LOSS, LOSS_CLIPPING, MAX_GRAD_NORM, NUM_SGD_ITER, SUMMARY, VF_CLIP
from xt.model.tf_compat import tf
from xt.model.tf_dist import make_dist
from xt.model.tf_utils import TFVariables
from zeus.common.util.common import import_config
from zeus.common.util.register import Registers


@Registers.model
class PPO(XTModel):
    """Build PPO MLP network."""
    def __init__(self, model_info):
        model_config = model_info.get('model_config')
        import_config(globals(), model_config)

        # fixme: could read action_dim&obs_dim from env.info
        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        self.input_dtype = model_info.get('input_dtype', 'float32')

        self.action_type = model_config.get('action_type')
        self._lr = model_config.get('LR', LR)
        self._batch_size = model_config.get('BATCH_SIZE', BATCH_SIZE)
        self.critic_loss_coef = model_config.get('CRITIC_LOSS_COEF', CRITIC_LOSS_COEF)
        self.ent_coef = model_config.get('ENTROPY_LOSS', ENTROPY_LOSS)
        self.clip_ratio = model_config.get('LOSS_CLIPPING', LOSS_CLIPPING)
        self._max_grad_norm = model_config.get('MAX_GRAD_NORM', MAX_GRAD_NORM)
        self.num_sgd_iter = model_config.get('NUM_SGD_ITER', NUM_SGD_ITER)
        self.verbose = model_config.get('SUMMARY', SUMMARY)
        self.vf_clip = model_config.get('VF_CLIP', VF_CLIP)

        self.dist = make_dist(self.action_type, self.action_dim)

        super().__init__(model_info)

    def build_graph(self, input_type, model):
        # pylint: disable=W0201
        self.state_ph = tf.placeholder(input_type, name='state', shape=(None, *self.state_dim))
        self.old_logp_ph = tf.placeholder(tf.float32, name='old_log_p', shape=(None, 1))
        self.adv_ph = tf.placeholder(tf.float32, name='advantage', shape=(None, 1))
        self.old_v_ph = tf.placeholder(tf.float32, name='old_v', shape=(None, 1))
        self.target_v_ph = tf.placeholder(tf.float32, name='target_value', shape=(None, 1))

        pi_latent, self.out_v = model(self.state_ph)

        if self.action_type == 'Categorical':
            self.behavior_action_ph = tf.placeholder(tf.int32, name='behavior_action', shape=(None,))
            dist_param = pi_latent
        elif self.action_type == 'DiagGaussian':
            # fixme: add input dependant log_std logic
            self.behavior_action_ph = tf.placeholder(tf.float32, name='real_action', shape=(None, self.action_dim))
            log_std = tf.get_variable('pi_logstd', shape=(1, self.action_dim), initializer=tf.zeros_initializer())
            dist_param = tf.concat([pi_latent, pi_latent * 0.0 + log_std], axis=-1)
        else:
            raise NotImplementedError(
                'action type: {} not match any implemented distributions.'.format(self.action_type))

        self.dist.init_by_param(dist_param)
        self.action = self.dist.sample()
        self.action_log_prob = self.dist.log_prob(self.action)
        self.actor_var = TFVariables([self.action_log_prob, self.out_v], self.sess)

        self.actor_loss = actor_loss_with_entropy(self.dist, self.adv_ph, self.old_logp_ph, self.behavior_action_ph,
                                                  self.clip_ratio, self.ent_coef)
        self.critic_loss = critic_loss(self.target_v_ph, self.out_v, self.old_v_ph, self.vf_clip)
        self.loss = self.actor_loss + self.critic_loss_coef * self.critic_loss
        self.train_op = self.build_train_op(self.loss)

        self.sess.run(tf.initialize_all_variables())

    def build_train_op(self, loss):
        trainer = tf.train.AdamOptimizer(learning_rate=self._lr)
        grads_and_var = trainer.compute_gradients(loss)
        grads, var = zip(*grads_and_var)
        grads, _ = tf.clip_by_global_norm(grads, self._max_grad_norm)
        return trainer.apply_gradients(zip(grads, var))

    def predict(self, state):
        """Predict state."""
        with self.graph.as_default():
            feed_dict = {self.state_ph: state}
            action, logp, v_out = self.sess.run([self.action, self.action_log_prob, self.out_v], feed_dict)
        return action, logp, v_out

    def train(self, state, label):
        with self.graph.as_default():
            nbatch = state[0].shape[0]
            inds = np.arange(nbatch)
            loss_val = []
            for _ in range(self.num_sgd_iter):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, self._batch_size):
                    end = start + self._batch_size
                    mbinds = inds[start:end]
                    feed_dict = {self.state_ph: state[0][mbinds],
                                 self.behavior_action_ph: label[0][mbinds],
                                 self.old_logp_ph: label[1][mbinds],
                                 self.adv_ph: label[2][mbinds],
                                 self.old_v_ph: label[3][mbinds],
                                 self.target_v_ph: label[4][mbinds]}
                    ret_value = self.sess.run([self.train_op, self.loss], feed_dict)
                    loss_val.append(np.mean(ret_value[1]))

            return np.mean(loss_val)
