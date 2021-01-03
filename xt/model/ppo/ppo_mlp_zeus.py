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
import os
import time
import numpy as np
from xt.model.tf_compat import tf

from xt.model.model_zeus import XTModelZeus
from xt.model.ppo.default_config import \
    LR, BATCH_SIZE, CRITIC_LOSS_COEF, ENTROPY_LOSS, LOSS_CLIPPING, MAX_GRAD_NORM, NUM_SGD_ITER, SUMMARY, VF_CLIP
from zeus.common.util.common import import_config

from zeus.common.util.register import Registers
from zeus import set_backend
from zeus.trainer_api import Trainer
from zeus.common.class_factory import ClassFactory, ClassType
from zeus.trainer.modules.conf.loss import LossConfig
from zeus.trainer.modules.conf.optim import OptimConfig
from zeus.modules.module import Module
from zeus.modules.operators.ops import Relu, Linear, Conv2d, View, softmax, Lambda
from zeus.modules.connections import Sequential

set_backend(backend='tensorflow', device_category='GPU')


@Registers.model
class PpoMlpZeus(XTModelZeus):
    """Docstring for ActorNetwork."""

    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']

        self.action_type = model_config.get('action_type')
        self.num_sgd_iter = model_config.get('NUM_SGD_ITER', NUM_SGD_ITER)
        super().__init__(model_info)

    def create_model(self, model_info):
        zeus_model = PpoMlpNet(state_dim=self.state_dim, action_dim=self.action_dim)

        LossConfig.type = 'ppo_loss'
        OptimConfig.type = 'Adam'
        OptimConfig.params.update({'lr': LR})

        loss_input = dict()
        loss_input['inputs'] = [{"name": "input_state", "type": "float32", "shape": self.state_dim}]
        loss_input['labels'] = [{"name": "old_v", "type": "float32", "shape": 1}]
        loss_input['labels'].append({"name": "target_v", "type": "float32", "shape": 1})
        loss_input['labels'].append({"name": "old_p", "type": "float32", "shape": self.action_dim})
        loss_input['labels'].append({"name": "target_p", "type": "int32", "shape": 1})
        loss_input['labels'].append({"name": "adv", "type": "float32", "shape": 1})

        model = Trainer(model=zeus_model, lazy_build=False, loss_input=loss_input)
        return model

    def train(self, state, label):
        nbatch_train = BATCH_SIZE
        nbatch = state[0].shape[0]

        inds = np.arange(nbatch)
        loss_val = []
        start_time = time.time()
        for _ in range(self.num_sgd_iter):
            # Randomize the indexes
            np.random.shuffle(inds)
            # 0 to batch_size with batch_train_size step
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = inds[start:end]

                inputs = [state[0][mbinds]]
                action = np.expand_dims(label[0][mbinds], -1)
                labels = [label[3][mbinds], label[4][mbinds], label[1][mbinds],
                          action, label[2][mbinds]]

                loss = self.model.train(inputs, labels)
                loss_val.append(np.mean(loss))

        return np.mean(loss_val)

    def predict(self, state):
        """Predict state."""
        prob, logit, value = self.model.predict(state)
        action = np.random.choice(self.action_dim, p=np.nan_to_num(prob[0]))
        action = np.array([action])

        return [action, logit, value]


class PpoMlpNet(Module):
    """Create DQN net with FineGrainedSpace."""
    def __init__(self, **descript):
        """Create layers."""
        super().__init__()
        state_dim = descript.get("state_dim")
        action_dim = descript.get("action_dim")

        self.fc1 = Sequential(Linear(64, 64), Linear(64, action_dim))
        self.fc2 = Sequential(Linear(64, 64), Linear(64, 1))

    def __call__(self, inputs):
        """Override compile function, conect models into a seq."""
        logit = self.fc1(inputs)
        value = self.fc2(inputs)
        prob = softmax(logit)
        return prob, logit, value


@ClassFactory.register(ClassType.LOSS, 'ppo_loss')
def ppo_loss_zeus(logits, labels):
    out_p, out_logits, out_v = logits
    old_v, target_v, old_logits, action, adv = labels

    loss = CRITIC_LOSS_COEF * value_loss(target_v, out_v, old_v)
    loss += actor_loss_with_entropy(adv, old_logits, action, out_logits)
    return loss


def value_loss(target_v, out_v, old_v):
    """Compute value loss for PPO."""
    vpredclipped = old_v + tf.clip_by_value(out_v - old_v, -VF_CLIP, VF_CLIP)
    vf_losses1 = tf.square(out_v - target_v)
    vf_losses2 = tf.square(vpredclipped - target_v)
    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
    return vf_loss


def actor_loss_with_entropy(adv, old_logits, behavior_action, out_logits):
    """Calculate actor loss with entropy."""
    old_log_p = neglog_prob(behavior_action, old_logits)
    action_log_prob = neglog_prob(behavior_action, out_logits)
    ratio = tf.exp(action_log_prob - old_log_p)

    surr_loss_1 = ratio * adv
    surr_loss_2 = tf.clip_by_value(ratio, 1.0 - LOSS_CLIPPING, 1.0 + LOSS_CLIPPING) * adv
    surr_loss = tf.reduce_mean(tf.minimum(surr_loss_1, surr_loss_2))

    ent = entropy(out_logits)
    ent = tf.reduce_mean(ent)

    return -surr_loss - ENTROPY_LOSS * ent


def neglog_prob(x, logits):
    size = logits.shape[-1]
    x = tf.one_hot(x, size)
    neglogp = tf.nn.softmax_cross_entropy_with_logits_v2(labels=x, logits=logits)
    return -tf.expand_dims(neglogp, axis=-1)


def entropy(logits):
    rescaled_logits = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
    exp_logits = tf.exp(rescaled_logits)
    z = tf.reduce_sum(exp_logits, axis=-1, keepdims=True)
    p = exp_logits / z
    return tf.reduce_sum(p * (tf.log(z) - rescaled_logits), axis=-1, keepdims=True)
