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
from xt.model.ppo.default_config import LR, BATCH_SIZE, CRITIC_LOSS_COEF,\
    ENTROPY_LOSS, LOSS_CLIPPING, MAX_GRAD_NORM, NUM_SGD_ITER, SUMMARY, VF_CLIP
from xt.model.ms_dist import make_dist
from zeus.common.util.common import import_config
from zeus.common.util.register import Registers
from xt.model.ms_compat import Cell, TrainOneStepCell, LossBase, ReduceMean, ReduceSum, Tensor, Adam
from xt.model.ms_compat import Depend, value_and_grad, clip_by_global_norm, Minimum, Maximum, Exp, Square, clip_by_value, DynamicLossScaleUpdateCell, FixedLossScaleUpdateCell
from xt.model.model_ms import XTModel_MS
from xt.model.ms_utils import MSVariables
import mindspore as ms
from xt.model.dqn.dqn_cnn_ms import MyTrainOneStepCell

ms.set_context(runtime_num_threads=30)
@Registers.model
class PPOMS(XTModel_MS):

    class PPOPredictPolicy(Cell):
        def __init__(self, net, dist):
            super(PPOMS.PPOPredictPolicy, self).__init__()
            self.network = net
            self.dist = dist

        def construct(self, state):
            pi_latent, v_out = self.network(state)
            action = self.dist.sample(pi_latent)
            logp = self.dist.log_prob(action, pi_latent)
            return action, logp, v_out

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
        self.critic_loss_coef = model_config.get(
            'CRITIC_LOSS_COEF', CRITIC_LOSS_COEF)
        self.ent_coef = Tensor(model_config.get('ENTROPY_LOSS', ENTROPY_LOSS))
        self.clip_ratio = Tensor(model_config.get(
            'LOSS_CLIPPING', LOSS_CLIPPING))
        self._max_grad_norm = model_config.get('MAX_GRAD_NORM', MAX_GRAD_NORM)
        self.num_sgd_iter = model_config.get('NUM_SGD_ITER', NUM_SGD_ITER)
        self.verbose = model_config.get('SUMMARY', SUMMARY)
        self.vf_clip = Tensor(model_config.get('VF_CLIP', VF_CLIP))
        self.dist = make_dist(self.action_type, self.action_dim)
        self.amsgrad = model_config.get('USE_AMSGRAD', False)
        super().__init__(model_info)
        self.predict_net = self.PPOPredictPolicy(self.model, self.dist)
        adam = Adam(params=self.predict_net.trainable_params(), learning_rate=self._lr, use_amsgrad=self.amsgrad, use_locking=True)
        loss_fn = WithLossCell(self.critic_loss_coef, self.clip_ratio, self.ent_coef, self.vf_clip)
        forward_fn = NetWithLoss(self.model, loss_fn, self.dist)
        device_target = ms.get_context("device_target")
        if device_target == 'Ascend':
            manager = FixedLossScaleUpdateCell(loss_scale_value=2**14)
            self.train_net = MyTrainOneStepCell(forward_fn, adam, manager, grad_clip=True, clipnorm=self._max_grad_norm)
        elif device_target == "GPU" or device_target == "CPU":
            self.train_net = myTrainOneStepCell(forward_fn, optimizer=adam, max_grad_norm=self._max_grad_norm)
        else:
            raise Exception("Target error, GPU or Ascend is supported.")
        self.predict_net.compile(ms.Tensor(np.zeros((1, 84, 84, 4))).astype(ms.float32))

    def predict(self, state):
        """Predict state."""
        state = Tensor.from_numpy(state)
        action, logp, v_out = self.predict_net(state)
        action = action.asnumpy()
        logp = logp.asnumpy()
        v_out = v_out.asnumpy()
        return action, logp, v_out

    def train(self, state, label):
        nbatch = state[0].shape[0]
        inds = np.arange(nbatch)
        loss_val = []
        for _ in range(self.num_sgd_iter):
            np.random.shuffle(inds)
            for start in range(0, nbatch, self._batch_size):
                end = start + self._batch_size
                mbinds = inds[start:end]
                state_ph = Tensor.from_numpy(state[0][mbinds])
                behavior_action_ph = Tensor.from_numpy(label[0][mbinds])
                old_logp_ph = Tensor.from_numpy(label[1][mbinds])
                adv_ph = Tensor.from_numpy(label[2][mbinds]).astype(ms.float32)
                old_v_ph = Tensor.from_numpy(label[3][mbinds])
                target_v_ph = Tensor.from_numpy(label[4][mbinds]).astype(ms.float32)
                loss = self.train_net(state_ph, adv_ph, old_logp_ph, behavior_action_ph, target_v_ph, old_v_ph)
                loss = loss.asnumpy()
                loss_val.append(np.mean(loss))
        self.actor_var = MSVariables(self.predict_net)
        return np.mean(loss_val)


class myTrainOneStepCell(TrainOneStepCell):
    def __init__(self, network, optimizer, max_grad_norm, sens=1.0):
        super(myTrainOneStepCell, self).__init__(network, optimizer, sens)
        self.sens = sens
        self.depend = Depend()
        self.max_grad_norm = max_grad_norm
        self.grad_fn = value_and_grad(self.network, grad_position=None, weights=self.weights)

    def construct(self, *inputs):
        loss, grads = self.grad_fn(*inputs)
        grads = clip_by_global_norm(grads, self.max_grad_norm)
        grads = self.grad_reducer(grads)
        loss = self.depend(loss, self.optimizer(grads))
        return loss


class NetWithLoss(Cell):
    def __init__(self, net, loss_fn, dist):
        super(NetWithLoss, self).__init__(auto_prefix=False)
        self.net = net
        self._loss_fn = loss_fn
        self.dist = dist

    def construct(self, state_ph, adv_ph, old_logp_ph, behavior_action, target_v, old_v_ph):
        pi_latent, v_out = self.net(state_ph)
        ent = self.dist.entropy(pi_latent)
        action_log_prob = self.dist.log_prob(behavior_action, pi_latent)
        loss = self._loss_fn(action_log_prob, ent, adv_ph, old_logp_ph, target_v, v_out, old_v_ph)
        return loss


class WithLossCell(LossBase):
    def __init__(self, critic_loss_coef, clip_ratio, ent_coef, val_clip):
        super(WithLossCell, self).__init__()
        self.reduce_mean = ReduceMean(keep_dims=True)
        self.critic_loss_coef = critic_loss_coef
        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef
        self.val_clip = val_clip
        self.minimum = Minimum()
        self.maximum = Maximum()
        self.exp = Exp()
        self.square = Square()

    def construct(self, action_log_prob, ent, adv, old_log_p, target_v, out_v, old_v):
        ratio = self.exp(action_log_prob - old_log_p)

        surr_loss_1 = ratio * adv
        surr_loss_2 = clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
        surr_loss = self.reduce_mean(self.minimum(surr_loss_1, surr_loss_2))
        ent = self.reduce_mean(ent)

        actor_loss = -surr_loss - self.ent_coef * ent

        vf_losses1 = self.square(out_v - target_v)
        val_pred_clipped = old_v + clip_by_value(out_v - old_v, -self.val_clip, self.val_clip)
        vf_losses2 = self.square(val_pred_clipped - target_v)

        critic_loss = 0.5 * self.reduce_mean(self.maximum(vf_losses1, vf_losses2))
        loss = actor_loss + self.critic_loss_coef * critic_loss
        return loss
