import numpy as np
from xt.model.ppo.default_config import \
    LR, BATCH_SIZE, CRITIC_LOSS_COEF, ENTROPY_LOSS, LOSS_CLIPPING, MAX_GRAD_NORM, NUM_SGD_ITER, SUMMARY, VF_CLIP
from xt.model.ms_dist import make_dist
from zeus.common.util.common import import_config
from zeus.common.util.register import Registers
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from xt.model.ms_compat import ms, ReduceMean, Tensor, Adam, MultitypeFuncGraph, Reciprocal
from xt.model.model_ms import XTModel_MS
from xt.model.ms_utils import MSVariables
from mindspore import nn, common, ops, ParameterTuple, Parameter
import mindspore.nn.probability.distribution as msd
from mindspore import set_context
import time
import os
import psutil
@Registers.model
class PPOMS(XTModel_MS):
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
        self.ent_coef = Tensor(model_config.get('ENTROPY_LOSS', ENTROPY_LOSS))
        self.clip_ratio = Tensor(model_config.get('LOSS_CLIPPING', LOSS_CLIPPING))
        self._max_grad_norm = model_config.get('MAX_GRAD_NORM', MAX_GRAD_NORM)
        self.num_sgd_iter = model_config.get('NUM_SGD_ITER', NUM_SGD_ITER)
        self.verbose = model_config.get('SUMMARY', SUMMARY)
        self.vf_clip = Tensor(model_config.get('VF_CLIP', VF_CLIP))
        
        self.dist = make_dist(self.action_type, self.action_dim)
        
        super().__init__(model_info)
        
        '''创建训练网络'''
        adam = Adam(params=self.model.trainable_params(), learning_rate=0.0005,use_amsgrad=True)
        loss_fn = WithLossCell(self.critic_loss_coef,self.clip_ratio, self.ent_coef, self.vf_clip)
        forward_fn = NetWithLoss(self.model,loss_fn, self.dist, self.action_type)
        self.train_net = MyTrainOneStepCell(forward_fn, optimizer=adam, max_grad_norm=self._max_grad_norm)
        self.train_net.set_train()

    def predict(self, state):
        """Predict state."""
        self.model.set_train(False)
        
        state = Tensor(state, ms.uint8)
        pi_latent, v_out = self.model(state)
        
        if self.action_type == 'DiagGaussian':
            std = ms.common.initializer('ones', [pi_latent.shape[0], self.action_dim], ms.float32)
            self.action = self.dist.sample(pi_latent, std)
            self.logp = self.dist.log_prob(self.action, pi_latent, std)
        elif self.action_type == 'Categorical':
            
            self.action = self.dist.sample(pi_latent)
            self.logp = self.dist.log_prob(self.action, pi_latent)
            
        self.action = self.action.asnumpy()
        self.logp = self.logp.asnumpy()
        v_out = v_out.asnumpy()
        return self.action, self.logp, v_out

    def train(self, state, label):
        
        self.model.set_train(True)
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
                state_ph = Tensor(state[0][mbinds])
                behavior_action_ph = Tensor(label[0][mbinds])
                old_logp_ph = Tensor(label[1][mbinds],ms.float32)
                adv_ph = Tensor(label[2][mbinds],ms.float32)
                old_v_ph = Tensor(label[3][mbinds],ms.float32)
                target_v_ph = Tensor(label[4][mbinds],ms.float32)

                loss = self.train_net( state_ph,adv_ph, old_logp_ph,behavior_action_ph, target_v_ph, old_v_ph ).asnumpy()             
                loss_val.append(np.mean(loss))

        self.actor_var = MSVariables(self.model)
        
        return np.mean(loss_val)

class MyTrainOneStepCell(nn.TrainOneStepCell):
    def __init__(self, network, optimizer, max_grad_norm, sens=1.0):
        super(MyTrainOneStepCell, self).__init__(network, optimizer, sens)
        self.sens = sens
        self.max_grad_norm = max_grad_norm
    
    def construct(self,*inputs):
        weights = self.weights
        loss, grads = ops.value_and_grad(self.network, grad_position=None, weights=weights)(*inputs)  
        grads = ops.clip_by_global_norm(grads, self.max_grad_norm)
        grads = self.grad_reducer(grads)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss
        

class NetWithLoss(nn.Cell):
    def __init__(self, net,loss_fn, dist,action_type):
        super(NetWithLoss, self).__init__(auto_prefix=False)
        self.net = net
        self._loss_fn = loss_fn
        self.action_type = action_type
        self.dist = dist
        self.reduce_sum = ops.ReduceSum(keep_dims=True)
        self.split = ops.Split()
        self.concat = ops.Concat()
        self.exp = ops.Exp()
        self.log = ops.Log()
    
    def construct(self,state_ph,adv_ph, old_logp_ph,behavior_action, target_v, old_v_ph ):
        pi_latent, v_out = self.net(state_ph)
        if self.action_type == 'DiagGaussian':
            log_std = ms.common.initializer('zeros', [1, self.action_dim], ms.float32)
            dist_param = self.concat([pi_latent, pi_latent * 0.0 + log_std], axis=-1)
            mean, log_std = self.split(dist_param, axis=-1, output_num=2)
            sd = self.exp(log_std)
            ent = self.dist.entropy(sd)
            action_log_prob = self.dist.log_prob(behavior_action,mean,sd)
        else:
            ent = self.dist.entropy(pi_latent)
            action_log_prob = self.dist.log_prob(behavior_action,pi_latent)
        loss = self._loss_fn(action_log_prob,ent, adv_ph, old_logp_ph, target_v, v_out, old_v_ph)
        return loss

class WithLossCell(nn.LossBase):
    def __init__(self, critic_loss_coef,clip_ratio,ent_coef,val_clip):
        super(WithLossCell, self).__init__()
        self.reduce_mean = ReduceMean(keep_dims=True)
        self.critic_loss_coef = critic_loss_coef
        self.Mul = ops.Mul()
        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef
        self.val_clip = val_clip
        self.reducesum = ops.ReduceSum(keep_dims=True)
        self.minimum = ops.Minimum()
        self.maximum = ops.Maximum()
        self.exp = ops.Exp()
        self.square = ops.Square()
        self.squeeze = ops.Squeeze()
    
    def construct(self, action_log_prob,ent,  adv, old_log_p,  target_v, out_v, old_v):
        ratio = self.exp(action_log_prob - old_log_p)

        surr_loss_1 = ratio * adv
        surr_loss_2 = ops.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
        surr_loss = self.reduce_mean(self.minimum(surr_loss_1, surr_loss_2))
        ent = self.reduce_mean(ent)

        actor_loss = -surr_loss - self.ent_coef * ent

        vf_losses1 = self.square(out_v - target_v)
        val_pred_clipped = old_v + ops.clip_by_value(out_v - old_v, -self.val_clip, self.val_clip)
        vf_losses2 = self.square(val_pred_clipped - target_v)

        critic_loss =  0.5 * self.reduce_mean(self.maximum(vf_losses1, vf_losses2))
        loss = actor_loss + self.critic_loss_coef * critic_loss
        return loss

