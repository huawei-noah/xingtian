"""Action distribution with mindspore"""
import numpy as np
from xt.model.ms_compat import ms, Cast, ReduceSum, ReduceMax, SoftmaxCrossEntropyWithLogits, Tensor
from mindspore import ops
import mindspore.nn.probability.distribution as msd
from mindspore import Parameter
from mindspore.common.initializer import initializer


class DiagGaussianDist(msd.Normal):
    """Build Diagonal Gaussian distribution, each vector represented one distribution."""

    def __init__(self, size):
        self.size = size
        self.reduce_sum = ReduceSum(keep_dims=True)
        self.log = ops.Log()
        self.shape = ops.Shape()
        self.square = ops.Square()
        self.Normal = ops.StandardNormal()

    def init_by_param(self, param):
        self.param = param
        self.mean, self.log_std = ops.split(self.param, axis=-1, output_num=2)
        self.std = ops.exp(self.log_std)

    def flatparam(self):
        return self.param

    def sample_dtype(self):
        return ms.float32

    def _log_prob(self, x, mean, sd):
        log_sd = self.log(sd)
        neglog_prob = 0.5 * self.log(2.0 * np.pi) * Cast()((self.shape(x)[-1]), ms.float32) + \
                      0.5 * self.reduce_sum(self.square((x - mean) / sd), axis=-1) + \
                      self.reduce_sum(log_sd, axis=-1)
        return -neglog_prob

    def mode(self):
        return self.mean

    def _entropy(self, sd):
        log_sd = self.log(sd)
        return self.reduce_sum(log_sd + 0.5 * (self.log(2.0 * np.pi) + 1.0), axis=-1)

    def kl(self, other):
        assert isinstance(other, DiagGaussianDist), 'Distribution type not match.'
        reduce_sum = ReduceSum(keep_dims=True)
        return reduce_sum(
            (ops.square(self.std) + ops.square(self.mean - other.mean)) / (2.0 * ops.square(other.std)) +
            other.log_std - self.log_std - 0.5,
            axis=-1)

    def _sample(self, mean, sd):
        return mean + sd * self.normal(self.shape(mean), dtype=ms.float32)


class CategoricalDist(msd.Categorical):

    def __init__(self, size):
        super(CategoricalDist, self).__init__()
        self.size = size
        self.OneHot = ops.OneHot()
        self.softmax_cross = ops.SoftmaxCrossEntropyWithLogits()
        self.reduce_max = ReduceMax(keep_dims=True)
        self.reduce_sum = ReduceSum(keep_dims=True)
        self.exp = ops.Exp()
        self.log = ops.Log()
        self.squeeze = ops.Squeeze()
        self.random_categorical = ops.RandomCategorical(dtype=ms.int32)
        self.expand_dims = ops.ExpandDims()

    def init_by_param(self, logits):
        self.logits = logits
        return

    def flatparam(self):
        return self.logits

    def sample_dtype(self):
        return ms.int32

    def _log_prob(self, x, logits):
        on_value, off_value = Tensor(1.0, ms.float32), Tensor(0.0, ms.float32)
        x = self.OneHot(x, self.size, on_value, off_value)
        loss, dlogits = self.softmax_cross(logits, x)
        return -self.expand_dims(loss, -1)

    def _entropy(self, logits):
        rescaled_logits = logits - self.reduce_max(logits, -1)
        exp_logits = self.exp(rescaled_logits)

        z = self.reduce_sum(exp_logits, -1)
        p = exp_logits / z
        return self.reduce_sum(p * (self.log(z) - rescaled_logits), -1)

    def kl(self, other):
        assert isinstance(other, CategoricalDist), 'Distribution type not match.'
        reduce_max = ReduceMax(keep_dims=True)
        reduce_sum = ReduceSum(keep_dims=True)
        rescaled_logits_self = self.logits - reduce_max(self.logits, -1)
        rescaled_logits_other = other.logits - reduce_max(other.logits, -1)
        exp_logits_self = ops.exp(rescaled_logits_self)
        exp_logits_other = ops.exp(rescaled_logits_other)
        z_self = reduce_sum(exp_logits_self, -1)
        z_other = reduce_sum(exp_logits_other, -1)
        p = exp_logits_self / z_self
        return reduce_sum(p * (rescaled_logits_self - ops.log(z_self) - rescaled_logits_other + ops.log(z_other)),
                          -1)

    def _sample(self, logits):
        # u = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        # return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1, output_type=tf.int32)
        return logits.random_categorical(1, dtype=ms.int32).squeeze(-1)


def make_dist(ac_type, ac_dim):
    if ac_type == 'Categorical':
        return CategoricalDist(ac_dim)
    elif ac_type == 'DiagGaussian':
        return DiagGaussianDist(ac_dim)
    else:
        raise NotImplementedError