"""Action distribution with mindspore"""
import numpy as np
from xt.model.ms_compat import ms, Cast, ReduceSum, ReduceMax, Tensor
from mindspore import ops
from mindspore import ms_class


@ms_class
class ActionDist:
    """Build base action distribution."""

    def init_by_param(self, param):
        raise NotImplementedError

    def flatparam(self):
        raise NotImplementedError

    def sample(self, repeat):
        """Sample action from this distribution."""
        raise NotImplementedError

    def sample_dtype(self):
        raise NotImplementedError

    def get_shape(self):
        return self.flatparam().shape.as_list()

    @property
    def shape(self):
        return self.get_shape()

    def __getitem__(self, idx):
        return self.flatparam()[idx]

    def neglog_prob(self, x, logits):
        raise NotImplementedError

    def log_prob(self, x, logits):
        """Calculate the log-likelihood."""
        return -self.neglog_prob(x, logits)

    def mode(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError


class DiagGaussianDist(ActionDist):
    """Build Diagonal Gaussian distribution, each vector represented one distribution."""

    def __init__(self, size):
        self.size = size
        self.reduce_sum = ReduceSum(keep_dims=True)
        self.log = ops.Log()
        self.shape = ops.Shape()
        self.square = ops.Square()
        self.normal = ops.StandardNormal()
        self.cast = Cast()

    def init_by_param(self, param):
        self.param = param
        self.mean, self.log_std = ops.split(self.param, axis=-1, output_num=2)
        self.std = ops.exp(self.log_std)

    def flatparam(self):
        return self.param

    def sample_dtype(self):
        return ms.float32

    def log_prob(self, x, mean, sd=None):
        if sd is not None:
            log_sd = self.log(sd)
            neglog_prob = 0.5 * self.log(2.0 * np.pi) * self.cast((self.shape(x)[-1]), ms.float32) + \
                0.5 * self.reduce_sum(self.square((x - mean) / sd), axis=-1) + \
                self.reduce_sum(log_sd, axis=-1)
        else:
            neglog_prob = 0.5 * self.log(2.0 * np.pi) * self.cast((self.shape(
                x)[-1]), ms.float32) + 0.5 * self.reduce_sum(self.square((x - mean) / sd), axis=-1)
        return -neglog_prob

    def mode(self):
        return self.mean

    def entropy(self, mean, sd=None):
        if sd is not None:
            log_sd = self.log(sd)
            return self.reduce_sum(
                log_sd + 0.5 * (self.log(2.0 * np.pi) + 1.0), axis=-1)
        return 0.5 * (self.log(2.0 * np.pi) + 1.0)

    def kl(self, other):
        assert isinstance(
            other, DiagGaussianDist), 'Distribution type not match.'
        reduce_sum = ReduceSum(keep_dims=True)
        return reduce_sum((self.square(self.std) +
                           self.square(self.mean - other.mean)) /
                          (2.0 * self.square(other.std)) +
                          other.log_std - self.log_std - 0.5, axis=-1)

    def sample(self, mean, sd=None):
        if sd is not None:
            return mean + sd * self.normal(self.shape(mean), dtype=ms.float32)
        return mean + self.normal(self.shape(mean), dtype=ms.float32)


class CategoricalDist(ActionDist):

    def __init__(self, size):
        self.size = size
        self.oneHot = ops.OneHot()
        self.softmax_cross = ops.SoftmaxCrossEntropyWithLogits()
        self.reduce_max = ReduceMax(keep_dims=True)
        self.reduce_sum = ReduceSum(keep_dims=True)
        self.exp = ops.Exp()
        self.log = ops.Log()
        self.expand_dims = ops.ExpandDims()
        self.random_categorical = ops.RandomCategorical(dtype=ms.int32)
        self.on_value, self.off_value = Tensor(
            1.0, ms.float32), Tensor(0.0, ms.float32)

    def init_by_param(self, logits):
        self.logits = logits

    def flatparam(self):
        return self.logits

    def sample_dtype(self):
        return ms.int32

    def log_prob(self, x, logits):
        x = self.oneHot(x, self.size, self.on_value, self.off_value)
        loss, _ = self.softmax_cross(logits, x)
        return -self.expand_dims(loss, -1)

    def entropy(self, logits):
        rescaled_logits = logits - self.reduce_max(logits, -1)
        exp_logits = self.exp(rescaled_logits)

        z = self.reduce_sum(exp_logits, -1)
        p = exp_logits / z
        return self.reduce_sum(p * (self.log(z) - rescaled_logits), -1)

    def kl(self, other):
        assert isinstance(
            other, CategoricalDist), 'Distribution type not match.'
        reduce_max = ReduceMax(keep_dims=True)
        reduce_sum = ReduceSum(keep_dims=True)
        rescaled_logits_self = self.logits - reduce_max(self.logits, axis=-1)
        rescaled_logits_other = other.logits - \
            reduce_max(other.logits, axis=-1)
        exp_logits_self = self.exp(rescaled_logits_self)
        exp_logits_other = self.exp(rescaled_logits_other)
        z_self = reduce_sum(exp_logits_self, axis=-1)
        z_other = reduce_sum(exp_logits_other, axis=-1)
        p = exp_logits_self / z_self
        return reduce_sum(p *
                          (rescaled_logits_self -
                           self.log(z_self) -
                              rescaled_logits_other +
                              self.log(z_other)), axis=-1)

    def sample(self, logits):
        return self.random_categorical(logits, 1, 0).squeeze(-1)


def make_dist(ac_type, ac_dim):
    if ac_type == 'Categorical':
        return CategoricalDist(ac_dim)
    elif ac_type == 'DiagGaussian':
        return DiagGaussianDist(ac_dim)
    else:
        raise NotImplementedError
