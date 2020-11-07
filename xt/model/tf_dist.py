"""Action distribution with tensorflow"""
import numpy as np
from xt.model.tf_compat import tf


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

    def neglog_prob(self, x):
        raise NotImplementedError

    def log_prob(self, x):
        """Calculate the log-likelihood."""
        return -self.neglog_prob(x)

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

    def init_by_param(self, param):
        self.param = param
        self.mean, self.log_std = tf.split(self.param, num_or_size_splits=2, axis=-1)
        self.std = tf.exp(self.log_std)

    def flatparam(self):
        return self.param

    def sample_dtype(self):
        return tf.float32

    def neglog_prob(self, x):
        return 0.5 * np.log(2.0 * np.pi) * tf.cast((tf.shape(x)[-1]), tf.float32) + \
            0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1, keepdims=True) + \
            tf.reduce_sum(self.log_std, axis=-1, keepdims=True)

    def mode(self):
        return self.mean

    def entropy(self):
        return tf.reduce_sum(self.log_std + 0.5 * (np.log(2.0 * np.pi) + 1.0), axis=-1, keepdims=True)

    def kl(self, other):
        assert isinstance(other, DiagGaussianDist), 'Distribution type not match.'
        return tf.reduce_sum(
            (tf.square(self.std) + tf.square(self.mean - other.mean)) / (2.0 * tf.square(other.std)) +
            other.log_std - self.log_std - 0.5,
            axis=-1,
            keepdims=True)

    def sample(self, repeat=None):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean), dtype=tf.float32)


class CategoricalDist(ActionDist):

    def __init__(self, size):
        self.size = size

    def init_by_param(self, logits):
        self.logits = logits

    def flatparam(self):
        return self.logits

    def sample_dtype(self):
        return tf.int32

    def neglog_prob(self, x):
        x = tf.one_hot(x, self.size)
        neglogp = tf.nn.softmax_cross_entropy_with_logits_v2(labels=x, logits=self.logits)
        return tf.expand_dims(neglogp, axis=-1)

    def entropy(self):
        rescaled_logits = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        exp_logits = tf.exp(rescaled_logits)
        z = tf.reduce_sum(exp_logits, axis=-1, keepdims=True)
        p = exp_logits / z
        return tf.reduce_sum(p * (tf.log(z) - rescaled_logits), axis=-1, keepdims=True)

    def kl(self, other):
        assert isinstance(other, CategoricalDist), 'Distribution type not match.'
        rescaled_logits_self = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        rescaled_logits_other = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        exp_logits_self = tf.exp(rescaled_logits_self)
        exp_logits_other = tf.exp(rescaled_logits_other)
        z_self = tf.reduce_sum(exp_logits_self, axis=-1, keepdims=True)
        z_other = tf.reduce_sum(exp_logits_other, axis=-1, keepdims=True)
        p = exp_logits_self / z_self
        return tf.reduce_sum(p * (rescaled_logits_self - tf.log(z_self) - rescaled_logits_other + tf.log(z_other)),
                             axis=-1, keepdims=True)

    def sample(self):
        # u = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        # return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1, output_type=tf.int32)
        return tf.squeeze(tf.random.categorical(logits=self.logits, num_samples=1, dtype=tf.int32), axis=-1)


def make_dist(ac_type, ac_dim):
    if ac_type == 'Categorical':
        return CategoricalDist(ac_dim)
    elif ac_type == 'DiagGaussian':
        return DiagGaussianDist(ac_dim)
    else:
        raise NotImplementedError
