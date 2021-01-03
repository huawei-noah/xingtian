"""Makse transform utils for qmix algorithm."""
import numpy as np

from xt.model.tf_compat import tf


class Transform:
    """Make transform base class."""

    def transform(self, tensor):
        raise NotImplementedError

    def infer_output_info(self, vshape_in, dtype_in):
        raise NotImplementedError


class OneHotTf(Transform):
    """Make transform with tensorflow."""

    def __init__(self, out_dim, dtype=tf.float32):
        self.out_dim = out_dim
        self.dtype = dtype

    def transform(self, tensor):
        tensor_indices = np.squeeze(tensor, axis=-1)
        one_hot = tf.one_hot(
            indices=tensor_indices,
            depth=self.out_dim,
            on_value=1.0,
            off_value=0.0,
            axis=-1,
            dtype=self.dtype,
        )
        return one_hot

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), self.dtype


class OneHotNp(Transform):
    """Make transform with numpy."""

    def __init__(self, out_dim, dtype=np.float):
        self.out_dim = out_dim
        self.dtype = dtype

    def transform(self, tensor):
        if not isinstance(tensor, np.ndarray):
            tensor = np.array(tensor)
        # print(np.array(targets).reshape(-1))
        res = np.eye(self.out_dim)[tensor.reshape(-1)]
        targets = res.reshape([*(tensor.shape[:-1]), self.out_dim])
        return targets.astype(self.dtype)

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), self.dtype


def test_func():
    """Check with func between numpy and tf."""
    output_dim = 11

    data = [[[[10], [2], [5], [2], [10]]]]

    # # tf
    # oh = OneHotTf(output_dim)
    # r = oh.transform(a)
    # with tf.Session() as sess:
    #     r = sess.run(r)
    #     print(r, r.shape)

    # np
    np_onehot = OneHotNp(output_dim)
    np_ret = np_onehot.transform(data)
    print(np_ret)


if __name__ == "__main__":
    test_func()
