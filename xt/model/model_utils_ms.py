"""Retain model utils."""

import numpy as np

from xt.model.ms_compat import ms, SequentialCell, Dense, Conv2d, Flatten,\
                                get_activation, Cell
from mindspore._checkparam import twice

ACTIVATION_MAP_MS = {
    'sigmoid': 'sigmoid',
    'tanh': 'tanh',
    'softsign': 'softsign',
    'softplus': 'softplus',
    'relu': 'relu',
    'leakyrelu': 'leakyrelu',
    'elu': 'elu',
    'selu': 'seLU',
    'hswish': 'hswish',
    'gelu': 'gelu'
}


def cal_shape(input_shape, kernel_size, stride):
    kernel_size = twice(kernel_size)
    stride = twice(stride)
    return tuple(
        (v - kernel_size[i]) // stride[i] + 1 for i,
        v in enumerate(input_shape))


class MlpBackbone(Cell):
    def __init__(self, state_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.dense_layer_pi = bulid_mlp_layers_ms(
            state_dim[-1], hidden_sizes, activation)
        self.dense_pi = Dense(
            hidden_sizes[-1], act_dim, weight_init="XavierUniform")
        self.dense_layer_v = bulid_mlp_layers_ms(
            state_dim[-1], hidden_sizes, activation)
        self.dense_out = Dense(
            hidden_sizes[-1], 1, weight_init="XavierUniform")

    def construct(self, x):
        if x.dtype == ms.float64:
            x = x.astype(ms.float32)
        pi_latent = self.dense_layer_pi(x)
        pi_latent = self.dense_pi(pi_latent)
        out_value = self.dense_layer_v(x)
        out_value = self.dense_out(out_value)

        return [pi_latent, out_value]


class MlpBackboneShare(Cell):
    def __init__(self, state_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.dense_layer_share = bulid_mlp_layers_ms(
            state_dim[-1], hidden_sizes, activation
        )
        self.dense_pi = Dense(
            hidden_sizes[-1], act_dim, weight_init="XavierUniform")
        self.dense_out = Dense(
            hidden_sizes[-1], 1, weight_init="XavierUniform")

    def construct(self, x):
        if x.dtype == ms.float64:
            x = x.astype(ms.float32)
        share = self.dense_layer_share(x)
        pi_latent = self.dense_pi(share)
        out_value = self.dense_out(share)

        return [pi_latent, out_value]


class CnnBackbone(Cell):
    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_sizes,
        activation,
        filter_arches,
        dtype,
    ):
        super().__init__()
        self.dtype = dtype
        self.conv_layer_pi = build_conv_layers_ms(
            state_dim[-1], filter_arches, activation)
        self.flatten_layer = Flatten()
        height, width = state_dim[-3], state_dim[-2]
        filters = 1
        for filters, kernel_size, strides in filter_arches:
            height, width = cal_shape((height, width), kernel_size, strides)
        dim = height * width * filters
        self.dense_layer_pi = bulid_mlp_layers_ms(
            dim, hidden_sizes, activation)
        self.dense_pi = Dense(
            hidden_sizes[-1], act_dim, weight_init="XavierUniform")
        self.conv_layer_v = build_conv_layers_ms(
            state_dim[-1], filter_arches, activation)
        self.dense_layer_v = bulid_mlp_layers_ms(dim, hidden_sizes, activation)
        self.dense_v = Dense(hidden_sizes[-1], 1, weight_init="XavierUniform")

    def construct(self, x):
        x = x.transpose((0, 3, 1, 2))
        if self.dtype == "uint8":
            x = layer_function_ms(x)
        pi_latent = self.conv_layer_pi(x)
        pi_latent = self.flatten_layer(pi_latent)
        pi_latent = self.dense_layer_pi(pi_latent)
        pi_latent = self.dense_pi(pi_latent)
        out_value = self.conv_layer_v(x)
        out_value = self.flatten_layer(out_value)
        out_value = self.dense_layer_v(out_value)
        out_value = self.dense_v(out_value)

        return [pi_latent, out_value]


class CnnBackboneShare(Cell):
    def __init__(
        self,
        state_dim,
        act_dim,
        hidden_sizes,
        activation,
        filter_arches,
        dtype,
    ):
        super().__init__()
        self.dtype = dtype
        self.conv_layer_share = build_conv_layers_ms(
            state_dim[-1], filter_arches, activation
        )
        self.flatten_layer = Flatten()
        height, width = state_dim[-3], state_dim[-2]
        filters = 1
        for filters, kernel_size, strides in filter_arches:
            height, width = cal_shape((height, width), kernel_size, strides)
        dim = height * width * filters
        self.dense_layer_share = bulid_mlp_layers_ms(
            dim, hidden_sizes, activation)
        self.dense_pi = Dense(
            hidden_sizes[-1], act_dim, weight_init="XavierUniform")
        self.dense_v = Dense(hidden_sizes[-1], 1, weight_init="XavierUniform")

    def construct(self, x):
        x = x.transpose((0, 3, 1, 2))
        if self.dtype == "uint8":
            x = layer_function_ms(x)
        share = self.conv_layer_share(x)
        share = self.flatten_layer(share)
        share = self.dense_layer_share(share)
        pi_latent = self.dense_pi(share)
        out_value = self.dense_v(share)
        return [pi_latent, out_value]


def get_mlp_backbone_ms(
    state_dim,
    act_dim,
    hidden_sizes,
    activation,
    vf_share_layers=False,
    summary=False,
    dtype='float32',
):
    """Get mlp backbone."""
    if dtype != "float32":
        raise ValueError(
            'dtype: {} not supported automatically, please implement it yourself'.format(
                dtype
            )
        )
    if not vf_share_layers:
        return MlpBackbone(state_dim, act_dim, hidden_sizes, activation)

    return MlpBackboneShare(state_dim, act_dim, hidden_sizes, activation)


def get_cnn_backbone_ms(
    state_dim,
    act_dim,
    hidden_sizes,
    activation,
    filter_arches,
    vf_share_layers=True,
    summary=False,
    dtype='uint8',
):
    """Get CNN backbone."""
    if dtype != "uint8" and dtype != "float32":
        raise ValueError(
            'dtype: {} not supported automatically, \
                please implement it yourself'.format(
                dtype
            )
        )
    if vf_share_layers:
        return CnnBackboneShare(
            state_dim,
            act_dim,
            hidden_sizes,
            activation,
            filter_arches,
            dtype,
        )
    return CnnBackbone(
        state_dim,
        act_dim,
        hidden_sizes,
        activation,
        filter_arches,
        dtype,
    )


def bulid_mlp_layers_ms(input_size, hidden_sizes, activation):
    build_block = SequentialCell()
    for hidden_size in hidden_sizes:
        build_block.append(
            Dense(
                input_size,
                hidden_size,
                activation=activation,
                weight_init="XavierUniform",
            )
        )
        input_size = hidden_size
    return build_block


def build_conv_layers_ms(input_size, filter_arches, activation):
    build_block = SequentialCell()
    for filters, kernel_size, strides in filter_arches:
        build_block.append(
            Conv2d(
                input_size,
                filters,
                kernel_size,
                strides,
                pad_mode="valid",
                has_bias=True,
                weight_init="XavierUniform",
            )
        )
        build_block.append(get_activation(activation))
        input_size = filters
    return build_block


def get_mlp_default_settings_ms(kind):
    """Get default setting for mlp model."""
    if kind == "hidden_sizes":
        return [64, 64]
    elif kind == "activation":
        return "tanh"
    else:
        raise KeyError("unknown type: {}".format(kind))


def get_cnn_default_settings_ms(kind):
    """Get default setting for mlp model."""
    if kind == 'hidden_sizes':
        return [512]
    elif kind == 'activation':
        return 'relu'
    else:
        raise KeyError('unknown type: {}'.format(kind))


def get_default_filters_ms(shape):
    """Get default model set for atari environments."""
    shape = list(shape)
    if len(shape) != 3:
        raise ValueError(
            'Without default architecture for obs shape {}'.format(shape))
    filters_84x84 = [[32, (8, 8), (4, 4)], [32, (4, 4), (2, 2)], [
        64, (3, 3), (1, 1)]]
    filters_42x42 = [[32, (4, 4), (2, 2)], [32, (4, 4), (2, 2)], [
        64, (3, 3), (1, 1)]]
    filters_15x15 = [[32, (5, 5), (1, 1)], [64, (3, 3), (1, 1)], [
        64, (3, 3), (1, 1)]]
    if shape[:2] == [84, 84]:
        return filters_84x84
    elif shape[:2] == [42, 42]:
        return filters_42x42
    elif shape[:2] == [15, 15]:
        return filters_15x15
    else:
        filters = []
        input_w, input_h = shape[:2]
        flat_flag_w, flat_flag_h = False, False
        num_filters = 16
        while not flat_flag_w or not flat_flag_h:
            filter_w, stride_w, flat_flag_w = _infer_stride_and_kernel_ms(
                input_w, flat_flag_w
            )
            filter_h, stride_h, flat_flag_h = _infer_stride_and_kernel_ms(
                input_h, flat_flag_h
            )
            filters.append(
                (num_filters, (filter_w, filter_h), (stride_w, stride_h)))
            num_filters *= 2
            input_w = input_w // stride_w
            input_h = input_h // stride_h
        return filters


def _infer_stride_and_kernel_ms(size, flat_flag):
    if flat_flag or size <= 3:
        return 1, 1, True

    if size <= 8:
        return 3, 1, True
    elif size <= 64:
        return 5, 2, False
    else:
        power = int(np.floor(np.log2(size)))
        stride = 2**power
        return 2 * stride + 1, stride, False


def layer_function_ms(x):
    """Normalize data."""
    return x.astype(ms.float32) / 255.0

