"""Build CNN and MLP network model architecture."""

import numpy as np
from xt.model.tf_compat import K, Conv2D, Dense, Flatten, Input, Lambda, Model
from xt.model.tf_utils import norm_initializer


def get_cnn_backbone(state_dim, act_dim, hidden_sizes, activation,
                     filter_arches, vf_share_layers=True, summary=False):
    """Get CNN backbone."""
    state_input_raw = Input(shape=state_dim, name='obs')
    state_input = Lambda(layer_function)(state_input_raw)

    if vf_share_layers:
        conv_layer = build_conv_layers(state_input, filter_arches, activation, 'shared')
        flatten_layer = Flatten()(conv_layer)
        dense_layer = bulid_mlp_layers(flatten_layer, hidden_sizes, activation, 'shared')
        pi_latent = Dense(act_dim, activation=None, name='pi_latent')(dense_layer)
        out_value = Dense(1, activation=None, name='output_value')(dense_layer)
    else:
        conv_layer_pi = build_conv_layers(state_input, filter_arches, activation, 'pi')
        conv_layer_v = build_conv_layers(state_input, filter_arches, activation, 'v')
        flatten_layer_pi = Flatten()(conv_layer_pi)
        flatten_layer_v = Flatten()(conv_layer_v)
        dense_layer_pi = bulid_mlp_layers(flatten_layer_pi, hidden_sizes, activation, 'pi')
        dense_layer_v = bulid_mlp_layers(flatten_layer_v, hidden_sizes, activation, 'v')
        pi_latent = Dense(act_dim, activation=None, name='pi_latent')(dense_layer_pi)
        out_value = Dense(1, activation=None, name='output_value')(dense_layer_v)

    model = Model(inputs=[state_input_raw], outputs=[pi_latent, out_value])
    if summary:
        model.summary()

    return model


def build_conv_layers(input_layer, filter_arches, activation, prefix=''):
    conv_layer = input_layer
    for i, filter_arch in enumerate(filter_arches):
        filters, kernel_size, strides = filter_arch
        conv_layer = Conv2D(filters, kernel_size, strides, activation=activation, padding='valid',
                            name="{}_conv_layer_{}".format(prefix, i))(conv_layer)
    return conv_layer


def bulid_mlp_layers(input_layer, hidden_sizes, activation, prefix=''):
    output_layer = input_layer
    for i, hidden_size in enumerate(hidden_sizes):
        output_layer = \
            Dense(hidden_size, activation=activation, name='{}_hidden_mlp_{}'.format(prefix, i))(output_layer)
    return output_layer


def get_cnn_default_settings(kind):
    """Get default setting for mlp model."""
    if kind == 'hidden_sizes':
        return [256]
    elif kind == 'activation':
        return 'relu'
    else:
        raise KeyError('unknown type: {}'.format(kind))


def get_default_filters(shape):
    """Get default model set for atari environments."""
    shape = list(shape)
    if len(shape) != 3:
        raise ValueError('Without default architecture for obs shape {}'.format(shape))
    # (out_size, kernel, stride)
    filters_84x84 = [
        [32, (8, 8), (4, 4)],
        [32, (4, 4), (2, 2)],
        [64, (3, 3), (1, 1)]
    ]
    filters_42x42 = [
        [32, (4, 4), (2, 2)],
        [32, (4, 4), (2, 2)],
        [64, (3, 3), (1, 1)]
    ]
    if shape[:2] == [84, 84]:
        return filters_84x84
    elif shape[:2] == [42, 42]:
        return filters_42x42
    else:
        filters = []
        input_w, input_h = shape[:2]
        flat_flag_w, flat_flag_h = False, False
        num_filters = 16
        while not flat_flag_w or not flat_flag_h:
            filter_w, stride_w, flat_flag_w = _infer_stride_and_kernel(input_w, flat_flag_w)
            filter_h, stride_h, flat_flag_h = _infer_stride_and_kernel(input_h, flat_flag_h)
            filters.append((num_filters, (filter_w, filter_h), (stride_w, stride_h)))
            num_filters *= 2
            input_w = input_w // stride_w
            input_h = input_h // stride_h


def _infer_stride_and_kernel(size, flat_flag):
    if flat_flag:
        return 1, 1, True

    if size <= 8:
        return 3, 1, True
    elif size <= 64:
        return 5, 2, False
    else:
        power = int(np.floor(np.log2(size)))
        stride = 2 ** power
        return 2 * stride + 1, stride, False


def _infer_same_padding_size(old_size, stride):
    new_size = old_size // stride
    if new_size * stride == old_size:
        return new_size
    else:
        return new_size + 1


def layer_function(x):
    """Normalize data."""
    return K.cast(x, dtype='float32') / 255.
