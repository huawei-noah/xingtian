"""Build fully connected network model architecture."""

import numpy as np
from xt.model.tf_compat import Input, Model, Dense, Concatenate, tf
from xt.model.tf_utils import norm_initializer


def get_mlp_backbone(state_dim, act_dim, hidden_sizes, activation,
                     vf_share_layers=False, summary=False):
    """Get mlp backbone."""

    state_input = Input(shape=state_dim, name='obs')

    if not vf_share_layers:
        dense_layer_pi = bulid_mlp_layers(state_input, hidden_sizes, activation, 'pi')
        pi_latent = Dense(act_dim, activation=None, name='pi_latent')(dense_layer_pi)
        dense_layer_v = bulid_mlp_layers(state_input, hidden_sizes, activation, 'v')
        out_value = Dense(1, activation=None, name='output_value')(dense_layer_v)
    else:
        dense_layer = bulid_mlp_layers(state_input, hidden_sizes, activation, 'shared')
        pi_latent = Dense(act_dim, activation=None, name='pi_latent')(dense_layer)
        out_value = Dense(1, activation=None, name='output_value')(dense_layer)

    model = Model(inputs=[state_input], outputs=[pi_latent, out_value])
    if summary:
        model.summary()

    return model


def bulid_mlp_layers(input_layer, hidden_sizes, activation, prefix=''):
    output_layer = input_layer
    for i, hidden_size in enumerate(hidden_sizes):
        output_layer = \
            Dense(hidden_size, activation=activation, name='{}_hidden_mlp_{}'.format(prefix, i))(output_layer)
    return output_layer


def get_mlp_default_settings(kind):
    """Get default setting for mlp model."""
    if kind == "hidden_sizes":
        return [64, 64]
    elif kind == "activation":
        return "tanh"
    else:
        raise KeyError("unknown type: {}".format(kind))
