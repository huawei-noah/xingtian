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

from xt.model.model_utils_ms import ACTIVATION_MAP_MS, get_cnn_backbone_ms,\
    get_cnn_default_settings_ms, get_default_filters_ms
from xt.model.ppo.default_config import CNN_SHARE_LAYERS
from xt.model.ppo.ppo_ms import PPOMS
from zeus.common.util.register import Registers
from xt.model.ms_utils import MSVariables


@Registers.model
class PpoCnnMS(PPOMS):
    """Build PPO CNN network."""

    def __init__(self, model_info):
        model_config = model_info.get('model_config')

        self.vf_share_layers = model_config.get(
            'VF_SHARE_LAYERS', CNN_SHARE_LAYERS)
        self.hidden_sizes = model_config.get(
            'hidden_sizes', get_cnn_default_settings_ms('hidden_sizes'))
        activation = model_config.get(
            'activation', get_cnn_default_settings_ms('activation'))
        try:
            self.activation = ACTIVATION_MAP_MS[activation]
        except KeyError:
            raise KeyError('activation {} not implemented.'.format(activation))

        super().__init__(model_info)

    def create_model(self, model_info):
        filter_arches = get_default_filters_ms(self.state_dim)
        net = get_cnn_backbone_ms(
            self.state_dim,
            self.action_dim,
            self.hidden_sizes,
            self.activation,
            filter_arches,
            self.vf_share_layers,
            self.verbose,
            dtype=self.input_dtype)
        self.actor_var = MSVariables(net)
        return net
