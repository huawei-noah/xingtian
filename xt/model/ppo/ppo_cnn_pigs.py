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

from xt.model.model_utils import get_cnn_backbone, get_default_filters
from xt.model.ppo.ppo_cnn import PpoCnn
from zeus.common.util.register import Registers


@Registers.model
class PigPpoCnn(PpoCnn):
    """Build PPO CNN network for CatchPigs"""

    def create_model(self, model_info):
        dtype = 'float32'
        filter_arches = get_default_filters(self.state_dim)
        model = get_cnn_backbone(self.state_dim, self.action_dim, self.hidden_sizes, self.activation, filter_arches,
                            self.vf_share_layers, self.verbose, dtype=dtype)
        self.build_graph(dtype, model)
        return model
