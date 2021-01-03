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
from xt.model.model_zeus import XTModelZeus

from zeus.common.util.common import import_config
from zeus.common.util.register import Registers
from zeus import set_backend, register_zeus
from zeus.trainer.trainer_api import Trainer
from zeus.common.class_factory import ClassFactory, ClassType
from zeus.trainer.modules.conf.loss import LossConfig
from zeus.trainer.modules.conf.optim import OptimConfig

set_backend(backend='tensorflow', device_category='GPU')
register_zeus('tensorflow')


@Registers.model
class DqnZeus(XTModelZeus):
    """Docstring for DqnMlp."""
    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)
        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']

        super().__init__(model_info)

    def create_model(self, model_info):
        """Create Deep-Q network."""
        model_config = model_info['model_config']
        zeus_model_name = model_config['zeus_model_name']

        model_cls = ClassFactory.get_cls(ClassType.NETWORK, zeus_model_name)
        zeus_model = model_cls(state_dim=self.state_dim, action_dim=self.action_dim)
        # zeus_model = DqnMlpNet(state_dim=self.state_dim, action_dim=self.action_dim)

        LossConfig.type = model_config['loss']
        OptimConfig.type = model_config['optim']
        OptimConfig.params.update({'lr': model_config['lr']})

        loss_input = dict()
        loss_input['inputs'] = [{"name": "input_state", "type": "float32", "shape": self.state_dim}]
        loss_input['labels'] = [{"name": "target_value", "type": "float32", "shape": self.action_dim}]
        model = Trainer(model=zeus_model, backend='tensorflow', device='GPU',
                        loss_input=loss_input, lazy_build=False)
        return model
