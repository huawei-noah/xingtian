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
"""Build PPO algorithm."""
from absl import logging
import numpy as np

from xt.algorithm import Algorithm
from zeus.common.util.register import Registers
from zeus.common.util.common import import_config


@Registers.algorithm
class PPO(Algorithm):
    """Build PPO algorithm."""

    def __init__(self, model_info, alg_config, **kwargs):
        """
        Create Algorithm instance.

        Will create their model within the `__init__`.
        :param model_info:
        :param alg_config:
        :param kwargs:
        """
        import_config(globals(), alg_config)
        super().__init__(
            alg_name=kwargs.get('name') or 'ppo',
            model_info=model_info['actor'],
            alg_config=alg_config
        )

        self._init_train_list()
        self.async_flag = False  # fixme: refactor async_flag

        if model_info.get('finetune_weight'):
            self.actor.load_model(model_info['finetune_weight'], by_name=True)
            logging.info('load finetune weight: {}'.format(model_info['finetune_weight']))

    def _init_train_list(self):
        self.obs = list()
        self.behavior_action = list()
        self.old_logp = list()
        self.adv = list()
        self.old_v = list()
        self.target_v = list()

    def train(self, **kwargs):
        """Train PPO Agent."""
        obs = np.concatenate(self.obs)
        behavior_action = np.concatenate(self.behavior_action)
        old_logp = np.concatenate(self.old_logp)
        adv = np.concatenate(self.adv)
        old_v = np.concatenate(self.old_v)
        target_v = np.concatenate(self.target_v)

        # adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        loss = self.actor.train([obs], [behavior_action, old_logp, adv, old_v, target_v])

        self._init_train_list()
        return loss

    def prepare_data(self, train_data, **kwargs):
        self.obs.append(train_data['cur_state'])
        self.behavior_action.append(train_data['action'])
        self.old_logp.append(train_data['logp'])
        self.adv.append(train_data['adv'])
        self.old_v.append(train_data['old_value'])
        self.target_v.append(train_data['target_value'])

    def predict(self, state):
        """Overwrite the predict function, owing to the special input."""
        if not isinstance(state, (list, tuple)):
            state = state.reshape((1,) + state.shape)
        else:
            state = list(map(lambda x: x.reshape((1,) + x.shape), state))
            state = np.vstack(state)
        pred = self.actor.predict(state)
        return pred
