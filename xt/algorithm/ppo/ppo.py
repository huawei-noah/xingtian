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
"""PPO algorithm."""
import numpy as np

from xt.algorithm import Algorithm
from xt.framework.register import Registers
from xt.util.common import import_config


@Registers.algorithm
class PPO(Algorithm):
    """PPO algorithm"""

    def __init__(self, model_info, alg_config, **kwargs):
        """
        Algorithm instance, will create their model within the `__init__`.
        :param model_info:
        :param alg_config:
        :param kwargs:
        """
        import_config(globals(), alg_config)
        super(PPO, self).__init__(
            alg_name=kwargs.get("name") or "ppo",
            model_info=model_info["actor"],
            alg_config=alg_config,
        )

        self._init_train_list()
        self.async_flag = False  # fixme: refactor async_flag

        if model_info.get("finetune_weight"):
            self.actor.load_model(model_info["finetune_weight"], by_name=True)
            print("load finetune weight: ", model_info["finetune_weight"])

    def _init_train_list(self):
        self.obs = list()
        self.old_a = list()
        self.old_v = list()
        self.adv = list()
        self.target_a = list()
        self.target_v = list()

    def train(self, **kwargs):
        """ppo train process."""
        obs = np.concatenate(self.obs)
        old_a = np.concatenate(self.old_a)
        old_v = np.concatenate(self.old_v)
        adv = np.concatenate(self.adv)
        target_a = np.concatenate(self.target_a)
        target_v = np.concatenate(self.target_v)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        loss = self.actor.train([obs, adv, old_a, old_v],
                                [target_a, target_v])

        self._init_train_list()
        return loss

    def prepare_data(self, train_data, **kwargs):
        self.obs.append(train_data["cur_state"])
        self.old_a.append(train_data["action"])
        self.old_v.append(train_data["value"])
        self.adv.append(train_data["adv"])
        self.target_a.append(train_data["target_action"])
        self.target_v.append(train_data["target_value"])

    def predict(self, state):
        """overwrite the predict function, owing to the special input"""
        state = state.reshape((1,) + state.shape)
        pred = self.actor.predict(state)

        return pred
