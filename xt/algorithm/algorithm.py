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
"""Build Algorithm base class."""

import os
import numpy as np

from absl import logging
from xt.model import model_builder
from xt.algorithm.alg_utils import DefaultAlgDistPolicy

AGENT_PREFIX = "agent"
MODEL_PREFIX = "actor"
ZFILL_LENGTH = 5


class Algorithm(object):
    """
    Build base class for Algorithm.

    These must contains more than one model.
    """

    buff = None
    actor = None

    def __init__(self, alg_name, model_info, alg_config=None, **kwargs):
        """
        Use the model info to create a algorithm.

        :param alg_name:
        :param model_info: model_info["actor"]
        :param alg_config:
        """
        self.actor = model_builder(model_info)
        self.state_dim = model_info.get("state_dim")
        self.action_dim = model_info.get("action_dim")
        self.train_count = 0
        self.alg_name = alg_name
        self.alg_config = alg_config
        self.model_info = model_info

        self.async_flag = True
        # set default weights map, make compatibility to single agent
        self._weights_map = self.update_weights_map()

        # trainable state
        self._train_ready = True

        # train property
        self._prepare_times_per_train = alg_config.get(
            "prepare_times_per_train",
            alg_config["instance_num"] * alg_config["agent_num"],
        )
        self.dist_model_policy = DefaultAlgDistPolicy(alg_config["instance_num"],
                                                      prepare_times=self._prepare_times_per_train)

        self.learning_starts = alg_config.get("learning_starts", 0)

        self._train_per_checkpoint = alg_config.get("train_per_checkpoint", 1)
        logging.debug("train/checkpoint: {}".format(self.train_per_checkpoint))

        self.if_save_model = alg_config.get("save_model", False)
        self.save_interval = alg_config.get("save_interval", 500)

    def if_save(self, train_count):
        if not self.if_save_model:
            return False
        if train_count % self.save_interval == 0:
            return True

    @staticmethod
    def update_weights_map(agent_in_group="agent_0", agent_in_env="agent_0"):
        """
        Set custom weights map on there.

        e.g.
            {"agent_id": {"prefix": "actor", "name":"YOUR/PATH/TO/MODEL/FILE.h5"}}
            firstly, find the prefix,
            second, find name of the model file.

            All the agents will share an same model as default.

        Note:
        ----
            If user need update the map Dynamically,
            Call this function after train process within the `self.train()`
        """
        return {}

    def prepare_data(self, train_data, **kwargs):
        """
        Prepare the data for train function.

        Contains:
            1) put training data to queue/buff,
            2) processing the data for user's special train operation
        Each new algorithm must implement this function.
        """
        raise NotImplementedError

    @property
    def prepare_data_times(self):
        """Unify the prepare data time for each train operation."""
        return self._prepare_times_per_train

    def predict(self, state):
        """
        Predict action.

        The api will call the keras.model.predict as default,
        if the inputs is different from the normal state,
        You need overwrite this function.
        """
        inputs = state.reshape((1, ) + state.shape)
        out = self.actor.predict(inputs)

        return np.argmax(out)

    def train_ready(self, elapsed_episode, **kwargs):
        """
        Support custom train logic.

        :return: train ready flag
        """
        # we set train ready as default
        self._train_ready = True
        # if self.async_flag and elapsed_episode < self.learning_starts:
        #     self._train_ready = False
        # if use buffer, check the buffer.size
        if getattr(self, "buff") and self.learning_starts > 0:
            # logging.debug("buff vs start: {} vs {}".format(
            #     self.buff.size(), self.learning_starts))
            if self.buff.size() < self.learning_starts:
                self._train_ready = False

        return self._train_ready

    def train(self, **kwargs):
        """
        Train algorithm.

        Each new algorithm must implement this function.
        """
        raise NotImplementedError

    def checkpoint_ready(self, train_count, **kwargs):
        """Support custom checkpoint logic after training."""
        self._train_ready = False
        if train_count % self.train_per_checkpoint == 0:
            return True

        return False

    @property
    def train_per_checkpoint(self):
        return self._train_per_checkpoint

    @train_per_checkpoint.setter
    def train_per_checkpoint(self, interval):
        self._train_per_checkpoint = interval

    def save(self, model_path, model_index):
        """
        Save api call `keras.model.save_model` function to save model weight.

        To support save multi model within the algorithm,
            eg. [actor_00xx1.h5, critic_00xx2.h5]
        return name used a list type
        And, save the actor model as default.
        :param model_path: model save path
        :param model_index: the index will been zfill with 5.
        :return: a list of the name with saved model.
        """
        model_name = self.actor.save_model(
            os.path.join(model_path, "actor_{}".format(str(model_index).zfill(ZFILL_LENGTH))))
        return [model_name]

    def restore(self, model_name=None, model_weights=None):
        """
        Restore the model with the priority: model_weight > model_name.

        Owing to actor.set_weights would be faster than load model from disk.

        if user used multi model in one algorithm,
        they need overwrite this function.
        impala will use weights, not model name
        """
        if model_weights is not None:
            self.actor.set_weights(model_weights)
        else:
            logging.debug("{} load model: {}".format(self.alg_name, model_name))
            self.actor.load_model(model_name)

    def get_weights(self):
        """Get the actor model weights as default."""
        return self.actor.get_weights()

    def set_weights(self, weights):
        """Set the actor model weights as default."""
        return self.actor.set_weights(weights)

    @property
    def weights_map(self):
        return self._weights_map

    @weights_map.setter
    def weights_map(self, map_info):
        """
        Set weights map.

        Here, User also could set some policy for the weight map
        :param map_info:
        :return:
        """
        self._weights_map = map_info

    def shutdown(self):
        """Shutdown algorithm."""
        pass
